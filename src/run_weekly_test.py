"""
Step 10: run_weekly_test.py（Bug 修复版）

职责：5 周滚动回测（Walk-forward Validation）

流程（每周独立执行）：
  ① 用 cutoff_date 之前的数据训练模型（严格无未来数据泄漏）
  ② 对 cutoff_date 当天截面预测收益分数
  ③ 选出 Top-5 股票
  ④ 用测试周真实开盘价计算实际加权收益率
  ⑤ 汇总成绩单

修复记录：
  Bug1 - 标签泄漏：训练集排除 cutoff 前 periods 个交易日
  Bug2 - index 大小写：统一转小写后再 join
  Bug3 - ffill 跨股票污染：改为按 instrument 分组 ffill
  Bug4 - turnover 缺失：重新 dump 后解决
  Bug5 - scorer 口径：统一使用开盘价
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qlib
from src.data.loader          import load
from src.data.fetcher         import fetch
from src.features.technical   import compute as compute_tech
from src.features.fundamental import compute as compute_fund
from src.features.normalizer  import normalize
from src.features.market      import compute as compute_market
from src.model.lgbm_regressor import LGBMModel
from src.strategy.top_k       import select_top_k

# ── 日志配置 ──────────────────────────────────────────────────
os.makedirs("logs",   exist_ok=True)
os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run_weekly_test.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("WeeklyTest")

""" (已注释)
# ── 5 个测试周 ────────────────────────────────────────────────
TEST_WEEKS = [
    ("第1周 (3.09-3.13)", "2026-03-06", "2026-03-09", "2026-03-13"),
    ("第2周 (3.16-3.20)", "2026-03-13", "2026-03-16", "2026-03-20"),
    ("第3周 (3.23-3.27)", "2026-03-20", "2026-03-23", "2026-03-27"),
    ("第4周 (3.30-4.03)", "2026-03-27", "2026-03-30", "2026-04-03"),
    ("第5周 (4.13-4.17)", "2026-04-10", "2026-04-13", "2026-04-17"),
]
"""

LABEL_PERIODS = 5   # 未来5个交易日
TRAIN_WINDOW  = 504  # 只用最近2年（约504个交易日）训练，聚焦近期市场规律


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def to_qlib_code(code: str) -> str:
    """6位内部格式 → qlib 小写格式：600519 → sh600519"""
    code = str(code).zfill(6)
    prefix = "sh" if code.startswith(("6", "9", "688")) else "sz"
    return f"{prefix}{code}"


def to_internal_code(qlib_code: str) -> str:
    """qlib 格式 → 6位内部格式：sh600519 → 600519"""
    return qlib_code[2:]


def lowercase_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Bug2 修复] 把 MultiIndex 中 instrument 层统一转为小写。
    确保 technical / fundamental / label 三者 join 时 index 能对齐。
    """
    inst = df.index.get_level_values("instrument").str.lower()
    dt   = df.index.get_level_values("datetime")
    df.index = pd.MultiIndex.from_arrays(
        [inst, dt], names=["instrument", "datetime"]
    )
    return df


def generate_labels(price_df: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """
    生成标签：与 score_self.py 评分口径严格对齐。

    官方评分逻辑：
        return = (测试周最后一天开盘 - 测试周第一天开盘) / 测试周第一天开盘

    对应到 cutoff 日 T：
        label[T] = open[T+5] / open[T+1] - 1

        T   日：cutoff，特征截止日，收盘后预测
        T+1 日：下周第一个交易日，开盘买入
        T+5 日：下周最后一个交易日，开盘卖出
    
    score_self.py 的计算逻辑是：
        对测试周内每只股票取 tail(5) 行，
        return = (末行开盘 - 首行开盘) / 首行开盘
    
    对应到训练 label：
        label[T] = open[T+periods] / open[T+1] - 1
        含义：在 T+1 日开盘买入，T+periods 日开盘卖出
    """
    df = price_df.sort_values(["stock_code", "date"]).copy()

    # 核心改动：close → open
    df["label"] = df.groupby("stock_code")["open"].transform(
        lambda x: x.shift(-periods) / x.shift(-1) - 1
    )

    df["instrument"] = df["stock_code"].apply(to_qlib_code)
    df = df.rename(columns={"date": "datetime"})
    df = df.set_index(["instrument", "datetime"])[["label"]]
    return df


def safe_ffill(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    [Bug3 修复] 按 instrument 分组前向填充，避免跨股票污染。
    剩余 NaN 用 0 补（截面标准化后 0 = 均值，最安全的默认值）。
    """
    df = df.copy()
    df[cols] = (
        df[cols]
        .groupby(level="instrument")
        .ffill()
    )
    df[cols] = df[cols].fillna(0)
    return df


def calculate_actual_return(
    candidates_df: pd.DataFrame,
    price_df:      pd.DataFrame,
    week_start:    str,
    week_end:      str,
    k:             int = 5,
) -> tuple:
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end)

    week_data = price_df[
        (price_df["date"] >= ws) &
        (price_df["date"] <= we)
    ].copy()
    available = set(week_data["stock_code"].unique())

    # 按排名顺序取前 k 只有数据的股票
    selected = []
    for _, row in candidates_df.iterrows():
        code = str(row["stock_code"]).zfill(6)
        if code in available:
            selected.append(code)
            if len(selected) == k:
                break
        else:
            logger.warning(
                f"    ⚠️  {code} 测试周内停牌，"
                f"启用替补（原排名第 {_ + 1} 位）"
            )

    if len(selected) < k:
        logger.warning(
            f"    ⚠️  可用股票不足 {k} 只，实际 {len(selected)} 只"
        )

    weight = round(1.0 / k, 4)
    top5 = pd.DataFrame({
        "股票代码": [str(c).zfill(6) for c in selected],
        "权重":     [weight] * len(selected),
    })

    weight_map    = dict(zip(top5["股票代码"], top5["权重"]))
    week_data_sel = week_data[
        week_data["stock_code"].isin(selected)
    ].sort_values(["stock_code", "date"])

    total_ret = 0.0
    for code in selected:
        s       = week_data_sel[week_data_sel["stock_code"] == code]
        p_start = s.iloc[0]["open"]
        p_end   = s.iloc[-1]["open"]
        ret     = (p_end - p_start) / p_start
        total_ret += weight_map[code] * ret
        logger.info(
            f"    {code}: 开盘 {p_start:.2f}→{p_end:.2f}  "
            f"收益 {ret*100:+.2f}%  权重 {weight_map[code]:.2f}"
        )

    return total_ret, top5

def generate_test_weeks(
    price_df: pd.DataFrame,
    end_date: str,
    n_weeks: int = 26,
) -> list:
    """
    从真实交易日历自动生成 TEST_WEEKS，避免手写日期踩假期。

    逻辑：
        1. 从 price_df 提取全量交易日，按自然周分组
        2. 从包含 end_date 的那周往前取 n_weeks 个完整交易周
        3. 每周的 cutoff = 上一个完整交易周的最后一个交易日

    Parameters
    ----------
    price_df : loader.py 输出的价格 DataFrame，必须有 date 列
    end_date : 最后一个测试周的结束日期，格式 'YYYY-MM-DD'
    n_weeks  : 往前取多少周，默认 26（约半年）

    Returns
    -------
    list of (week_name, cutoff, week_start, week_end)
    """
    # 提取全量交易日并按自然周分组
    all_dates = pd.Series(
        sorted(price_df["date"].unique())
    ).dt.normalize()

    # isocalendar().week 有跨年问题，改用 to_period('W')
    week_groups = all_dates.groupby(
        all_dates.dt.to_period("W")
    ).apply(list).reset_index()
    week_groups.columns = ["week_period", "trading_days"]

    # 找到包含 end_date 的那周
    end_ts = pd.Timestamp(end_date)
    week_groups["week_end_date"] = week_groups["trading_days"].apply(
        lambda days: max(days)
    )
    week_groups["week_start_date"] = week_groups["trading_days"].apply(
        lambda days: min(days)
    )

    # 只保留结束日 <= end_date 的完整交易周
    valid_weeks = week_groups[
        week_groups["week_end_date"] <= end_ts
    ].tail(n_weeks).reset_index(drop=True)

    if len(valid_weeks) < n_weeks:
        print(f"[警告] 数据不足 {n_weeks} 周，实际只有 {len(valid_weeks)} 周")

    test_weeks = []
    for i, row in valid_weeks.iterrows():
        # cutoff = 上一个完整交易周的最后一个交易日
        if i == 0:
            # 第一周往前找 cutoff：取 week_start 之前最近的交易日
            cutoff = all_dates[
                all_dates < row["week_start_date"]
            ].iloc[-1]
        else:
            cutoff = valid_weeks.loc[i - 1, "week_end_date"]

        week_start = row["week_start_date"]
        week_end   = row["week_end_date"]

        # 生成周名，格式：第N周 (M.DD-M.DD)
        week_name = (
            f"第{i+1:02d}周 "
            f"({week_start.month}.{week_start.day:02d}"
            f"-"
            f"{week_end.month}.{week_end.day:02d})"
        )

        test_weeks.append((
            week_name,
            cutoff.strftime("%Y-%m-%d"),
            week_start.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d"),
        ))

    return test_weeks

def _log_feature_ic(df_train: pd.DataFrame, y_train: pd.Series, logger) -> None:
    """
    计算每个特征与 label 的截面 Spearman IC。
    仅供诊断，短期预测 IC 普遍偏低属正常现象。
    
    关注指标：
        IC_mean  : 均值，绝对值越大越好
        ICIR     : IC均值/IC标准差，绝对值越大信号越稳定
        IC_pos%  : IC为正的比例，远离50%说明方向一致
    """
    from scipy.stats import spearmanr

    dates   = df_train.index.get_level_values("datetime").unique()
    records = []

    for col in df_train.columns:
        ics = []
        for dt in dates:
            mask  = df_train.index.get_level_values("datetime") == dt
            x_day = df_train[mask][col]
            y_day = y_train[mask]
            valid = x_day.notna() & y_day.notna()
            if valid.sum() < 10:
                continue
            ic, _ = spearmanr(x_day[valid], y_day[valid])
            if not np.isnan(ic):
                ics.append(ic)

        if not ics:
            continue

        ics   = pd.Series(ics)
        mean  = ics.mean()
        std   = ics.std()
        icir  = mean / std if std > 0 else 0
        pos   = (ics > 0).mean()

        records.append({
            "feature":  col,
            "IC_mean":  round(mean,  4),
            "IC_std":   round(std,   4),
            "ICIR":     round(icir,  3),
            "IC_pos%":  round(pos * 100, 1),
        })

    ic_df = (
        pd.DataFrame(records)
        .sort_values("IC_mean", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        f"\n{'='*62}\n"
        f"  特征 IC 统计（第1周训练集，仅供参考）\n"
        f"{'='*62}\n"
        f"{ic_df.to_string(index=False)}\n"
        f"{'='*62}"
    )

# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════

def main():

    # ── [0] 初始化 ────────────────────────────────────────────────
    logger.info("========== [初始化] ==========")
    qlib.init(provider_uri="data/qlib_bin", region="cn")

    # ── [1] 加载价格数据 ──────────────────────────────────────────
    logger.info("加载价格数据...")
    price_df = load("data/stock_data.csv")
    price_df["stock_code"] = price_df["stock_code"].astype(str).str.zfill(6)
    price_df["date"]       = pd.to_datetime(price_df["date"])

    all_codes  = sorted(price_df["stock_code"].unique().tolist())
    qlib_codes = [to_qlib_code(c) for c in all_codes]

    # 交易日历（从价格数据提取，用于 Bug1 安全截止日计算）
    all_cal_dates = sorted(price_df["date"].unique().tolist())

    logger.info(
        f"共 {len(all_codes)} 只股票，"
        f"数据范围：{all_cal_dates[0].date()} ~ {all_cal_dates[-1].date()}"
    )

    # ── 自动生成半年测试周 ────────────────────────────────────────
    TEST_WEEKS = generate_test_weeks(
        price_df=price_df,
        end_date="2026-04-17",
        n_weeks=26,
    )
    logger.info(f"自动生成测试周：共 {len(TEST_WEEKS)} 周")
    logger.info(f"  首周：{TEST_WEEKS[0][0]}  cutoff：{TEST_WEEKS[0][1]}")
    logger.info(f"  末周：{TEST_WEEKS[-1][0]}  cutoff：{TEST_WEEKS[-1][1]}")

    # ── [2] 拉取基本面数据（有缓存则秒速完成）────────────────────
    logger.info("拉取基本面数据...")
    fund_df = fetch(
        stock_codes=all_codes,
        start_date="2020-01-01",
        end_date="2026-04-17",
    )

    # ── [3] 生成完整 Label ────────────────────────────────────────
    logger.info("生成完整 Label（未来5日开盘价收益率）...")
    label_df = generate_labels(price_df, periods=LABEL_PERIODS)
    # label_df.index = (instrument小写, datetime)

    # ════════════════════════════════════════════════════════════
    # [4] 逐周回测主循环
    # ════════════════════════════════════════════════════════════
    weekly_results = []

    for week_idx, (week_name, cutoff, week_start, week_end) in \
            enumerate(TEST_WEEKS, 1):

        logger.info(f"\n{'='*55}")
        logger.info(f"  {week_name}")
        logger.info(f"  训练截止：{cutoff}  |  测试周期：{week_start} ~ {week_end}")
        logger.info(f"{'='*55}")

        cutoff_ts = pd.Timestamp(cutoff)

        # ── [Bug1 修复] 安全截止日：往前推 LABEL_PERIODS 个交易日 ──
        # 防止训练集中最后几天的 label 用到了测试周价格
        try:
            cutoff_pos  = all_cal_dates.index(cutoff_ts)
        except ValueError:
            # cutoff 是非交易日，找最近的前一个交易日
            cutoff_ts   = max(d for d in all_cal_dates if d <= cutoff_ts)
            cutoff_pos  = all_cal_dates.index(cutoff_ts)
            logger.info(f"  cutoff 调整为最近交易日：{cutoff_ts.date()}")

        if cutoff_pos < LABEL_PERIODS:
            logger.warning("  历史数据不足，跳过本周")
            continue

        # 安全截止日：排除会用到测试周价格的那几行 label
        safe_cutoff = all_cal_dates[cutoff_pos - LABEL_PERIODS]
        logger.info(
            f"  安全训练截止日（防泄漏）：{safe_cutoff.date()}"
            f"（比 cutoff 早 {LABEL_PERIODS} 个交易日）"
        )

        # ── 4.1 计算技术特征 ──────────────────────────────────────
        logger.info("  [4.1] 计算技术特征...")
        tech_df = compute_tech(
            stock_list=qlib_codes,
            start_date="2020-01-01",
            end_date=cutoff,
        )
        tech_df = lowercase_index(tech_df)   # Bug2：统一小写

        # ── 4.2 计算基本面特征 ────────────────────────────────────
        logger.info("  [4.2] 计算基本面特征...")
        date_list = sorted(
            price_df[price_df["date"] <= cutoff_ts]["date"].unique().tolist()
        )
        fund_feat_df = compute_fund(
            fund_df=fund_df,
            date_list=date_list,
            code_list=all_codes,
        )
        fund_feat_df = lowercase_index(fund_feat_df)   # Bug2：统一小写

        logger.info("  [4.3] 合并技术 + 基本面特征...")
        label_df_lower = lowercase_index(label_df.copy())

        feat_df = tech_df.join(fund_feat_df, how="left")

        # ── 4.3.5 加入大盘特征 ───────────────────────────────────────
        logger.info("  [4.3.5] 计算大盘特征...")
        mkt_df = compute_market(price_df, end_date=cutoff)

        # 广播：大盘特征对所有股票的同一天取相同值
        dt_values   = feat_df.index.get_level_values("datetime")
        mkt_aligned = mkt_df.reindex(dt_values)
        mkt_aligned.index = feat_df.index
        feat_df = pd.concat([feat_df, mkt_aligned], axis=1)
        logger.info(f"  [4.3.5] 大盘特征已合并：{list(mkt_df.columns)}")

        # ── 4.3.6 计算超额收益特征（Alpha）────────────────────────────
        # 大盘特征在截面内无差异，但个股相对大盘的超额表现有截面差异
        # 前提：mkt_ret_1d/5d/20d 已通过 4.3.5 合并进 feat_df
        if "mkt_ret_1d" in feat_df.columns:
            feat_df["alpha_1d"]  = feat_df["ret_1d"]  - feat_df["mkt_ret_1d"]
            feat_df["alpha_5d"]  = feat_df["ret_5d"]  - feat_df["mkt_ret_5d"]
            feat_df["alpha_20d"] = feat_df["ret_20d"] - feat_df["mkt_ret_20d"]
            logger.info("  [4.3.6] Alpha特征已生成：alpha_1d / alpha_5d / alpha_20d")

        # ── 4.4 拼入 Label ──
        feat_df = feat_df.join(label_df_lower, how="left")

        # ── 4.5 验证基本面是否正常 join（诊断用，可注释掉）─────────
        fund_cols = ["is_profitable", "earnings_yield",
                     "pb_inv", "ps_inv", "pcf_inv", "days_since_report"]
        fund_nan_ratio = feat_df[
            [c for c in fund_cols if c in feat_df.columns]
        ].isna().mean().mean() if any(c in feat_df.columns for c in fund_cols) else 1.0
        logger.info(f"  基本面特征平均 NaN 比例：{fund_nan_ratio*100:.1f}%"
                    f" ")

        # ── 4.5.5 删除无效/低质量特征 ──────────────────────────────
        logger.info(f"  feat_df当前列名：{feat_df.columns.tolist()}")
        DROP_COLS = [
            # turnover_ratio_5d 已从黑名单移除，重新dump后有真实数据
            "is_profitable", "earnings_yield",
            "pb_inv", "ps_inv", "pcf_inv", "days_since_report",
            "ma5", "ma10", "ma20",
            # 新加但无效的特征
            "mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
            "mkt_vol_20d", "adv_ratio_5d", "adv_ratio_20d",
            "above_ma20_ratio", "mkt_trend",
            "alpha_1d", "alpha_5d", "alpha_20d",
            "vol_std_5d", "vol_std_20d",
            "gap", "gap_5d_mean",
            "amplitude", "amplitude_5d",
            "ret_vol_diverge_5d","close_position_inday",
            "sharpe_5d","sharpe_20d","lower_shadow",
            "vol_down","vol_up","vol_asym","close_pos_5d"
        ]
        actually_dropped = [c for c in DROP_COLS if c in feat_df.columns]
        feat_df = feat_df.drop(columns=actually_dropped)
        logger.info(f"  已删除无效特征（{len(actually_dropped)}个）：{actually_dropped}")
        logger.info(f"  剩余特征数：{len(feat_df.columns) - 1} 个（不含label）")

        # ── 4.6 划分训练集 / 预测截面 ───────────────────────────────
        dt_level = feat_df.index.get_level_values("datetime")

        # Bug1 修复：训练集截止到 safe_cutoff，而不是 cutoff
        # 滚动窗口：只用最近 TRAIN_WINDOW 个交易日，聚焦近期市场规律
        if cutoff_pos >= TRAIN_WINDOW:
            window_start = all_cal_dates[cutoff_pos - TRAIN_WINDOW]
        else:
            window_start = all_cal_dates[0]
        logger.info(f"  滚动训练窗口：{window_start.date()} ~ {safe_cutoff.date()}")
        train_mask = (
            (dt_level >= window_start) &
            (dt_level <= safe_cutoff) &
            feat_df["label"].notna()
        )
        test_mask  = (dt_level == cutoff_ts)

        df_train = feat_df[train_mask].copy()
        df_test  = feat_df[test_mask].copy()

        # ── 过滤：只保留测试周内有价格数据的股票 ────────────────────
        ws = pd.Timestamp(week_start)
        we = pd.Timestamp(week_end)

        # 测试周内实际有数据的股票集合
        tradeable_codes = set(
            price_df[
                (price_df["date"] >= ws) &
                (price_df["date"] <= we)
            ]["stock_code"].unique()
        )

        # df_test 的 index 是 qlib 格式（sh600519），转成内部格式再过滤
        df_test_codes = (
            df_test.index
            .get_level_values("instrument")
            .map(to_internal_code)
        )
        tradeable_mask = df_test_codes.isin(tradeable_codes)
        removed = (~tradeable_mask).sum()
        if removed > 0:
            logger.warning(
                f"  ⚠️  过滤掉 {removed} 只测试周无价格数据的股票（退市/未上市/数据缺失）"
            )
        df_test = df_test[tradeable_mask]

        if df_train.empty:
            logger.warning("  ⚠️  训练集为空，跳过本周")
            continue
        if df_test.empty:
            logger.warning(
                f"  ⚠️  预测截面 {cutoff_ts.date()} 无数据，跳过本周"
            )
            continue

        logger.info(
            f"  训练集：{df_train.shape[0]} 行 | "
            f"预测截面：{df_test.shape[0]} 只股票"
        )

        y_train = df_train.pop("label")
        df_test.drop(columns=["label"], errors="ignore", inplace=True)
        feature_cols = df_train.columns.tolist()

        # ── 4.7 Bug3 修复：按股票分组 ffill ──────────────────────
        df_train = safe_ffill(df_train, feature_cols)
        df_test  = safe_ffill(df_test,  feature_cols)

        # ── 4.8 截面标准化 ───────────────────────────────────────
        df_train = normalize(df_train, feature_cols)
        df_test  = normalize(df_test,  feature_cols)

        # ── 4.8.5 特征权重放大（基于单变量IC）────────────────────────
        HIGH_IC_COLS = []  # 不人工干预特征权重，让模型自己学
        for col in HIGH_IC_COLS:
            if col in df_train.columns:
                df_train[col] = df_train[col] * 2.0
                df_test[col]  = df_test[col]  * 2.0
        logger.info(f"  特征权重放大：{[c for c in HIGH_IC_COLS if c in df_train.columns]}")

        # ── 4.9 训练模型 ────────────────────────────────────────
        logger.info("  [4.9] 训练 LightGBM...")

        # ── IC 诊断（只在第1周执行一次）──────────────────────────
        if week_idx == 1:
            logger.info("  [诊断] 计算特征 IC（第1周，仅供参考）...")
            _log_feature_ic(df_train[feature_cols], y_train, logger)

        model = LGBMModel()
        model.fit(df_train[feature_cols], y_train)

        # ── 4.10 预测 + 选股 ─────────────────────────────────────
        logger.info("  [4.10] 预测并选出 Top-5...")
        pred = model.predict(df_test[feature_cols])
        pred.index = (
            pred.index
            .get_level_values("instrument")
            .map(to_internal_code)
        )

        # 直接用原始预测分数选股（rank 是严格单调变换，对选股结果无影响）
        candidates_df = select_top_k(pred, k=10)
        logger.info(
            f"  候选股票（含替补）：{candidates_df['stock_code'].tolist()}"
        )

        # ── 4.11 计算测试周实际收益（Bug5：用开盘价）──────────────
        logger.info(
            f"  [4.11] 计算测试周 {week_start}~{week_end} 实际收益..."
        )
        actual_ret, top5 = calculate_actual_return(
            candidates_df, price_df, week_start, week_end, k=5
        )
        logger.info(f"  本周最终选股：{top5['股票代码'].tolist()}")

        # ── 4.12 保存本周选股结果 ────────────────────────────────
        out_path = f"output/result_week{week_idx}.csv"
        top5.to_csv(out_path, index=False, encoding="utf-8-sig")

        weekly_results.append({
            "测试周":     week_name,
            "选出股票":   "  ".join(top5["股票代码"].tolist()),
            "加权收益率": actual_ret,
        })
        logger.info(
            f"  >>> {week_name} 加权收益率：{actual_ret*100:+.4f}%"
        )

    # ════════════════════════════════════════════════════════════
    # [5] 最终成绩单
    # ════════════════════════════════════════════════════════════
    if not weekly_results:
        logger.error("全部跳过，无结果！")
        return

    report_df = pd.DataFrame(weekly_results)

    logger.info("\n" + "="*65)
    logger.info("             回测成绩单             ")
    logger.info("="*65)
    for _, row in report_df.iterrows():
        sign = "📈" if row["加权收益率"] > 0 else "📉"
        logger.info(
            f"  {sign} {row['测试周']:<22} | "
            f"收益率 {row['加权收益率']*100:+.4f}% | "
            f"股票：{row['选出股票']}"
        )
    logger.info("-"*65)

    rets     = report_df["加权收益率"]
    total    = (1 + rets).prod() - 1
    avg      = rets.mean()
    pos_week = (rets > 0).sum()

    logger.info(f"  💰 累计收益率        ：{total*100:+.4f}%")
    logger.info(f"  📊 平均每周收益率 ：{avg*100:+.4f}%")
    logger.info(f"  ✅ 盈利周数         ：{pos_week} / {len(rets)}")
    logger.info("="*65 + "\n")

    report_df.to_csv(
        "output/weekly_report.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("报告已保存至 output/weekly_report.csv")

    # ── [6] 生成最终提交文件 ──────────────────────────────────────
    logger.info("========== [6] 生成最终提交文件 ==========")
    last_week_idx  = len(weekly_results)
    last_result_path = f"output/result_week{last_week_idx}.csv"

    if os.path.exists(last_result_path):
        last_top5 = pd.read_csv(last_result_path, dtype={'股票代码': str})
        # 双重保险：确保前导零不丢失
        last_top5['股票代码'] = last_top5['股票代码'].astype(str).str.zfill(6)
        last_top5.to_csv(
            "output/result.csv", index=False, encoding="utf-8-sig"
        )
        print("="*65)
        print("          📋  最终提交文件 (output/result.csv)")
        print("="*65)
        print(last_top5.to_string(index=False))
        print("="*65 + "\n")
        logger.info("✅ 提交文件已生成：output/result.csv")




if __name__ == "__main__":

    main()