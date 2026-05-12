"""
模块E：fundamental.py
职责：处理基本面数据（PE/PB/PS/PCF），与技术特征对齐日期，
      生成衍生特征并输出标准 DataFrame。

决策记录：
  - 负PE/负PB等处理：拆成 is_profitable(0/1) + earnings_yield(1/PE)
  - 季频→日频：前向填充（ffill），符合实际投资逻辑
  - 数据新鲜度：额外生成 days_since_report 特征

契约：
  输入：
    fund_df  : fetcher.py 输出的基本面 DataFrame
               列：stock_code, date, peTTM, pbMRQ, psTTM, pcfNcfTTM
    date_list: 需要对齐的交易日列表（list[pd.Timestamp]）
    code_list: 需要处理的股票代码列表（6位内部格式）

  输出：pd.DataFrame
    - index   : MultiIndex (instrument, datetime)
                instrument 为 qlib 格式（SH600519/SZ000001）
    - columns : FUNDAMENTAL_COLS（见下方定义）
    - 值类型  : float64

  断言：
    - 输出列名与 FUNDAMENTAL_COLS 完全一致
    - days_since_report 在 [0, 200] 之间（允许NaN）
    - NaN 比例 < 50%（否则警告）
    - 无 Inf 值
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# 输出特征列定义
# ════════════════════════════════════════════════════════════════════

FUNDAMENTAL_COLS = [
    # ── PE 衍生 ────────────────────────────────────────────────────
    "is_profitable",      # 是否盈利：peTTM > 0 → 1，否则 → 0
    "earnings_yield",     # 盈利收益率：1 / peTTM（保留负值）

    # ── PB 衍生 ────────────────────────────────────────────────────
    "pb_inv",             # 1 / pbMRQ（市净率倒数，越大越便宜）

    # ── PS 衍生 ────────────────────────────────────────────────────
    "ps_inv",             # 1 / psTTM

    # ── PCF 衍生 ───────────────────────────────────────────────────
    "pcf_inv",            # 1 / pcfNcfTTM

    # ── 数据新鲜度 ─────────────────────────────────────────────────
    "days_since_report",  # 距上次季报发布的天数（0~90为正常，>90说明延迟）
]


# ════════════════════════════════════════════════════════════════════
# 公开接口
# ════════════════════════════════════════════════════════════════════

def compute(
    fund_df:   pd.DataFrame,
    date_list: list,
    code_list: list,
) -> pd.DataFrame:
    """
    处理基本面数据，对齐日频，生成衍生特征。

    Parameters
    ----------
    fund_df   : fetcher.py 输出的基本面 DataFrame
    date_list : 需要对齐的交易日列表（pd.Timestamp 列表）
    code_list : 需要处理的6位内部格式股票代码列表

    Returns
    -------
    pd.DataFrame
        index   = MultiIndex(instrument, datetime)
        columns = FUNDAMENTAL_COLS
    """
    logger.info(
        f"[fundamental] 处理基本面特征：{len(code_list)} 只股票，"
        f"{len(date_list)} 个交易日"
    )

    date_index = pd.DatetimeIndex(sorted(date_list))
    result_dfs = []

    for stock_code in code_list:
        # ── 1. 取该股票的基本面数据 ──────────────────────────────────
        stock_data = fund_df[fund_df["stock_code"] == stock_code].copy()

        if len(stock_data) == 0:
            logger.warning(f"[fundamental] {stock_code} 无基本面数据，跳过")
            continue

        stock_data = stock_data.sort_values("date").set_index("date")

        # ── 2. 重索引到全量交易日，前向填充 ─────────────────────────
        # 先合并：把已有数据日期和目标日期合并，再前向填充，再取目标日期
        all_dates = date_index.union(stock_data.index)
        stock_data = stock_data.reindex(all_dates).ffill()
        stock_data = stock_data.reindex(date_index)

        # ── 3. 计算 days_since_report ────────────────────────────────
        # 找出原始数据中有数据的日期（即季报发布日）
        original_dates = fund_df[
            fund_df["stock_code"] == stock_code
        ]["date"].sort_values().values

        stock_data["days_since_report"] = _compute_days_since_report(
            date_index, original_dates
        )

        # ── 4. 生成衍生特征 ──────────────────────────────────────────
        feat = pd.DataFrame(index=date_index)

        # is_profitable：peTTM > 0
        feat["is_profitable"] = (stock_data["peTTM"] > 0).astype(float)
        # peTTM == 0 时也视为不可用
        feat.loc[stock_data["peTTM"] == 0, "is_profitable"] = np.nan

        # earnings_yield = 1 / peTTM（负值保留，表示亏损）
        feat["earnings_yield"] = _safe_inv(stock_data["peTTM"])

        # pb_inv = 1 / pbMRQ
        feat["pb_inv"] = _safe_inv(stock_data["pbMRQ"])

        # ps_inv = 1 / psTTM
        feat["ps_inv"] = _safe_inv(stock_data["psTTM"])

        # pcf_inv = 1 / pcfNcfTTM
        feat["pcf_inv"] = _safe_inv(stock_data["pcfNcfTTM"])

        # days_since_report
        feat["days_since_report"] = stock_data["days_since_report"].values

        # ── 5. 构建 MultiIndex ───────────────────────────────────────
        qlib_inst = _to_qlib_instrument(stock_code)
        feat.index = pd.MultiIndex.from_arrays(
            [[qlib_inst] * len(date_index), date_index],
            names=["instrument", "datetime"],
        )

        result_dfs.append(feat)

    if not result_dfs:
        raise ValueError("[fundamental] 未生成任何基本面特征，请检查输入数据")

    result = pd.concat(result_dfs)

    # ── 6. 处理 Inf ───────────────────────────────────────────────────
    inf_count = np.isinf(result.select_dtypes(include=[np.number]).values).sum()
    if inf_count > 0:
        logger.warning(f"[fundamental] 发现 {inf_count} 个 Inf，已替换为 NaN")
        result = result.replace([np.inf, -np.inf], np.nan)

    # ── 7. 契约断言 ───────────────────────────────────────────────────
    _assert_contract(result)

    logger.info(
        f"[fundamental] 完成：{len(result)} 行，"
        f"NaN比例={result.isna().mean().mean():.1%}"
    )
    return result


# ════════════════════════════════════════════════════════════════════
# 内部工具函数
# ════════════════════════════════════════════════════════════════════

def _to_qlib_instrument(stock_code: str) -> str:
    """6位内部代码 → qlib instrument 格式（大写前缀）"""
    prefix = "SH" if stock_code.startswith("6") else "SZ"
    return f"{prefix}{stock_code}"


def _safe_inv(series: pd.Series) -> pd.Series:
    """
    安全取倒数：1 / x
    - x == 0 → NaN（避免除零）
    - x 为 NaN → NaN
    - 负值正常保留
    """
    result = series.copy().astype(float)
    result[result == 0] = np.nan
    return 1.0 / result


def _compute_days_since_report(
    date_index: pd.DatetimeIndex,
    report_dates: np.ndarray,
) -> pd.Series:
    """
    对每个交易日，计算距上次季报发布日期的天数。

    Parameters
    ----------
    date_index   : 目标交易日索引
    report_dates : 季报实际发布日期数组（numpy datetime64）

    Returns
    -------
    pd.Series，index 与 date_index 一致
    """
    days = []
    report_ts = pd.DatetimeIndex(report_dates)

    for dt in date_index:
        # 找到所有 <= dt 的季报日期
        past_reports = report_ts[report_ts <= dt]
        if len(past_reports) == 0:
            days.append(np.nan)
        else:
            last_report = past_reports[-1]
            days.append((dt - last_report).days)

    return pd.Series(days, index=date_index)


# ════════════════════════════════════════════════════════════════════
# 契约验证
# ════════════════════════════════════════════════════════════════════

def _assert_contract(df: pd.DataFrame) -> None:
    """对输出 DataFrame 执行契约断言"""

    # 断言①：输出列名正确
    assert list(df.columns) == FUNDAMENTAL_COLS, (
        f"[fundamental][断言失败] 列名不匹配\n"
        f"  期望：{FUNDAMENTAL_COLS}\n"
        f"  实际：{list(df.columns)}"
    )

    # 断言②：无 Inf
    has_inf = np.isinf(
        df.select_dtypes(include=[np.number]).values
    ).any()
    assert not has_inf, "[fundamental][断言失败] 输出含 Inf 值"

    # 断言③：days_since_report 范围合理
    dsr = df["days_since_report"].dropna()
    if len(dsr) > 0:
        assert (dsr >= 0).all(), \
            "[fundamental][断言失败] days_since_report 含负值"
        if (dsr > 200).any():
            logger.warning(
                f"[fundamental] days_since_report 有值 > 200天，"
                f"可能数据缺失：max={dsr.max():.0f}"
            )

    # 断言④：is_profitable 只含 0/1/NaN
    ip = df["is_profitable"].dropna()
    assert ip.isin([0.0, 1.0]).all(), \
        "[fundamental][断言失败] is_profitable 含非0/1值"

    # 断言⑤：NaN 比例警告
    for col in FUNDAMENTAL_COLS:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > 0.5:
            logger.warning(
                f"[fundamental] 列 '{col}' NaN 比例过高：{nan_ratio:.1%}"
            )