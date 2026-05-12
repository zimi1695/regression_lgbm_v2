"""
模块D：technical.py
职责：基于已初始化的 qlib 环境，用 qlib 表达式引擎计算技术指标特征。

为什么用 qlib 表达式引擎而不是手写 pandas？
  - qlib 表达式有内置缓存机制，同一特征不重复计算
  - 表达式语法简洁，便于后续扩展新因子
  - 与 qlib 生态（Alpha158等）保持一致，便于对比

契约：
  输入：
    stock_list : list[str]，qlib 格式股票代码，如 ["SH600519", "SZ000001"]
    start_date : str，"YYYY-MM-DD"
    end_date   : str，"YYYY-MM-DD"

  输出：pd.DataFrame
    - index   : MultiIndex (datetime, instrument)
    - columns : 21个技术特征（见 FEATURE_COLS）
    - 值类型  : float64
    - 无 Inf  : 已替换为 NaN

  断言：
    - 输出列名与 FEATURE_COLS 完全一致
    - RSI 值在 [0, 100] 范围内（允许 NaN）
    - 无 Inf 值
    - 每列 NaN 比例 < 80%（否则警告）
"""

import logging
import numpy as np
import pandas as pd
from qlib.data import D

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# 特征定义：qlib 表达式 → 输出列名
# ════════════════════════════════════════════════════════════════════

# qlib 表达式语法说明：
#   $close        : 收盘价
#   Ref($close, 5): 5天前的收盘价
#   Mean($close,5): 过去5天收盘价均值（滚动）
#   Std($close, 5): 过去5天收盘价标准差
#   $volume       : 成交量

FEATURE_EXPRESSIONS = {
    # ── 均线 ──────────────────────────────────────────────────────
    "ma5":  "Mean($close, 5)",
    "ma10": "Mean($close, 10)",
    "ma20": "Mean($close, 20)",

    # ── 价格相对均线比值（去量纲）────────────────────────────────
    "price_ma5_ratio":  "$close / Mean($close, 5)",
    "price_ma10_ratio": "$close / Mean($close, 10)",
    "price_ma20_ratio": "$close / Mean($close, 20)",

    # ── 历史收益率 ────────────────────────────────────────────────
    # Ref($close, N) 是 N 天前的收盘价
    # ($close / Ref($close, N)) - 1 = N天收益率
    "ret_1d":  "$close / Ref($close, 1) - 1",
    "ret_5d":  "$close / Ref($close, 5) - 1",
    "ret_10d": "$close / Ref($close, 10) - 1",
    "ret_20d": "$close / Ref($close, 20) - 1",

    # ── 波动率（收益率的滚动标准差）──────────────────────────────
    "vol_5d":  "Std($close / Ref($close, 1) - 1, 5)",
    "vol_10d": "Std($close / Ref($close, 1) - 1, 10)",
    "vol_20d": "Std($close / Ref($close, 1) - 1, 20)",

    # ── 成交量相对特征 ────────────────────────────────────────────
    "vol_ratio_5d":  "$volume / Mean($volume, 5)",
    "vol_ratio_10d": "$volume / Mean($volume, 10)",

    # ── 价格在近20日高低中的位置（0~1）───────────────────────────
    "price_position_20d": (
        "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-8)"
    ),

    # ── RSI（相对强弱指数）────────────────────────────────────────
   # 用基础算子调换减法顺序手动实现 RSI(qlib 这个版本没有内置 RSI 算子)
    "rsi_14": (
        "100 - 100 / (1 + "
        "Mean(($close-Ref($close,1)+Abs($close-Ref($close,1)))/2, 14) / "
        "(Mean((Ref($close,1)-$close+Abs($close-Ref($close,1)))/2, 14) + 1e-8)"
        ")"
    ),

    # ── MACD ──────────────────────────────────────────────────────
    # EMA(N) = 指数移动平均
    # MACD线 = EMA12 - EMA26
    "macd": "EMA($close, 12) - EMA($close, 26)",

    # 信号线 = MACD的9日EMA（qlib里用Ref近似）
    "macd_signal": "EMA(EMA($close, 12) - EMA($close, 26), 9)",

    # 柱状图 = MACD线 - 信号线
    "macd_hist": (
        "(EMA($close, 12) - EMA($close, 26))"
        " - EMA(EMA($close, 12) - EMA($close, 26), 9)"
    ),

    # ── 换手率相对特征 ────────────────────────────────────────────
    "turnover_ratio_5d": "$turnover / Mean($turnover, 5)",

    # ── 超短期动量（2~3日）──────────────────────────────────────
    "ret_2d": "$close / Ref($close, 2) - 1",
    "ret_3d": "$close / Ref($close, 3) - 1",

    # ── 超短期量比（3日）────────────────────────────────────────
    "vol_ratio_3d": "$volume / Mean($volume, 3)",

    # ── 价格距近期高低点 ─────────────────────────────────────────
    "price_high5_ratio": "$close / Max($high, 5)",
    "price_low5_ratio":  "$close / Min($low,  5)",

    # ── 成交额加速度 ─────────────────────────────────────────────
    "amount_ratio_5d": "$amount / Mean($amount, 5)",
    "amount_ratio_3d": "$amount / Mean($amount, 3)",

    # ── 价格在近5日高低中的位置 ──────────────────────────────────
    "price_position_5d": (
        "($close - Min($low, 5)) / (Max($high, 5) - Min($low, 5) + 1e-8)"
    ),

    "price_position_5d": (
        "($close - Min($low, 5)) / (Max($high, 5) - Min($low, 5) + 1e-8)"
    ),
    
    # ── 个股波动率 ────────────────────────────────────────────────
    "vol_std_5d":    "Std(Ref($close,1)/Ref($close,2)-1, 5)",
    "vol_std_20d":   "Std(Ref($close,1)/Ref($close,2)-1, 20)",

    # ── 跳空缺口 ─────────────────────────────────────────────────
    "gap":           "$open/Ref($close,1)-1",
    "gap_5d_mean":   "Mean($open/Ref($close,1)-1, 5)",

    # ── 日内振幅 ─────────────────────────────────────────────────
    "amplitude":     "($high-$low)/Ref($close,1)",
    "amplitude_5d":  "Mean(($high-$low)/Ref($close,1), 5)",

    # ── 蜡烛图形态 ────────────────────────────────────────────────
    # 收盘在当日高低范围内的位置（0=收在最低，1=收在最高）
    "close_position_inday": (
        "($close - $open) / ($high - $low + 1e-8)"
    ),
    # 上影线长度占当日振幅的比例（上影线长 = 多头力竭信号）
    "upper_shadow": (
        "($high - Greater($open, $close)) / ($high - $low + 1e-8)"
    ),
    # 下影线长度占当日振幅的比例（下影线长 = 空头力竭信号）
    "lower_shadow": (
        "(Less($open, $close) - $low) / ($high - $low + 1e-8)"
    ),

    # ── 个股 Sharpe 比（收益/波动）────────────────────────────────
    # 近5日收益/波动，选"涨得稳"而非"涨得猛"的股票
    "sharpe_5d": (
        "Mean($close/Ref($close,1)-1, 5) "
        "/ (Std($close/Ref($close,1)-1, 5) + 1e-8)"
    ),
    # 近20日版本
    "sharpe_20d": (
        "Mean($close/Ref($close,1)-1, 20) "
        "/ (Std($close/Ref($close,1)-1, 20) + 1e-8)"
    ),

    # ── 距历史高点的位置 ──────────────────────────────────────────
    # 距252日（约1年）最高点的比例，捕捉年线突破信号
    "near_52w_high": "$close / Max($high, 252)",

    # ── 量价背离：价涨量缩 or 价跌量缩 ─────────────────────────
    # 近5日收益率 / 近5日成交量比值
    # 正值大 = 以小成交量涨价（可能是真突破）
    # 负值 = 以放量下跌（恐慌抛售）
    "ret_vol_diverge_5d": (
        "($close / Ref($close,5) - 1) "
        "/ (Mean($volume, 5) / Ref(Mean($volume,5), 5) + 1e-8)"
    ),

    "vol_down":  "Std(If($close<Ref($close,1), $close/Ref($close,1)-1, 0), 5)",  # 下跌日波动率
    "vol_up":    "Std(If($close>Ref($close,1), $close/Ref($close,1)-1, 0), 5)",  # 上涨日波动率
    "vol_asym":  "Std(If($close<Ref($close,1), $close/Ref($close,1)-1, 0), 5) / (Std(If($close>Ref($close,1), $close/Ref($close,1)-1, 0), 5) + 1e-8)",  # 下跌波动率 / 上涨波动率

    "close_pos_5d": "Mean(($close - $low) / ($high - $low + 1e-8), 5)",  # 5日平均收盘位置
}

# 输出列名列表（顺序固定，供下游模块使用）
FEATURE_COLS = list(FEATURE_EXPRESSIONS.keys())


# ════════════════════════════════════════════════════════════════════
# 公开接口
# ════════════════════════════════════════════════════════════════════

def compute(
    stock_list: list,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    logger.info(
        f"[technical] 计算技术特征：{len(stock_list)} 只股票，"
        f"{start_date} ~ {end_date}"
    )

    # ── 1. 构建 qlib 表达式列表 ──────────────────────────────────────
    fields = list(FEATURE_EXPRESSIONS.values())  # 35个表达式
    names  = FEATURE_COLS                         # 35个名字

    # ── 2. 调用 D.features() ─────────────────────────────────────────
    instruments = D.instruments("all") if stock_list is None else stock_list
    df = D.features(
        instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )

    # ── 3. 重命名列 ──────────────────────────────────────────────────
    df.columns = names

    # ── 4. 处理 Inf 值 ───────────────────────────────────────────────
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        logger.warning(f"[technical] 发现 {inf_count} 个 Inf 值，已替换为 NaN")
        df = df.replace([np.inf, -np.inf], np.nan)

    # ── 5. 契约断言 ──────────────────────────────────────────────────
    _assert_contract(df)

    logger.info(
        f"[technical] 完成：{len(df)} 行，{len(df.columns)} 列，"
        f"NaN比例={df.isna().mean().mean():.1%}"
    )

    return df


# ════════════════════════════════════════════════════════════════════
# 契约验证
# ════════════════════════════════════════════════════════════════════

def _assert_contract(df: pd.DataFrame) -> None:
    """对输出 DataFrame 执行契约断言"""

    # 断言①：输出列名与 FEATURE_COLS 完全一致
    assert list(df.columns) == FEATURE_COLS, (
        f"[technical][断言失败] 列名不匹配\n"
        f"  期望：{FEATURE_COLS}\n"
        f"  实际：{list(df.columns)}"
    )

    # 断言②：无 Inf 值
    has_inf = np.isinf(df.select_dtypes(include=[np.number]).values).any()
    assert not has_inf, "[technical][断言失败] 输出含 Inf 值"

    # 断言③：RSI 在 [0, 100] 范围内（忽略 NaN）
    rsi_vals = df["rsi_14"].dropna()
    if len(rsi_vals) > 0:
        assert rsi_vals.between(0, 100).all(), (
            f"[technical][断言失败] rsi_14 超出 [0,100] 范围，"
            f"min={rsi_vals.min():.2f}, max={rsi_vals.max():.2f}"
        )

    # 断言④：每列 NaN 比例警告
    for col in FEATURE_COLS:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > 0.8:
            logger.warning(
                f"[technical] 列 '{col}' NaN 比例过高：{nan_ratio:.1%}"
            )
