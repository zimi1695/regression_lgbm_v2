"""
模块：market.py
职责：从价格数据计算每日大盘特征，广播给所有股票。

特征列表：
    mkt_ret_1d      全市场等权1日平均收益率
    mkt_ret_5d      全市场等权5日平均收益率
    mkt_ret_20d     全市场等权20日平均收益率
    mkt_vol_20d     全市场20日收益率波动率（时序标准差）
    adv_ratio_5d    5日内上涨股票占比（市场宽度）
    adv_ratio_20d   20日内上涨股票占比
    above_ma20_ratio 价格在20日均线上方的股票占比
    mkt_trend       mkt_ret_5d / mkt_vol_20d（市场风险调整后动量）
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_COLS = [
    "mkt_ret_1d",
    "mkt_ret_5d",
    "mkt_ret_20d",
    "mkt_vol_20d",
    "adv_ratio_5d",
    "adv_ratio_20d",
    "above_ma20_ratio",
    "mkt_trend",
]


def compute(price_df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """
    计算每日大盘特征。

    Parameters
    ----------
    price_df : loader.py 输出，含 stock_code / date / close 列
    end_date : 计算截止日期（包含），格式 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        index   = datetime（单层索引）
        columns = MARKET_COLS
        供 run_weekly_test.py 广播到所有股票
    """
    end_ts = pd.Timestamp(end_date)
    df = price_df[price_df["date"] <= end_ts].copy()
    df = df.sort_values(["stock_code", "date"])

    # ── Step1：构建宽表（行=date，列=stock_code，值=close）────────
    close_wide = df.pivot(index="date", columns="stock_code", values="close")

    # ── Step2：计算各股每日收益率 ─────────────────────────────────
    ret_wide = close_wide.pct_change()  # 1日收益率宽表

    # ── Step3：大盘日均收益（等权）────────────────────────────────
    mkt_ret_1d  = ret_wide.mean(axis=1)
    mkt_ret_5d  = mkt_ret_1d.rolling(5).mean()
    mkt_ret_20d = mkt_ret_1d.rolling(20).mean()

    # ── Step4：大盘波动率（20日时序标准差）───────────────────────
    mkt_vol_20d = mkt_ret_1d.rolling(20).std()

    # ── Step5：市场宽度（上涨股票占比）──────────────────────────
    # 5日累计收益 > 0 的股票比例
    ret_5d_wide    = close_wide.pct_change(5)
    adv_ratio_5d   = (ret_5d_wide > 0).mean(axis=1)

    # 20日累计收益 > 0 的股票比例
    ret_20d_wide   = close_wide.pct_change(20)
    adv_ratio_20d  = (ret_20d_wide > 0).mean(axis=1)

    # ── Step6：均线上方股票占比 ───────────────────────────────────
    ma20_wide        = close_wide.rolling(20).mean()
    above_ma20_ratio = (close_wide > ma20_wide).mean(axis=1)

    # ── Step7：市场趋势（风险调整后动量）─────────────────────────
    mkt_trend = mkt_ret_5d / (mkt_vol_20d + 1e-8)

    # ── Step8：组装输出 ───────────────────────────────────────────
    result = pd.DataFrame({
        "mkt_ret_1d":       mkt_ret_1d,
        "mkt_ret_5d":       mkt_ret_5d,
        "mkt_ret_20d":      mkt_ret_20d,
        "mkt_vol_20d":      mkt_vol_20d,
        "adv_ratio_5d":     adv_ratio_5d,
        "adv_ratio_20d":    adv_ratio_20d,
        "above_ma20_ratio": above_ma20_ratio,
        "mkt_trend":        mkt_trend,
    }, index=close_wide.index)

    result.index.name = "datetime"

    # ── Step9：处理 Inf ───────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan)

    nan_ratio = result.isna().mean().mean()
    logger.info(
        f"[market] 完成：{len(result)} 个交易日，"
        f"NaN比例={nan_ratio:.1%}"
    )

    return result