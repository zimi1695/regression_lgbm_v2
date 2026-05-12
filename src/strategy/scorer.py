"""
模块I：scorer.py
职责：给定 top_k 选股结果和测试周，用真实价格计算加权收益率。

契约：
    score_portfolio(top_k_df, week_start, week_end, price_df) -> float

输入：
    top_k_df   : pd.DataFrame，列 stock_code / weight
                 来自 top_k.select_top_k() 的输出
    week_start : str 或 datetime，周初日期（用这天的开盘价买入）
    week_end   : str 或 datetime，周末日期（用这天的开盘价卖出）
    price_df   : pd.DataFrame，来自 loader.py 的输出
                 必须包含列：stock_code / date / open

输出：
    float，组合加权收益率
    例如：0.023 表示本周组合涨了 2.3%

断言：
    - top_k_df 格式合法（列存在，weight > 0，weight 之和 <= 1）
    - week_start < week_end
    - price_df 包含必要的列
    - 所选股票在 price_df 中有价格数据
    - 收盘价 > 0
    - 返回值不是 NaN / Inf
    - 返回值 = -999 时抛出异常（-999 是比赛的"无效"标志）
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def score_portfolio(
    top_k_df:   pd.DataFrame,
    week_start,
    week_end,
    price_df:   pd.DataFrame,
) -> float:
    """
    计算 top_k 组合在 [week_start, week_end] 区间的加权收益率。

    Parameters
    ----------
    top_k_df   : top_k.select_top_k() 的输出
    week_start : 周初日期（开盘买入）
    week_end   : 周末日期（开盘卖出）
    price_df   : loader.py 输出的完整价格 DataFrame

    Returns
    -------
    float，加权收益率（如 0.023 表示 +2.3%）
    """
    # ── Step1：输入校验 ──────────────────────────────────────────
    week_start = pd.Timestamp(week_start)
    week_end   = pd.Timestamp(week_end)
    _assert_inputs(top_k_df, week_start, week_end, price_df)

    # ── Step2：提取所需股票的价格 ────────────────────────────────
    selected_codes = top_k_df["stock_code"].tolist()

    # 过滤出所选股票的数据
    mask = price_df["stock_code"].isin(selected_codes)
    sub_df = price_df[mask].copy()

    # date 列统一转为 Timestamp
    sub_df["date"] = pd.to_datetime(sub_df["date"])

    # ── Step3：找"最近可用交易日"的收盘价 ────────────────────────
    # 原因：week_start / week_end 可能是周末或节假日（非交易日），
    #       需要向前找最近的有数据的交易日
    price_start = _get_closest_price(sub_df, week_start, direction="forward")
    price_end   = _get_closest_price(sub_df, week_end,   direction="forward")

    logger.info(
        f"[scorer] 周期 {week_start.date()} → {week_end.date()}，"
        f"实际使用 {price_start['date'].iloc[0].date()} → "
        f"{price_end['date'].iloc[0].date()}"
    )

    # ── Step4：计算每只股票的周收益率 ────────────────────────────
    returns = {}
    missing = []

    for code in selected_codes:
        p_start = price_start.loc[
            price_start["stock_code"] == code, "open"
        ]
        p_end = price_end.loc[
            price_end["stock_code"] == code, "open"
        ]

        if p_start.empty or p_end.empty:
            logger.warning(f"[scorer] 股票 {code} 缺少价格数据，跳过")
            missing.append(code)
            continue

        start_price = p_start.iloc[0]
        end_price   = p_end.iloc[0]

        assert start_price > 0, (
            f"[scorer][断言失败] 股票 {code} 周初收盘价 <= 0：{start_price}"
        )
        assert end_price > 0, (
            f"[scorer][断言失败] 股票 {code} 周末收盘价 <= 0：{end_price}"
        )

        ret = (end_price - start_price) / start_price
        returns[code] = ret
        logger.info(
            f"[scorer] {code}: {start_price:.2f} → {end_price:.2f}，"
            f"收益率 {ret*100:.2f}%"
        )

    # ── Step5：处理缺失股票（与 score_self.py 对齐）────────────────────
    weight_map = dict(zip(top_k_df["stock_code"], top_k_df["weight"]))

    if missing:
        logger.warning(
            f"[scorer] {len(missing)} 只股票无价格数据，"
            f"直接剔除，剩余权重保持不变（与官方评分口径一致）"
        )
        for code in missing:
            weight_map.pop(code, None)
    # ✅ 不做任何归一化，权重之和允许 < 1.0
    # 这与 score_self.py 的 merge 后直接计算行为完全一致

    # ── Step6：计算加权收益率 ────────────────────────────────────
    if not returns:
        raise ValueError(
            "[scorer] 所有选中股票均无价格数据，无法计算收益率"
        )

    weighted_return = sum(
        weight_map.get(code, 0.0) * ret
        for code, ret in returns.items()
    )

    # ── Step7：输出契约校验 ──────────────────────────────────────
    assert not np.isnan(weighted_return), (
        "[scorer][断言失败] 最终收益率为 NaN"
    )
    assert not np.isinf(weighted_return), (
        "[scorer][断言失败] 最终收益率为 Inf"
    )
    if weighted_return == -999:
        raise ValueError(
            "[scorer] 收益率为 -999，触发无效标志，请检查数据"
        )

    logger.info(f"[scorer] 本周组合加权收益率：{weighted_return*100:.4f}%")
    return weighted_return


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def _get_closest_price(
    sub_df: pd.DataFrame,
    target_date: pd.Timestamp,
    direction: str = "forward",
) -> pd.DataFrame:
    """
    找目标日期"最近可用交易日"的收盘价。

    direction='forward'：从 target_date 向后找（含当天）
    direction='backward'：从 target_date 向前找（含当天）
    """
    if direction == "forward":
        # >= target_date 的最近一天
        candidates = sub_df[sub_df["date"] >= target_date]
        if candidates.empty:
            # 退而求其次，取最后一天
            candidates = sub_df
        closest_date = candidates["date"].min()
    else:
        candidates = sub_df[sub_df["date"] <= target_date]
        if candidates.empty:
            candidates = sub_df
        closest_date = candidates["date"].max()

    return sub_df[sub_df["date"] == closest_date]


# ════════════════════════════════════════════════════════════════
# 内部校验
# ════════════════════════════════════════════════════════════════

def _assert_inputs(top_k_df, week_start, week_end, price_df):
    """校验所有输入的合法性"""

    # top_k_df 格式
    assert isinstance(top_k_df, pd.DataFrame), (
        "[scorer][断言失败] top_k_df 必须是 DataFrame"
    )
    for col in ("stock_code", "weight"):
        assert col in top_k_df.columns, (
            f"[scorer][断言失败] top_k_df 缺少列：{col}"
        )
    assert len(top_k_df) > 0, (
        "[scorer][断言失败] top_k_df 不能为空"
    )
    assert (top_k_df["weight"] > 0).all(), (
        "[scorer][断言失败] top_k_df 存在 weight <= 0"
    )
    assert top_k_df["weight"].sum() <= 1.0 + 1e-6, (
        f"[scorer][断言失败] top_k_df weight 之和 > 1.0"
    )

    # 日期
    assert week_start < week_end, (
        f"[scorer][断言失败] week_start ({week_start}) "
        f"必须早于 week_end ({week_end})"
    )

    # price_df 格式
    assert isinstance(price_df, pd.DataFrame), (
        "[scorer][断言失败] price_df 必须是 DataFrame"
    )
    for col in ("stock_code", "date", "open"):
        assert col in price_df.columns, (
            f"[scorer][断言失败] price_df 缺少列：{col}"
        )
    assert len(price_df) > 0, (
        "[scorer][断言失败] price_df 不能为空"
    )