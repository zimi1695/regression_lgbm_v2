"""
模块H：top_k.py
职责：从预测分数中选出 Top-K 只股票，分配等权重。

契约：
    select_top_k(pred_score, k=5) -> pd.DataFrame

输入：
    pred_score : pd.Series
        index = stock_code（字符串，6位）
        value = 模型预测的收益分数（越大越好）
    k : int，默认 5

输出：
    pd.DataFrame，列：
        stock_code : str
        weight     : float（等权，= 1/k）

断言：
    - weight 之和 <= 1.0（含浮点误差容忍）
    - 所有 weight > 0
    - 行数 == min(k, len(pred_score))
    - 输出股票必须来自输入的 index
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def select_top_k(pred_score: pd.Series, k: int = 5) -> pd.DataFrame:
    """
    从预测分数中选出 Top-K 只股票，分配等权重。

    Parameters
    ----------
    pred_score : pd.Series
        index = stock_code，value = 预测分数（越大越好）
    k : int
        选股数量，默认 5 (在run_weekly_test.py 4.10 选股部分可能会调整 k 值)

    Returns
    -------
    pd.DataFrame
        列：stock_code / weight
        按 weight 降序排列（等权时顺序按分数降序）
    """
    # ── 输入校验 ────────────────────────────────────────────────
    _assert_inputs(pred_score, k)

    # ── 实际选股数量（防止股票数不足 k）──────────────────────────
    actual_k = min(k, len(pred_score))
    if actual_k < k:
        logger.warning(
            f"[top_k] 可用股票数 ({len(pred_score)}) < k ({k})，"
            f"实际选 {actual_k} 只"
        )

    # ── 选 Top-K（按分数降序，取前 actual_k）────────────────────
    top_stocks = (
        pred_score
        .sort_values(ascending=False)   # 分数从高到低
        .head(actual_k)
        .index
        .tolist()
    )

    # ── 分配等权重 ───────────────────────────────────────────────
    weight = 1.0 / actual_k

    # 强制保留6位前导零（防止 003816 变成 3816）
    top_stocks = [str(c).zfill(6) for c in top_stocks]

    result = pd.DataFrame({
        "stock_code": top_stocks,
        "weight":     [weight] * actual_k,
    })

    logger.info(
        f"[top_k] 选出 {actual_k} 只股票，"
        f"每只权重 {weight:.4f}，"
        f"总权重 {result['weight'].sum():.4f}"
    )

    # ── 输出契约校验 ─────────────────────────────────────────────
    _assert_outputs(result, pred_score, k)

    return result


# ════════════════════════════════════════════════════════════════
# 内部校验函数
# ════════════════════════════════════════════════════════════════

def _assert_inputs(pred_score: pd.Series, k: int) -> None:
    """校验输入合法性"""

    # 类型
    assert isinstance(pred_score, pd.Series), (
        "[top_k][断言失败] pred_score 必须是 pd.Series，"
        f"实际类型：{type(pred_score)}"
    )
    # 非空
    assert len(pred_score) > 0, (
        "[top_k][断言失败] pred_score 不能为空"
    )
    # k 合法
    assert isinstance(k, int) and k > 0, (
        f"[top_k][断言失败] k 必须是正整数，实际值：{k}"
    )
    # 无 NaN 分数（NaN 会导致排序结果不确定）
    nan_count = pred_score.isna().sum()
    assert nan_count == 0, (
        f"[top_k][断言失败] pred_score 含 {nan_count} 个 NaN，"
        "请在调用前清理"
    )
    # index 不能有重复（同一股票出现两次会产生歧义）
    assert pred_score.index.is_unique, (
        "[top_k][断言失败] pred_score 的 index（stock_code）含重复值"
    )


def _assert_outputs(
    result: pd.DataFrame,
    pred_score: pd.Series,
    k: int,
) -> None:
    """校验输出合法性"""

    actual_k = min(k, len(pred_score))

    # 行数
    assert len(result) == actual_k, (
        f"[top_k][断言失败] 输出行数 {len(result)} != 期望 {actual_k}"
    )
    # 必要的列
    for col in ("stock_code", "weight"):
        assert col in result.columns, (
            f"[top_k][断言失败] 输出缺少列：{col}"
        )
    # 所有 weight > 0
    assert (result["weight"] > 0).all(), (
        "[top_k][断言失败] 存在 weight <= 0"
    )
    # weight 之和 <= 1.0（留 1e-6 浮点误差容忍）
    total_weight = result["weight"].sum()
    assert total_weight <= 1.0 + 1e-6, (
        f"[top_k][断言失败] weight 之和 {total_weight:.6f} > 1.0"
    )
    # 所有股票来自原始 pred_score
    valid_codes = set(pred_score.index)
    selected_codes = set(result["stock_code"])
    assert selected_codes.issubset(valid_codes), (
        f"[top_k][断言失败] 选出了不存在于 pred_score 中的股票："
        f"{selected_codes - valid_codes}"
    )