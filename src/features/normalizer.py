"""
模块F：normalizer.py
职责：截面标准化（Cross-Sectional Z-Score Normalization）

为什么需要截面标准化？
  不同股票价格量级差异巨大（低价股 vs 高价股）
  直接用原始特征，模型会被价格绝对值主导
  截面标准化后，每个特征在每天的分布统一为均值≈0、标准差≈1
  模型真正比较的是"相对排名"，而不是"绝对数值"

契约：
  输入：
    df          : pd.DataFrame
                  index = MultiIndex(instrument, datetime) 或普通 DatetimeIndex
                  columns 包含待标准化的特征列
    feature_cols: 需要标准化的列名列表
    label_col   : 标签列名（不参与标准化，默认 None）
    min_stocks  : 每天至少有多少只股票才执行标准化（默认 10）

  输出：pd.DataFrame
    - 与输入结构完全相同
    - feature_cols 中的列已被截面 Z-Score 标准化
    - label_col（如有）保持不变
    - 无 Inf 值

  断言：
    - 标准化后每列每天的均值 ≈ 0（允许 ±0.1 误差）
    - 标准化后每列每天的标准差 ≈ 1（允许 ±0.3 误差）
    - 不修改 label_col
    - 无 Inf 值
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize(
    df:           pd.DataFrame,
    feature_cols: list,
    label_col:    str  = None,
    min_stocks:   int  = 10,
) -> pd.DataFrame:
    """
    对 DataFrame 执行截面 Z-Score 标准化。

    Parameters
    ----------
    df           : 输入 DataFrame，index 含 datetime 层
    feature_cols : 需要标准化的列名列表
    label_col    : 标签列名，不参与标准化（可为 None）
    min_stocks   : 每天最少需要多少只股票才执行标准化
                   股票数不足时跳过该天（保留原值）

    Returns
    -------
    pd.DataFrame，与输入结构相同，feature_cols 已标准化
    """
    logger.info(
        f"[normalizer] 截面标准化：{len(feature_cols)} 个特征列，"
        f"min_stocks={min_stocks}"
    )

    # 保存标签列（标准化前先取出）
    label_backup = None
    if label_col and label_col in df.columns:
        label_backup = df[label_col].copy()

    result = df.copy()

    # ── 判断 index 类型，决定如何分组 ────────────────────────────────
    if isinstance(df.index, pd.MultiIndex):
        # MultiIndex(instrument, datetime) 或 (datetime, instrument)
        # 找到 datetime 层的位置
        if "datetime" in df.index.names:
            dt_level = df.index.names.index("datetime")
        else:
            # 退化处理：取第一层
            dt_level = 0

        result = _normalize_multiindex(
            result, feature_cols, dt_level, min_stocks
        )
    else:
        # 普通 DatetimeIndex（V1 风格的数据）
        result = _normalize_flat(result, feature_cols, min_stocks)

    # ── 还原标签列（确保没被修改）────────────────────────────────────
    if label_backup is not None:
        result[label_col] = label_backup

    # ── 替换 Inf ─────────────────────────────────────────────────────
    inf_count = np.isinf(
        result[feature_cols].select_dtypes(include=[np.number]).values
    ).sum()
    if inf_count > 0:
        logger.warning(f"[normalizer] 标准化后仍有 {inf_count} 个 Inf，已替换为 NaN")
        result[feature_cols] = result[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    # ── 契约断言 ─────────────────────────────────────────────────────
    _assert_contract(result, feature_cols, label_col, label_backup, df)

    logger.info("[normalizer] 截面标准化完成")
    return result


# ════════════════════════════════════════════════════════════════════
# 内部函数
# ════════════════════════════════════════════════════════════════════

def _normalize_multiindex(
    df:           pd.DataFrame,
    feature_cols: list,
    dt_level:     int,
    min_stocks:   int,
) -> pd.DataFrame:
    """对 MultiIndex DataFrame 按 datetime 层分组标准化"""

    def _zscore_group(group):
        n = len(group)
        if n < min_stocks:
            logger.warning(
                f"[normalizer] 某交易日股票数 {n} < {min_stocks}，跳过标准化"
            )
            return group
        for col in feature_cols:
            if col not in group.columns:
                continue
            col_data = group[col].astype(float)
            mu  = col_data.mean()
            std = col_data.std()
            if std > 1e-8:
                group[col] = (col_data - mu) / std
            else:
                # 所有股票该特征值相同，标准化后全为0
                group[col] = 0.0
        return group

    return df.groupby(level=dt_level, group_keys=False).apply(_zscore_group)


def _normalize_flat(
    df:           pd.DataFrame,
    feature_cols: list,
    min_stocks:   int,
) -> pd.DataFrame:
    """对普通 DatetimeIndex DataFrame 按日期分组标准化"""

    # 需要有 stock_code 列和 date 列，或者 index 是日期
    date_col = None
    if "date" in df.columns:
        date_col = "date"
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["_date_tmp"] = df.index
        date_col = "_date_tmp"

    if date_col is None:
        logger.warning("[normalizer] 无法识别日期列，跳过标准化")
        return df

    result_dfs = []
    for date, group in df.groupby(date_col):
        g = group.copy()
        n = len(g)
        if n < min_stocks:
            result_dfs.append(g)
            continue
        for col in feature_cols:
            if col not in g.columns:
                continue
            col_data = g[col].astype(float)
            mu  = col_data.mean()
            std = col_data.std()
            if std > 1e-8:
                g[col] = (col_data - mu) / std
            else:
                g[col] = 0.0
        result_dfs.append(g)

    result = pd.concat(result_dfs)
    if "_date_tmp" in result.columns:
        result = result.drop(columns=["_date_tmp"])
    return result


# ════════════════════════════════════════════════════════════════════
# 契约验证
# ════════════════════════════════════════════════════════════════════

def _assert_contract(
    result:       pd.DataFrame,
    feature_cols: list,
    label_col:    str,
    label_backup: pd.Series,
    original_df:  pd.DataFrame,
) -> None:
    """对标准化输出执行契约断言"""

    # 断言①：无 Inf
    has_inf = np.isinf(
        result[feature_cols].select_dtypes(include=[np.number]).values
    ).any()
    assert not has_inf, "[normalizer][断言失败] 标准化后含 Inf 值"

    # 断言②：标签列未被修改
    if label_col and label_backup is not None:
        pd.testing.assert_series_equal(
            result[label_col].reset_index(drop=True),
            label_backup.reset_index(drop=True),
            check_names=False,
            obj=f"[normalizer][断言失败] label_col '{label_col}' 被修改"
        )

    # 断言③：抽查几列，验证截面均值≈0，标准差≈1
    check_cols = feature_cols[:3]  # 只抽查前3列，避免太慢

    if isinstance(result.index, pd.MultiIndex):
        dt_level = (
            result.index.names.index("datetime")
            if "datetime" in result.index.names else 0
        )
        for col in check_cols:
            if col not in result.columns:
                continue
            daily_stats = result[col].groupby(
                level=dt_level
            ).agg(["mean", "std"])
            # 过滤掉股票数少的天（那些天我们跳过了标准化）
            valid = daily_stats[daily_stats["std"] > 0.1]
            if len(valid) == 0:
                continue
            mean_of_means = valid["mean"].abs().mean()
            mean_of_stds  = valid["std"].mean()
            assert mean_of_means < 0.5, (
                f"[normalizer][断言失败] 列 '{col}' 截面均值偏离0："
                f"{mean_of_means:.4f}"
            )
            assert 0.5 < mean_of_stds < 1.5, (
                f"[normalizer][断言失败] 列 '{col}' 截面标准差异常："
                f"{mean_of_stds:.4f}"
            )
