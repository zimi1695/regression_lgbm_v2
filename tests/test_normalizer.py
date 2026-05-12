"""
单元测试：test_normalizer.py
测试目标：src/features/normalizer.py

特点：全部用构造数据，不需要网络，速度极快。
运行方式：python -m pytest tests/test_normalizer.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.normalizer import normalize


# ════════════════════════════════════════════════════════════════════
# 构造测试数据
# ════════════════════════════════════════════════════════════════════

def make_flat_df(n_stocks=20, n_days=10, seed=42):
    """
    构造普通 DataFrame（V1风格）
    columns: date, stock_code, feat_a, feat_b, label
    """
    rng = np.random.default_rng(seed)
    dates  = pd.date_range("2024-01-01", periods=n_days, freq="B")
    stocks = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    rows = []
    for d in dates:
        for s in stocks:
            rows.append({
                "date":       d,
                "stock_code": s,
                "feat_a":     rng.normal(100, 20),   # 均值100，非标准化
                "feat_b":     rng.normal(50,  10),
                "label":      rng.normal(0,   0.05), # 标签，不应被修改
            })
    return pd.DataFrame(rows)


def make_multiindex_df(n_stocks=20, n_days=10, seed=42):
    """
    构造 MultiIndex DataFrame（qlib风格）
    index: MultiIndex(instrument, datetime)
    """
    rng = np.random.default_rng(seed)
    dates      = pd.date_range("2024-01-01", periods=n_days, freq="B")
    instruments = [f"SH{i:06d}" for i in range(1, n_stocks + 1)]

    idx = pd.MultiIndex.from_product(
        [instruments, dates],
        names=["instrument", "datetime"]
    )
    df = pd.DataFrame({
        "feat_a": rng.normal(100, 20, len(idx)),
        "feat_b": rng.normal(50,  10, len(idx)),
        "label":  rng.normal(0,   0.05, len(idx)),
    }, index=idx)
    return df


FEAT_COLS = ["feat_a", "feat_b"]


# ════════════════════════════════════════════════════════════════════
# 第一组：普通 DataFrame 测试
# ════════════════════════════════════════════════════════════════════

class TestFlatDataFrame:

    def test_returns_dataframe(self):
        df     = make_flat_df()
        result = normalize(df, FEAT_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_unchanged(self):
        df     = make_flat_df()
        result = normalize(df, FEAT_COLS)
        assert result.shape == df.shape

    def test_feature_cols_normalized(self):
        """标准化后每天截面均值≈0，标准差≈1"""
        df     = make_flat_df(n_stocks=50, n_days=5)
        result = normalize(df, FEAT_COLS)

        for col in FEAT_COLS:
            for date, group in result.groupby("date"):
                vals = group[col].dropna()
                if len(vals) < 5:
                    continue
                assert abs(vals.mean()) < 0.5, \
                    f"日期 {date} 列 '{col}' 均值 {vals.mean():.4f} 偏离0过大"
                assert 0.5 < vals.std() < 1.5, \
                    f"日期 {date} 列 '{col}' 标准差 {vals.std():.4f} 异常"

    def test_label_not_modified(self):
        """label 列不应被标准化"""
        df     = make_flat_df()
        result = normalize(df, FEAT_COLS, label_col="label")
        pd.testing.assert_series_equal(
            df["label"].reset_index(drop=True),
            result["label"].reset_index(drop=True),
            check_names=False,
        )

    def test_non_feature_cols_unchanged(self):
        """非特征列（date, stock_code）不应被修改"""
        df     = make_flat_df()
        result = normalize(df, FEAT_COLS)
        pd.testing.assert_series_equal(
            df["stock_code"].reset_index(drop=True),
            result["stock_code"].reset_index(drop=True),
        )

    def test_no_inf_output(self):
        """输出不含 Inf"""
        df     = make_flat_df()
        result = normalize(df, FEAT_COLS)
        assert not np.isinf(result[FEAT_COLS].values).any()


# ════════════════════════════════════════════════════════════════════
# 第二组：MultiIndex DataFrame 测试
# ════════════════════════════════════════════════════════════════════

class TestMultiIndexDataFrame:

    def test_returns_dataframe(self):
        df     = make_multiindex_df()
        result = normalize(df, FEAT_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_unchanged(self):
        df     = make_multiindex_df()
        result = normalize(df, FEAT_COLS)
        assert result.shape == df.shape

    def test_index_unchanged(self):
        """MultiIndex 结构不应被破坏"""
        df     = make_multiindex_df()
        result = normalize(df, FEAT_COLS)
        assert isinstance(result.index, pd.MultiIndex)
        assert set(result.index.names) == {"instrument", "datetime"}

    def test_feature_cols_normalized(self):
        """每天截面均值≈0，标准差≈1"""
        df     = make_multiindex_df(n_stocks=50, n_days=5)
        result = normalize(df, FEAT_COLS)

        dt_level = result.index.names.index("datetime")
        for col in FEAT_COLS:
            daily = result[col].groupby(level=dt_level).agg(["mean", "std"])
            valid = daily[daily["std"] > 0.1]
            assert valid["mean"].abs().mean() < 0.3, \
                f"列 '{col}' 截面均值偏离0：{valid['mean'].abs().mean():.4f}"
            assert valid["std"].mean() > 0.5, \
                f"列 '{col}' 截面标准差过小：{valid['std'].mean():.4f}"

    def test_label_not_modified(self):
        """label 列不应被标准化"""
        df     = make_multiindex_df()
        result = normalize(df, FEAT_COLS, label_col="label")
        pd.testing.assert_series_equal(
            df["label"].reset_index(drop=True),
            result["label"].reset_index(drop=True),
            check_names=False,
        )

    def test_no_inf_output(self):
        df     = make_multiindex_df()
        result = normalize(df, FEAT_COLS)
        assert not np.isinf(result[FEAT_COLS].values).any()


# ════════════════════════════════════════════════════════════════════
# 第三组：边界情况测试
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_min_stocks_threshold(self):
        """
        某天股票数低于 min_stocks 时，该天数据不应被标准化
        （保留原值）
        """
        # 只有5只股票，min_stocks=10，应跳过标准化
        df        = make_flat_df(n_stocks=5, n_days=3)
        original  = df["feat_a"].copy()
        result    = normalize(df, FEAT_COLS, min_stocks=10)
        pd.testing.assert_series_equal(
            original.reset_index(drop=True),
            result["feat_a"].reset_index(drop=True),
            check_names=False,
        )

    def test_constant_feature_becomes_zero(self):
        """
        某天所有股票的同一特征值相同（std=0）
        标准化后应全变为 0，而不是 NaN 或 Inf
        """
        df = make_flat_df(n_stocks=20, n_days=2)
        # 把第一天的 feat_a 全部设为同一值
        first_date = df["date"].min()
        df.loc[df["date"] == first_date, "feat_a"] = 100.0

        result = normalize(df, ["feat_a"])
        first_day_vals = result[result["date"] == first_date]["feat_a"]
        assert (first_day_vals == 0.0).all(), \
            "常数特征标准化后应为0，而非NaN或Inf"

    def test_nan_in_input_preserved(self):
        """输入中的 NaN 标准化后应仍为 NaN"""
        df = make_flat_df(n_stocks=20, n_days=3)
        # 手动设置一些 NaN
        df.loc[df.index[:5], "feat_a"] = np.nan
        result = normalize(df, FEAT_COLS)
        # 原来是 NaN 的位置，结果也应是 NaN
        assert result.loc[df.index[:5], "feat_a"].isna().all()

    def test_only_specified_cols_changed(self):
        """只有 feature_cols 中的列被修改，其他列保持不变"""
        df     = make_flat_df()
        result = normalize(df, ["feat_a"])  # 只标准化 feat_a
        # feat_b 不应被修改
        pd.testing.assert_series_equal(
            df["feat_b"].reset_index(drop=True),
            result["feat_b"].reset_index(drop=True),
        )