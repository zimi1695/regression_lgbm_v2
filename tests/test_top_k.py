"""
单元测试：test_top_k.py
测试目标：src/strategy/top_k.py

运行方式：python -m pytest tests/test_top_k.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.top_k import select_top_k


# ════════════════════════════════════════════════════════════════
# 构造测试数据
# ════════════════════════════════════════════════════════════════

def make_pred_score(n=10, seed=42):
    """构造一个有 n 只股票的预测分数 Series"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    scores = rng.normal(0, 1, n)
    return pd.Series(scores, index=codes, name="pred_score")


# ════════════════════════════════════════════════════════════════
# 第一组：基础功能测试
# ════════════════════════════════════════════════════════════════

class TestBasicFunction:

    def test_returns_dataframe(self):
        """返回值是 DataFrame"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self):
        """输出包含 stock_code 和 weight 列"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert "stock_code" in result.columns
        assert "weight"     in result.columns

    def test_output_row_count_equals_k(self):
        """输出行数等于 k（股票数充足时）"""
        pred = make_pred_score(20)
        result = select_top_k(pred, k=5)
        assert len(result) == 5

    def test_output_row_count_when_k_equals_1(self):
        """k=1 时只选1只"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=1)
        assert len(result) == 1

    def test_output_row_count_when_k_equals_stock_count(self):
        """k 等于股票数时，全部选出"""
        pred = make_pred_score(5)
        result = select_top_k(pred, k=5)
        assert len(result) == 5


# ════════════════════════════════════════════════════════════════
# 第二组：权重契约测试
# ════════════════════════════════════════════════════════════════

class TestWeightContract:

    def test_all_weights_positive(self):
        """所有权重 > 0"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert (result["weight"] > 0).all()

    def test_weight_sum_leq_1(self):
        """权重之和 <= 1.0"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert result["weight"].sum() <= 1.0 + 1e-6

    def test_equal_weight(self):
        """等权重：每只股票权重 == 1/k"""
        pred = make_pred_score(10)
        k = 5
        result = select_top_k(pred, k=k)
        expected_weight = 1.0 / k
        assert (result["weight"] - expected_weight).abs().max() < 1e-9

    def test_weight_sum_close_to_1_for_k5(self):
        """k=5 时权重之和应 ≈ 1.0"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert abs(result["weight"].sum() - 1.0) < 1e-9


# ════════════════════════════════════════════════════════════════
# 第三组：选股逻辑测试
# ════════════════════════════════════════════════════════════════

class TestSelectionLogic:

    def test_selects_highest_scores(self):
        """选出的是分数最高的 k 只"""
        codes = [f"{i:06d}" for i in range(1, 11)]
        # 构造已知顺序的分数：股票 000010 分最高，000001 最低
        scores = list(range(1, 11))   # [1, 2, 3, ..., 10]
        pred = pd.Series(scores, index=codes)

        result = select_top_k(pred, k=3)
        selected = set(result["stock_code"])

        # 分数最高的3只：000010, 000009, 000008
        expected = {"000010", "000009", "000008"}
        assert selected == expected

    def test_selected_stocks_from_input(self):
        """选出的股票必须来自输入"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        valid_codes = set(pred.index)
        for code in result["stock_code"]:
            assert code in valid_codes

    def test_no_duplicate_stocks(self):
        """选出的股票不重复"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=5)
        assert result["stock_code"].nunique() == len(result)


# ════════════════════════════════════════════════════════════════
# 第四组：边界情况测试
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_k_greater_than_available_stocks(self):
        """k > 股票数时，最多返回所有股票"""
        pred = make_pred_score(3)    # 只有3只
        result = select_top_k(pred, k=5)   # 要5只
        assert len(result) == 3            # 最多3只

    def test_k_equal_to_1(self):
        """k=1 时选出分数最高的1只"""
        pred = make_pred_score(10)
        result = select_top_k(pred, k=1)
        best_code = pred.idxmax()
        assert result["stock_code"].iloc[0] == best_code

    def test_pred_score_with_negative_values(self):
        """负分数也能正常处理（选相对最高的）"""
        codes = [f"{i:06d}" for i in range(1, 6)]
        scores = [-5.0, -3.0, -1.0, -4.0, -2.0]
        pred = pd.Series(scores, index=codes)
        result = select_top_k(pred, k=2)
        # -1.0 和 -2.0 是最高的两个
        selected = set(result["stock_code"])
        assert "000003" in selected   # -1.0
        assert "000005" in selected   # -2.0

    def test_pred_score_single_stock(self):
        """只有1只股票时正常工作"""
        pred = pd.Series([0.5], index=["000001"])
        result = select_top_k(pred, k=5)
        assert len(result) == 1
        assert result["stock_code"].iloc[0] == "000001"


# ════════════════════════════════════════════════════════════════
# 第五组：异常处理测试
# ════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_non_series_input_raises(self):
        """非 Series 输入应报错"""
        with pytest.raises(AssertionError, match="必须是 pd.Series"):
            select_top_k([0.1, 0.2, 0.3], k=2)

    def test_empty_series_raises(self):
        """空 Series 应报错"""
        with pytest.raises(AssertionError, match="不能为空"):
            select_top_k(pd.Series([], dtype=float), k=5)

    def test_invalid_k_zero_raises(self):
        """k=0 应报错"""
        pred = make_pred_score(10)
        with pytest.raises(AssertionError, match="必须是正整数"):
            select_top_k(pred, k=0)

    def test_invalid_k_negative_raises(self):
        """k 为负数应报错"""
        pred = make_pred_score(10)
        with pytest.raises(AssertionError, match="必须是正整数"):
            select_top_k(pred, k=-1)

    def test_nan_in_scores_raises(self):
        """分数含 NaN 应报错"""
        codes = [f"{i:06d}" for i in range(1, 6)]
        scores = [0.1, float("nan"), 0.3, 0.4, 0.5]
        pred = pd.Series(scores, index=codes)
        with pytest.raises(AssertionError, match="含.*NaN"):
            select_top_k(pred, k=3)

    def test_duplicate_index_raises(self):
        """index 含重复股票代码应报错"""
        pred = pd.Series(
            [0.1, 0.2, 0.3],
            index=["000001", "000001", "000002"]   # 重复
        )
        with pytest.raises(AssertionError, match="含重复值"):
            select_top_k(pred, k=2)