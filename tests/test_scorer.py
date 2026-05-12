"""
单元测试：test_scorer.py
测试目标：src/strategy/scorer.py

运行方式：python -m pytest tests/test_scorer.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.scorer import score_portfolio


# ════════════════════════════════════════════════════════════════
# 构造测试数据
# ════════════════════════════════════════════════════════════════

def make_price_df(
    codes=None,
    start="2024-01-01",
    end="2024-03-31",
    seed=42,
):
    """
    构造价格 DataFrame。
    每只股票每个交易日有一行，收盘价从 10.0 起随机游走。
    """
    if codes is None:
        codes = [f"{i:06d}" for i in range(1, 6)]

    dates = pd.bdate_range(start=start, end=end)  # 工作日
    rng   = np.random.default_rng(seed)
    rows  = []

    for code in codes:
        price = 10.0
        for d in dates:
            price = max(price * (1 + rng.normal(0, 0.01)), 0.1)
            rows.append({
                "stock_code": code,
                "date":       d,
                "close":      round(price, 2),
                "open":       round(price * 0.99, 2),
                "high":       round(price * 1.01, 2),
                "low":        round(price * 0.98, 2),
                "volume":     int(rng.integers(1e5, 1e6)),
            })

    return pd.DataFrame(rows)


def make_top_k_df(codes=None, k=5):
    """构造等权重 top_k DataFrame"""
    if codes is None:
        codes = [f"{i:06d}" for i in range(1, k + 1)]
    weight = 1.0 / len(codes)
    return pd.DataFrame({
        "stock_code": codes,
        "weight":     [weight] * len(codes),
    })


# ════════════════════════════════════════════════════════════════
# 第一组：基础功能测试
# ════════════════════════════════════════════════════════════════

class TestBasicFunction:

    def test_returns_float(self):
        """返回值是 float"""
        price_df = make_price_df()
        top_k_df = make_top_k_df()
        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert isinstance(result, float)

    def test_return_not_nan(self):
        """返回值不是 NaN"""
        price_df = make_price_df()
        top_k_df = make_top_k_df()
        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert not np.isnan(result)

    def test_return_not_inf(self):
        """返回值不是 Inf"""
        price_df = make_price_df()
        top_k_df = make_top_k_df()
        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert not np.isinf(result)

    def test_return_reasonable_range(self):
        """单周收益率绝对值 < 50%（合理范围）"""
        price_df = make_price_df()
        top_k_df = make_top_k_df()
        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert abs(result) < 0.5, f"收益率异常：{result}"


# ════════════════════════════════════════════════════════════════
# 第二组：收益率计算正确性测试
# ════════════════════════════════════════════════════════════════

class TestReturnCalculation:

    def test_zero_return_when_prices_same(self):
        """
        周初=周末价格相同时，收益率=0
        （构造两行完全相同的价格）
        """
        rows = []
        for code in ["000001", "000002"]:
            for d in ["2024-01-08", "2024-01-12"]:
                rows.append({
                    "stock_code": code,
                    "date":       pd.Timestamp(d),
                    "close":      10.0,
                    "open": 10.0, "high": 10.0,
                    "low":  10.0, "volume": 100000,
                })
        price_df = pd.DataFrame(rows)
        top_k_df = make_top_k_df(codes=["000001", "000002"], k=2)

        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert abs(result) < 1e-9, f"期望0，实际 {result}"

    def test_known_return_single_stock(self):
        """
        单只股票，价格从10涨到11，收益率应为10%
        等权weight=1.0，加权收益=0.1
        """
        rows = [
            {"stock_code": "000001", "date": pd.Timestamp("2024-01-08"),
             "close": 10.0, "open": 10.0, "high": 10.0,
             "low":  10.0, "volume": 100000},
            {"stock_code": "000001", "date": pd.Timestamp("2024-01-12"),
             "close": 11.0, "open": 11.0, "high": 11.0,
             "low":  11.0, "volume": 100000},
        ]
        price_df = pd.DataFrame(rows)
        top_k_df = pd.DataFrame({
            "stock_code": ["000001"],
            "weight":     [1.0],
        })

        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert abs(result - 0.1) < 1e-9, f"期望0.1，实际 {result}"

    def test_known_return_two_stocks(self):
        """
        两只股票等权(各0.5)：
        000001: 10→11 = +10%
        000002: 10→9  = -10%
        加权收益 = 0.5*0.1 + 0.5*(-0.1) = 0.0
        """
        rows = []
        data = {
            "000001": (10.0, 11.0),
            "000002": (10.0, 9.0),
        }
        for code, (p_start, p_end) in data.items():
            rows += [
                {"stock_code": code, "date": pd.Timestamp("2024-01-08"),
                 "close": p_start, "open": p_start, "high": p_start,
                 "low":  p_start, "volume": 100000},
                {"stock_code": code, "date": pd.Timestamp("2024-01-12"),
                 "close": p_end, "open": p_end, "high": p_end,
                 "low":  p_end,  "volume": 100000},
            ]
        price_df = pd.DataFrame(rows)
        top_k_df = pd.DataFrame({
            "stock_code": ["000001", "000002"],
            "weight":     [0.5, 0.5],
        })

        result = score_portfolio(
            top_k_df, "2024-01-08", "2024-01-12", price_df
        )
        assert abs(result - 0.0) < 1e-9, f"期望0.0，实际 {result}"


# ════════════════════════════════════════════════════════════════
# 第三组：非交易日处理测试
# ════════════════════════════════════════════════════════════════

class TestNonTradingDay:

    def test_weekend_date_uses_nearest_trading_day(self):
        """
        输入周末日期时，应自动找到最近交易日，不报错
        """
        price_df = make_price_df()
        top_k_df = make_top_k_df()

        # 2024-01-06 是周六，2024-01-07 是周日
        result = score_portfolio(
            top_k_df, "2024-01-06", "2024-01-14", price_df
        )
        assert isinstance(result, float)
        assert not np.isnan(result)


# ════════════════════════════════════════════════════════════════
# 第四组：异常处理测试
# ════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_week_start_after_week_end_raises(self):
        """week_start > week_end 应报错"""
        price_df = make_price_df()
        top_k_df = make_top_k_df()
        with pytest.raises(AssertionError, match="必须早于"):
            score_portfolio(
                top_k_df, "2024-01-12", "2024-01-08", price_df
            )

    def test_empty_top_k_raises(self):
        """空 top_k_df 应报错"""
        price_df = make_price_df()
        top_k_df = pd.DataFrame({"stock_code": [], "weight": []})
        with pytest.raises(AssertionError, match="不能为空"):
            score_portfolio(
                top_k_df, "2024-01-08", "2024-01-12", price_df
            )

    def test_missing_weight_column_raises(self):
        """top_k_df 缺列应报错"""
        price_df = make_price_df()
        bad_top_k = pd.DataFrame({"stock_code": ["000001"]})
        with pytest.raises(AssertionError, match="缺少列"):
            score_portfolio(
                bad_top_k, "2024-01-08", "2024-01-12", price_df
            )

    def test_missing_close_column_raises(self):
        """price_df 缺 close 列应报错"""
        price_df = make_price_df().drop(columns=["close"])
        top_k_df = make_top_k_df()
        with pytest.raises(AssertionError, match="缺少列"):
            score_portfolio(
                top_k_df, "2024-01-08", "2024-01-12", price_df
            )

    def test_weight_sum_exceeds_1_raises(self):
        """weight 之和 > 1.0 应报错"""
        price_df = make_price_df()
        bad_top_k = pd.DataFrame({
            "stock_code": ["000001", "000002"],
            "weight":     [0.8, 0.8],   # 总和 1.6
        })
        with pytest.raises(AssertionError, match="weight 之和"):
            score_portfolio(
                bad_top_k, "2024-01-08", "2024-01-12", price_df
            )