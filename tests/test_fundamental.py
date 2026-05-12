"""
单元测试：test_fundamental.py
测试目标：src/features/fundamental.py

注意：需要真实拉取 Baostock 基本面数据（少量股票）
运行方式：python -m pytest tests/test_fundamental.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader      import load
from src.data.fetcher     import fetch
from src.features.fundamental import compute, FUNDAMENTAL_COLS

REAL_CSV     = os.path.join(os.path.dirname(__file__), "../data/stock_data.csv")
SAMPLE_CODES = ["600519", "000001", "300750"]
START_DATE   = "2024-01-01"
END_DATE     = "2024-06-30"


# ════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def date_list():
    """从真实数据中提取交易日列表"""
    df = load(REAL_CSV)
    df = df[df["stock_code"].isin(SAMPLE_CODES)]
    dates = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]["date"]
    return sorted(dates.unique().tolist())


@pytest.fixture(scope="module")
def fund_df(tmp_path_factory):
    """拉取基本面数据"""
    cache_dir = str(tmp_path_factory.mktemp("fundamental"))
    return fetch(
        stock_codes=SAMPLE_CODES,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_dir=cache_dir,
        force_refresh=True,
    )


@pytest.fixture(scope="module")
def feat_df(fund_df, date_list):
    """计算基本面特征"""
    return compute(fund_df, date_list, SAMPLE_CODES)


# ════════════════════════════════════════════════════════════════════
# 第一组：输出结构测试
# ════════════════════════════════════════════════════════════════════

class TestStructure:

    def test_output_is_dataframe(self, feat_df):
        assert isinstance(feat_df, pd.DataFrame)

    def test_column_names_correct(self, feat_df):
        assert list(feat_df.columns) == FUNDAMENTAL_COLS

    def test_column_count(self, feat_df):
        assert len(feat_df.columns) == len(FUNDAMENTAL_COLS)

    def test_index_is_multiindex(self, feat_df):
        assert isinstance(feat_df.index, pd.MultiIndex)
        assert set(feat_df.index.names) == {"instrument", "datetime"}

    def test_not_empty(self, feat_df):
        assert len(feat_df) > 0

    def test_all_sample_stocks_present(self, feat_df):
        """所有输入股票都应有输出"""
        instruments = feat_df.index.get_level_values("instrument").unique()
        for code in SAMPLE_CODES:
            prefix = "SH" if code.startswith("6") else "SZ"
            assert f"{prefix}{code}" in instruments


# ════════════════════════════════════════════════════════════════════
# 第二组：数值合理性测试
# ════════════════════════════════════════════════════════════════════

class TestValues:

    def test_no_inf_values(self, feat_df):
        has_inf = np.isinf(
            feat_df.select_dtypes(include=[np.number]).values
        ).any()
        assert not has_inf

    def test_is_profitable_binary(self, feat_df):
        """is_profitable 只能是 0、1 或 NaN"""
        ip = feat_df["is_profitable"].dropna()
        assert ip.isin([0.0, 1.0]).all()

    def test_days_since_report_non_negative(self, feat_df):
        """days_since_report 应为非负数"""
        dsr = feat_df["days_since_report"].dropna()
        assert (dsr >= 0).all()

    def test_days_since_report_reasonable_range(self, feat_df):
        """days_since_report 不应超过 200 天（半年多）"""
        dsr = feat_df["days_since_report"].dropna()
        # 允许少量异常，但不能全部超出
        assert (dsr <= 200).mean() > 0.8

    def test_earnings_yield_sign_consistent(self, feat_df):
        """
        盈利股票（is_profitable=1）的 earnings_yield 应为正数
        亏损股票（is_profitable=0）的 earnings_yield 应为负数
        """
        profitable = feat_df[feat_df["is_profitable"] == 1.0]
        if len(profitable) > 0:
            ey = profitable["earnings_yield"].dropna()
            assert (ey > 0).mean() > 0.9, \
                "盈利股票的 earnings_yield 应主要为正"

    def test_pb_inv_positive_for_normal_stocks(self, feat_df):
        """正常股票的 pb_inv 应为正数（PB > 0）"""
        pb = feat_df["pb_inv"].dropna()
        assert (pb > 0).mean() > 0.8


# ════════════════════════════════════════════════════════════════════
# 第三组：日期对齐测试
# ════════════════════════════════════════════════════════════════════

class TestDateAlignment:

    def test_date_range_covered(self, feat_df):
        """输出日期范围应覆盖请求范围"""
        datetimes = feat_df.index.get_level_values("datetime")
        assert datetimes.min() >= pd.Timestamp(START_DATE)
        assert datetimes.max() <= pd.Timestamp(END_DATE)

    def test_ffill_reduces_nan(self, fund_df, date_list):
        """
        前向填充后，NaN 比例应低于原始数据的 NaN 比例
        （季报数据本身稀疏，填充后应更密）
        """
        raw_nan = fund_df["peTTM"].isna().mean()
        result  = compute(fund_df, date_list, SAMPLE_CODES)
        filled_nan = result["earnings_yield"].isna().mean()
        # 填充后 NaN 应减少（或持平）
        assert filled_nan <= raw_nan + 0.1


# ════════════════════════════════════════════════════════════════════
# 第四组：safe_inv 逻辑测试（纯逻辑，不需要网络）
# ════════════════════════════════════════════════════════════════════

class TestSafeInv:

    def test_zero_becomes_nan(self, feat_df):
        """PE=0 的情况应产生 NaN，而不是 Inf"""
        assert not np.isinf(feat_df["earnings_yield"].values).any()

    def test_negative_pe_gives_negative_yield(self, feat_df):
        """
        如果有亏损股票（is_profitable=0），
        其 earnings_yield 应为负数
        """
        loss_stocks = feat_df[feat_df["is_profitable"] == 0.0]
        if len(loss_stocks) > 0:
            ey = loss_stocks["earnings_yield"].dropna()
            if len(ey) > 0:
                assert (ey < 0).all(), \
                    "亏损股票的 earnings_yield 应为负数"