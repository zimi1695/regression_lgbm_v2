"""
单元测试：test_fetcher.py
测试目标：src/data/fetcher.py

注意：网络测试（真实拉取）用少量股票+短时间范围，避免太慢。
运行方式：python -m pytest tests/test_fetcher.py -v
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.fetcher import (
    fetch,
    _to_bs_code,
    _to_internal_code,
    _cache_path,
)

# ── 测试用常量 ────────────────────────────────────────────────────────
SAMPLE_CODES = ["600519", "000001", "300750"]   # 茅台、平安、宁德
START_DATE   = "2024-01-01"
END_DATE     = "2024-01-31"


# ════════════════════════════════════════════════════════════════════
# 第一组：代码格式转换（纯逻辑，不需要网络）
# ════════════════════════════════════════════════════════════════════

class TestCodeConversion:

    def test_sh_prefix(self):
        """6字头股票应转为 sh. 前缀"""
        assert _to_bs_code("600519") == "sh.600519"
        assert _to_bs_code("601318") == "sh.601318"

    def test_sz_prefix_0(self):
        """0字头股票应转为 sz. 前缀"""
        assert _to_bs_code("000001") == "sz.000001"
        assert _to_bs_code("000858") == "sz.000858"

    def test_sz_prefix_3(self):
        """3字头股票（创业板）应转为 sz. 前缀"""
        assert _to_bs_code("300750") == "sz.300750"

    def test_internal_code_roundtrip(self):
        """sh./sz. 转回内部码后应与原始码一致"""
        for code in ["600519", "000001", "300750"]:
            bs_code = _to_bs_code(code)
            assert _to_internal_code(bs_code) == code

    def test_internal_code_leading_zeros(self):
        """转回内部码时应保留前导零"""
        assert _to_internal_code("sz.000001") == "000001"


# ════════════════════════════════════════════════════════════════════
# 第二组：缓存路径生成（纯逻辑，不需要网络）
# ════════════════════════════════════════════════════════════════════

class TestCachePath:

    def test_cache_path_contains_code(self, tmp_path):
        """缓存路径应包含股票代码"""
        p = _cache_path(str(tmp_path), "600519", "2024-01-01", "2024-12-31")
        assert "600519" in p

    def test_cache_path_contains_dates(self, tmp_path):
        """缓存路径应包含日期范围（避免不同日期的缓存互相污染）"""
        p = _cache_path(str(tmp_path), "600519", "2024-01-01", "2024-12-31")
        assert "20240101" in p
        assert "20241231" in p

    def test_different_dates_different_paths(self, tmp_path):
        """不同日期范围应生成不同的缓存路径"""
        p1 = _cache_path(str(tmp_path), "600519", "2024-01-01", "2024-06-30")
        p2 = _cache_path(str(tmp_path), "600519", "2024-01-01", "2024-12-31")
        assert p1 != p2


# ════════════════════════════════════════════════════════════════════
# 第三组：真实网络拉取测试（少量股票，短时间范围）
# ════════════════════════════════════════════════════════════════════

class TestFetchReal:

    @pytest.fixture(scope="class")
    def fetched_df(self, tmp_path_factory):
        """拉取少量股票数据供本组测试共享"""
        cache_dir = str(tmp_path_factory.mktemp("fundamental"))
        df = fetch(
            stock_codes=SAMPLE_CODES,
            start_date=START_DATE,
            end_date=END_DATE,
            cache_dir=cache_dir,
            force_refresh=True,
        )
        return df

    def test_output_columns_exist(self, fetched_df):
        """输出必须包含所有契约列"""
        required = ["stock_code", "date", "peTTM", "pbMRQ",
                    "psTTM", "pcfNcfTTM"]
        for col in required:
            assert col in fetched_df.columns, f"缺少列：{col}"

    def test_stock_code_is_6_digits(self, fetched_df):
        """stock_code 必须是6位字符串"""
        bad = fetched_df[
            fetched_df["stock_code"].str.len() != 6
        ]["stock_code"].unique()
        assert len(bad) == 0, f"股票代码不是6位：{bad}"

    def test_date_no_nan(self, fetched_df):
        """date 列不允许有 NaN"""
        assert fetched_df["date"].isna().sum() == 0

    def test_all_sample_stocks_present(self, fetched_df):
        """三只测试股票都应有数据"""
        fetched_codes = fetched_df["stock_code"].unique()
        for code in SAMPLE_CODES:
            assert code in fetched_codes, f"股票 {code} 数据缺失"

    def test_date_in_range(self, fetched_df):
        """返回数据的日期应在请求范围内"""
        assert fetched_df["date"].min() >= pd.Timestamp(START_DATE)
        assert fetched_df["date"].max() <= pd.Timestamp(END_DATE)

    def test_fundamental_cols_are_numeric(self, fetched_df):
        """基本面列应为数值类型"""
        for col in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            assert pd.api.types.is_float_dtype(fetched_df[col]), \
                f"列 '{col}' 不是 float 类型"


# ════════════════════════════════════════════════════════════════════
# 第四组：缓存机制测试
# ════════════════════════════════════════════════════════════════════

class TestCache:

    def test_cache_file_created(self, tmp_path):
        """拉取后应在缓存目录创建文件"""
        cache_dir = str(tmp_path / "fundamental")
        fetch(
            stock_codes=["600519"],
            start_date=START_DATE,
            end_date=END_DATE,
            cache_dir=cache_dir,
            force_refresh=True,
        )
        files = os.listdir(cache_dir)
        assert len(files) == 1, "应创建1个缓存文件"
        assert "600519" in files[0]

    def test_cache_hit_no_network(self, tmp_path):
        """
        第二次调用（有缓存）应直接读缓存
        通过比较两次结果一致性来验证
        """
        cache_dir = str(tmp_path / "fundamental")

        # 第一次：真实拉取
        df1 = fetch(
            stock_codes=["600519"],
            start_date=START_DATE,
            end_date=END_DATE,
            cache_dir=cache_dir,
            force_refresh=True,
        )

        # 第二次：应命中缓存
        df2 = fetch(
            stock_codes=["600519"],
            start_date=START_DATE,
            end_date=END_DATE,
            cache_dir=cache_dir,
            force_refresh=False,   # 不强制刷新
        )

        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )