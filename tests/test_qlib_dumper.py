"""
单元测试：test_qlib_dumper.py
测试目标：src/data/qlib_dumper.py

注意：使用真实 stock_data.csv 的一个小子集（5只股票）
      避免全量转换太慢。

运行方式：python -m pytest tests/test_qlib_dumper.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load
from src.data.qlib_dumper import (
    dump,
    init_qlib,
    _to_qlib_code,
    _to_qlib_instrument,
)

REAL_CSV   = os.path.join(os.path.dirname(__file__), "../data/stock_data.csv")

# 只取5只股票做测试，速度快
SAMPLE_CODES = ["600519", "000001", "300750", "601318", "000858"]


@pytest.fixture(scope="module")
def sample_df():
    """加载真实数据，只取5只股票的子集"""
    df = load(REAL_CSV)
    return df[df["stock_code"].isin(SAMPLE_CODES)].copy()


@pytest.fixture(scope="module")
def dumped_qlib_dir(tmp_path_factory, sample_df):
    """执行一次 dump，供所有测试共享"""
    qlib_dir = str(tmp_path_factory.mktemp("qlib_data"))
    dump(sample_df, qlib_dir)
    return qlib_dir


# ════════════════════════════════════════════════════════════════════
# 第一组：代码格式转换（纯逻辑）
# ════════════════════════════════════════════════════════════════════

class TestCodeFormat:

    def test_sh_prefix_lowercase(self):
        """上交所股票目录名应为 sh + 代码"""
        assert _to_qlib_code("600519") == "sh600519"

    def test_sz_prefix_lowercase(self):
        """深交所股票目录名应为 sz + 代码"""
        assert _to_qlib_code("000001") == "sz000001"
        assert _to_qlib_code("300750") == "sz300750"

    def test_instrument_uppercase(self):
        """instruments 文件中代码应为大写"""
        assert _to_qlib_instrument("600519") == "SH600519"
        assert _to_qlib_instrument("000001") == "SZ000001"


# ════════════════════════════════════════════════════════════════════
# 第二组：目录结构验证
# ════════════════════════════════════════════════════════════════════

class TestDirectoryStructure:

    def test_calendars_dir_exists(self, dumped_qlib_dir):
        """calendars 目录应存在"""
        assert os.path.isdir(
            os.path.join(dumped_qlib_dir, "calendars")
        )

    def test_instruments_dir_exists(self, dumped_qlib_dir):
        """instruments 目录应存在"""
        assert os.path.isdir(
            os.path.join(dumped_qlib_dir, "instruments")
        )

    def test_features_dir_exists(self, dumped_qlib_dir):
        """features 目录应存在"""
        assert os.path.isdir(
            os.path.join(dumped_qlib_dir, "features")
        )

    def test_stock_dirs_created(self, dumped_qlib_dir):
        """features 下应有对应股票目录"""
        features_dir = os.path.join(dumped_qlib_dir, "features")
        dirs = os.listdir(features_dir)
        assert "sh600519" in dirs
        assert "sz000001" in dirs
        assert "sz300750" in dirs


# ════════════════════════════════════════════════════════════════════
# 第三组：calendars/day.txt 验证
# ════════════════════════════════════════════════════════════════════

class TestCalendars:

    def test_day_txt_exists(self, dumped_qlib_dir):
        """calendars/day.txt 必须存在"""
        path = os.path.join(dumped_qlib_dir, "calendars", "day.txt")
        assert os.path.exists(path)

    def test_day_txt_not_empty(self, dumped_qlib_dir):
        """day.txt 不能为空"""
        path = os.path.join(dumped_qlib_dir, "calendars", "day.txt")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0

    def test_day_txt_format(self, dumped_qlib_dir):
        """day.txt 每行应为 YYYY-MM-DD 格式"""
        path = os.path.join(dumped_qlib_dir, "calendars", "day.txt")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines[:5]:
            pd.Timestamp(line)   # 能解析说明格式正确

    def test_day_txt_sorted(self, dumped_qlib_dir):
        """day.txt 中日期应升序排列"""
        path = os.path.join(dumped_qlib_dir, "calendars", "day.txt")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert lines == sorted(lines)


# ════════════════════════════════════════════════════════════════════
# 第四组：instruments/all.txt 验证
# ════════════════════════════════════════════════════════════════════

class TestInstruments:

    def test_all_txt_exists(self, dumped_qlib_dir):
        """instruments/all.txt 必须存在"""
        path = os.path.join(dumped_qlib_dir, "instruments", "all.txt")
        assert os.path.exists(path)

    def test_stock_count_matches(self, dumped_qlib_dir, sample_df):
        """instruments 中股票数量应与输入一致"""
        path = os.path.join(dumped_qlib_dir, "instruments", "all.txt")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == sample_df["stock_code"].nunique()

    def test_instrument_format(self, dumped_qlib_dir):
        """每行格式应为：代码<tab>起始日<tab>结束日"""
        path = os.path.join(dumped_qlib_dir, "instruments", "all.txt")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 3, f"格式错误：{line}"
            code, start, end = parts
            assert code.startswith(("SH", "SZ")), \
                f"代码前缀错误：{code}"
            pd.Timestamp(start)   # 起始日期可解析
            pd.Timestamp(end)     # 结束日期可解析


# ════════════════════════════════════════════════════════════════════
# 第五组：.bin 文件验证
# ════════════════════════════════════════════════════════════════════

class TestBinFiles:

    def test_bin_files_exist(self, dumped_qlib_dir):
        """每只股票目录下应有 close.bin 等字段文件"""
        stock_dir = os.path.join(
            dumped_qlib_dir, "features", "sh600519"
        )
        for field in ["open", "close", "high", "low",
                      "volume", "factor"]:
            bin_file = os.path.join(stock_dir, f"{field}.day.bin")
            assert os.path.exists(bin_file), \
                f"缺少文件：{field}.bin"

    def test_bin_file_not_empty(self, dumped_qlib_dir):
        """close.bin 文件不能为空"""
        bin_path = os.path.join(
            dumped_qlib_dir, "features", "sh600519", "close.day.bin"
        )
        size = os.path.getsize(bin_path)
        assert size > 4, "close.bin 文件过小（应含多个float32）"

    def test_bin_file_readable_as_float32(self, dumped_qlib_dir):
        """close.bin 应可以作为 float32 读取"""
        bin_path = os.path.join(
            dumped_qlib_dir, "features", "sh600519", "close.day.bin"
        )
        data = np.fromfile(bin_path, dtype=np.float32)
        # 第一个元素是起始索引，其余是价格数据
        assert len(data) > 1
        prices = data[1:]
        # 价格应为正数（排除NaN）
        valid_prices = prices[~np.isnan(prices)]
        assert (valid_prices > 0).all(), "close.bin 含非正价格"

    def test_factor_bin_all_ones(self, dumped_qlib_dir):
        """factor.bin 应全部为 1.0（已复权数据）"""
        bin_path = os.path.join(
            dumped_qlib_dir, "features", "sh600519", "factor.day.bin"
        )
        data  = np.fromfile(bin_path, dtype=np.float32)
        # 跳过第一个元素（起始索引）
        factors = data[1:]
        assert np.allclose(factors, 1.0), \
            "factor.bin 中存在非1.0的值"


# ════════════════════════════════════════════════════════════════════
# 第六组：qlib 初始化验证
# ════════════════════════════════════════════════════════════════════

class TestQlibInit:

    def test_init_qlib_no_error(self, dumped_qlib_dir):
        """qlib.init() 应能正常初始化，不抛异常"""
        try:
            init_qlib(dumped_qlib_dir)
        except Exception as e:
            pytest.fail(f"qlib.init() 失败：{e}")

    def test_qlib_features_queryable(self, dumped_qlib_dir):
        """初始化后应能通过 D.features() 查询数据"""
        import qlib
        from qlib.data import D

        init_qlib(dumped_qlib_dir)

        try:
            instruments = D.instruments("all")
            df = D.features(
                instruments,
                fields=["$close", "$volume"],
                start_time="2024-01-01",
                end_time="2024-01-31",
            )
            assert df is not None
            assert len(df) > 0, "D.features() 返回空数据"
        except Exception as e:
            pytest.fail(f"D.features() 查询失败：{e}")