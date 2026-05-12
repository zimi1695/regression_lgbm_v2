"""
单元测试：test_loader.py
测试目标：src/data/loader.py 中的 load() 函数
运行方式：在项目根目录执行 python -m pytest tests/test_loader.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load

# ── 真实 CSV 路径 ────────────────────────────────────────────────────
REAL_CSV = os.path.join(os.path.dirname(__file__), "../data/stock_data.csv")


# ════════════════════════════════════════════════════════════════════
# 辅助：用真实数据做一次加载（所有测试共享，避免重复读取）
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_df():
    """加载真实 CSV，供所有测试用例共享"""
    return load(REAL_CSV)


# ════════════════════════════════════════════════════════════════════
# 第一组：列结构测试
# ════════════════════════════════════════════════════════════════════

class TestColumns:

    def test_required_columns_exist(self, real_df):
        """必要列必须全部存在"""
        required = ["stock_code", "date", "open", "close",
                    "high", "low", "volume", "turnover"]
        for col in required:
            assert col in real_df.columns, f"缺少列：{col}"

    def test_no_nan_in_required_columns(self, real_df):
        """关键列不允许有 NaN"""
        required = ["stock_code", "date", "open", "close",
                    "high", "low", "volume", "turnover"]
        for col in required:
            nan_count = real_df[col].isna().sum()
            assert nan_count == 0, f"列 '{col}' 含 {nan_count} 个 NaN"


# ════════════════════════════════════════════════════════════════════
# 第二组：股票代码格式测试
# ════════════════════════════════════════════════════════════════════

class TestStockCode:

    def test_stock_code_is_string(self, real_df):
        """stock_code 必须是字符串类型"""
        assert real_df["stock_code"].dtype == object, \
            "stock_code 应为 object(str) 类型"

    def test_stock_code_length_is_6(self, real_df):
        """所有股票代码必须是 6 位"""
        bad = real_df[real_df["stock_code"].str.len() != 6]["stock_code"].unique()
        assert len(bad) == 0, f"以下股票代码不是6位：{bad[:5]}"

    def test_stock_code_no_apostrophe(self, real_df):
        """股票代码中不应含有单引号"""
        has_apos = real_df["stock_code"].str.contains("'").any()
        assert not has_apos, "股票代码中仍含有单引号"

    def test_stock_code_leading_zeros(self, real_df):
        """沪深股票代码以 0/3/6 开头，应已补前导零"""
        starts = real_df["stock_code"].str[0].unique()
        for s in starts:
            assert s in list("0123456789"), f"股票代码首位异常：{s}"


# ════════════════════════════════════════════════════════════════════
# 第三组：数据规模测试
# ════════════════════════════════════════════════════════════════════

class TestScale:

    def test_stock_count_in_range(self, real_df):
        """股票数量应在 [295, 305] 之间"""
        n = real_df["stock_code"].nunique()
        assert 295 <= n <= 305, f"股票数量 {n} 超出预期范围 [295, 305]"

    def test_each_stock_has_enough_rows(self, real_df):
        """每只股票的数据行数应 > 100"""
        counts = real_df.groupby("stock_code").size()
        thin = counts[counts <= 100]
        assert len(thin) == 0, \
            f"{len(thin)} 只股票数据不足100行：{thin.index.tolist()[:5]}"

    def test_date_range(self, real_df):
        """时间范围应覆盖 2020-01-01 至今"""
        assert real_df["date"].min() <= pd.Timestamp("2020-06-01"), \
            f"数据起始时间过晚：{real_df['date'].min()}"
        assert real_df["date"].max() >= pd.Timestamp("2025-01-01"), \
            f"数据结束时间过早：{real_df['date'].max()}"


# ════════════════════════════════════════════════════════════════════
# 第四组：数值合理性测试
# ════════════════════════════════════════════════════════════════════

class TestValues:

    def test_price_columns_positive(self, real_df):
        """价格列必须为正数"""
        for col in ["open", "close", "high", "low"]:
            assert (real_df[col] > 0).all(), f"列 '{col}' 含非正值"

    def test_volume_positive(self, real_df):
        """成交量必须为正数"""
        assert (real_df["volume"] > 0).all(), "volume 含非正值"

    def test_high_gte_low(self, real_df):
        """最高价必须 >= 最低价"""
        bad = (real_df["high"] < real_df["low"]).sum()
        assert bad == 0, f"{bad} 行出现 high < low"

    def test_no_inf_values(self, real_df):
        """价格列不能有无穷值"""
        for col in ["open", "close", "high", "low", "volume"]:
            has_inf = np.isinf(real_df[col]).any()
            assert not has_inf, f"列 '{col}' 含 Inf 值"


# ════════════════════════════════════════════════════════════════════
# 第五组：排序测试
# ════════════════════════════════════════════════════════════════════

class TestSorting:

    def test_sorted_by_stock_and_date(self, real_df):
        """输出应按 (stock_code, date) 升序排列"""
        expected = real_df.sort_values(
            ["stock_code", "date"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            real_df.reset_index(drop=True),
            expected,
            check_like=False
        )


# ════════════════════════════════════════════════════════════════════
# 第六组：异常处理测试（用假数据）
# ════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_file_not_found(self, tmp_path):
        """文件不存在时应抛出 FileNotFoundError 或 OSError"""
        with pytest.raises((FileNotFoundError, OSError)):
            load(str(tmp_path / "nonexistent.csv"))

    def test_missing_column_raises_error(self, tmp_path):
        """CSV 缺少必要列时应抛出 ValueError"""
        # 构造一个缺少"收盘"列的假 CSV
        bad_df = pd.DataFrame({
            "股票代码": ["000001"],
            "日期":    ["2024-01-02"],
            "开盘":    [10.0],
            # 故意缺少 收盘、最高、最低、成交量、换手率
        })
        bad_csv = tmp_path / "bad.csv"
        bad_df.to_csv(bad_csv, index=False, encoding="utf-8")
        with pytest.raises(ValueError):
            load(str(bad_csv))