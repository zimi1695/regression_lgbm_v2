"""
单元测试：test_lgbm_regressor.py
测试目标：src/model/lgbm_regressor.py

特点：用构造数据，不需要网络，速度极快。
运行方式：python -m pytest tests/test_lgbm_regressor.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.lgbm_regressor import LGBMModel, DEFAULT_PARAMS


# ════════════════════════════════════════════════════════════════════
# 构造测试数据
# ════════════════════════════════════════════════════════════════════

def make_dataset(n_train=200, n_test=50, n_features=10, seed=42):
    """
    构造一个有"信号"的简单回归数据集
    label = sum(features) + noise
    """
    rng = np.random.default_rng(seed)
    feat_cols = [f"f{i}" for i in range(n_features)]

    X_train = pd.DataFrame(
        rng.normal(0, 1, (n_train, n_features)),
        columns=feat_cols,
    )
    y_train = X_train.sum(axis=1) + rng.normal(0, 0.3, n_train)

    X_test = pd.DataFrame(
        rng.normal(0, 1, (n_test, n_features)),
        columns=feat_cols,
    )
    y_test = X_test.sum(axis=1) + rng.normal(0, 0.3, n_test)

    return X_train, y_train, X_test, y_test


# ════════════════════════════════════════════════════════════════════
# 第一组：基础 fit/predict 测试
# ════════════════════════════════════════════════════════════════════

class TestBasicFitPredict:

    def test_can_be_instantiated(self):
        """模型能够被实例化"""
        model = LGBMModel()
        assert model is not None
        assert not model.is_fitted

    def test_fit_runs_without_error(self):
        """fit 不抛异常"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel()
        model.fit(X_train, y_train)
        assert model.is_fitted

    def test_fit_returns_self(self):
        """fit 返回 self（链式调用）"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_predict_returns_series(self):
        """predict 返回 Series"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert isinstance(preds, pd.Series)

    def test_predict_length_matches_input(self):
        """预测长度与输入一致"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_index_matches_input(self):
        """预测 index 与输入一致"""
        X_train, y_train, X_test, _ = make_dataset()
        # 给 X_test 一个非默认 index
        X_test.index = pd.RangeIndex(start=1000, stop=1000 + len(X_test))
        model = LGBMModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert (preds.index == X_test.index).all()


# ════════════════════════════════════════════════════════════════════
# 第二组：模型质量测试（确保是真的在学）
# ════════════════════════════════════════════════════════════════════

class TestModelQuality:

    def test_predictions_correlate_with_truth(self):
        """
        在有信号的数据上，预测应与真实值正相关
        相关系数 > 0.5 视为"在学"
        """
        X_train, y_train, X_test, y_test = make_dataset(
            n_train=500, n_test=200
        )
        model = LGBMModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        corr = np.corrcoef(preds.values, y_test.values)[0, 1]
        assert corr > 0.5, f"预测相关性过低：{corr:.3f}"

    def test_predictions_no_inf(self):
        """预测输出不含 Inf"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert not np.isinf(preds.values).any()


# ════════════════════════════════════════════════════════════════════
# 第三组：列名契约测试
# ════════════════════════════════════════════════════════════════════

class TestColumnContract:

    def test_feature_cols_recorded_after_fit(self):
        """fit 后应记录特征列名"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        assert model.feature_cols == list(X_train.columns)

    def test_predict_with_missing_column_raises(self):
        """X_test 缺少训练时的列应报错"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        X_test_bad = X_test.drop(columns=["f0"])
        with pytest.raises(AssertionError, match="缺少训练时的列"):
            model.predict(X_test_bad)

    def test_predict_with_extra_column_warns_but_works(self):
        """X_test 多余的列应被忽略，不报错"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        X_test_extra = X_test.copy()
        X_test_extra["extra_col"] = 0.0
        # 应该正常预测，不抛异常
        preds = model.predict(X_test_extra)
        assert len(preds) == len(X_test_extra)

    def test_predict_with_different_column_order(self):
        """X_test 列顺序与训练时不同时，结果应一致"""
        X_train, y_train, X_test, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)

        # 第一次预测：原始顺序
        preds_1 = model.predict(X_test)

        # 第二次预测：反转列顺序
        X_test_reversed = X_test[X_test.columns[::-1]]
        preds_2 = model.predict(X_test_reversed)

        # 结果应一致（列对齐保证）
        np.testing.assert_array_almost_equal(
            preds_1.values, preds_2.values, decimal=6
        )


# ════════════════════════════════════════════════════════════════════
# 第四组：异常处理测试
# ════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_fit_with_mismatched_lengths_raises(self):
        """X 与 y 长度不一致应报错"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel()
        with pytest.raises(AssertionError, match="长度不一致"):
            model.fit(X_train, y_train.iloc[:50])

    def test_fit_with_nan_in_y_raises(self):
        """y_train 含 NaN 应报错"""
        X_train, y_train, _, _ = make_dataset()
        y_train_bad = y_train.copy()
        y_train_bad.iloc[0] = np.nan
        model = LGBMModel()
        with pytest.raises(AssertionError, match="y_train 含 NaN"):
            model.fit(X_train, y_train_bad)

    def test_fit_with_inf_in_x_raises(self):
        """X_train 含 Inf 应报错"""
        X_train, y_train, _, _ = make_dataset()
        X_train_bad = X_train.copy()
        X_train_bad.iloc[0, 0] = np.inf
        model = LGBMModel()
        with pytest.raises(AssertionError, match="含 Inf"):
            model.fit(X_train_bad, y_train)

    def test_predict_before_fit_raises(self):
        """未训练就预测应报错"""
        _, _, X_test, _ = make_dataset()
        model = LGBMModel()
        with pytest.raises(AssertionError, match="尚未训练"):
            model.predict(X_test)

    def test_fit_with_non_dataframe_raises(self):
        """X_train 不是 DataFrame 应报错"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel()
        with pytest.raises(AssertionError, match="必须是 DataFrame"):
            model.fit(X_train.values, y_train)


# ════════════════════════════════════════════════════════════════════
# 第五组：特征重要性测试
# ════════════════════════════════════════════════════════════════════

class TestFeatureImportance:

    def test_importance_returned_as_series(self):
        """特征重要性应返回 Series"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        imp = model.get_feature_importance()
        assert isinstance(imp, pd.Series)

    def test_importance_length_matches_features(self):
        """特征重要性长度与特征数一致"""
        X_train, y_train, _, _ = make_dataset(n_features=10)
        model = LGBMModel().fit(X_train, y_train)
        imp = model.get_feature_importance()
        assert len(imp) == 10

    def test_importance_sorted_descending(self):
        """特征重要性应按降序排列"""
        X_train, y_train, _, _ = make_dataset()
        model = LGBMModel().fit(X_train, y_train)
        imp = model.get_feature_importance()
        values = imp.values
        assert (values[:-1] >= values[1:]).all(), "特征重要性未按降序排列"

    def test_importance_before_fit_raises(self):
        """未训练就调用 importance 应报错"""
        model = LGBMModel()
        with pytest.raises(AssertionError, match="尚未训练"):
            model.get_feature_importance()


# ════════════════════════════════════════════════════════════════════
# 第六组：自定义参数测试
# ════════════════════════════════════════════════════════════════════

class TestCustomParams:

    def test_default_params_used(self):
        """无参数时使用 DEFAULT_PARAMS"""
        model = LGBMModel()
        for k, v in DEFAULT_PARAMS.items():
            assert model.params[k] == v

    def test_custom_params_override_defaults(self):
        """自定义参数应覆盖默认值"""
        custom = {"n_estimators": 100, "learning_rate": 0.1}
        model = LGBMModel(params=custom)
        assert model.params["n_estimators"] == 100
        assert model.params["learning_rate"] == 0.1
        # 未指定的参数仍然使用默认值
        assert model.params["num_leaves"] == DEFAULT_PARAMS["num_leaves"]