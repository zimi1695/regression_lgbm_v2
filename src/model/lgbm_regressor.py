"""
模块G：lgbm_regressor.py
职责：封装 LightGBM 回归模型，标准化训练/预测接口。

设计要点：
  - 严格的契约校验：训练前/预测前都检查列名、类型、长度
  - 训练后保存特征列名，预测时强制对齐
  - 训练后可输出特征重要性，便于诊断
  - 支持验证集（early stopping）

契约：
  fit(X_train, y_train, X_valid=None, y_valid=None)
    X_train : pd.DataFrame，每列一个特征
    y_train : pd.Series 或 np.ndarray，长度与 X_train 一致
    断言：
      - X_train 列数 > 0
      - len(X_train) == len(y_train)
      - X_train 中无 Inf
      - y_train 中无 NaN

  predict(X_test) -> pd.Series
    X_test : pd.DataFrame，列名必须与 fit 时一致
    返回   : pd.Series，index 与 X_test 一致
    断言：
      - X_test 列名集合 == 训练时列名集合
      - len(返回) == len(X_test)
      - 输出无 Inf
"""

import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 默认参数（从 V1 经验得出）
# ════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {                  
    "n_estimators":       500,
    "learning_rate":      0.05,
    "num_leaves":         64,
    "min_child_samples":  20,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.5,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}


# ════════════════════════════════════════════════════════════════════
# 模型封装类
# ════════════════════════════════════════════════════════════════════

class LGBMModel:
    """
    LightGBM 回归模型封装。

    Usage
    -----
    >>> model = LGBMModel()
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> imp   = model.get_feature_importance()
    """

    def __init__(self, params: dict = None):
        """
        Parameters
        ----------
        params : LightGBM 参数字典，None 则使用 DEFAULT_PARAMS
        """
        self.params       = {**DEFAULT_PARAMS, **(params or {})}
        self.model        = None
        self.feature_cols = None    # 训练时记录的列名（用于预测时校验）
        self.is_fitted    = False

    # ────────────────────────────────────────────────────────────────
    # 训练
    # ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train,
        X_valid: pd.DataFrame = None,
        y_valid             = None,
    ) -> "LGBMModel":
        """
        训练模型。

        Parameters
        ----------
        X_train : 特征 DataFrame
        y_train : 标签 Series/ndarray
        X_valid : 验证集特征（可选）
        y_valid : 验证集标签（可选）

        Returns
        -------
        self
        """
        # ── 输入校验 ─────────────────────────────────────────────────
        self._assert_fit_inputs(X_train, y_train, X_valid, y_valid)

        # ── 记录列名（用于预测校验）──────────────────────────────────
        self.feature_cols = list(X_train.columns)

        logger.info(
            f"[lgbm] 开始训练：{len(X_train)} 行，"
            f"{len(self.feature_cols)} 个特征"
        )

        # ── 训练 ─────────────────────────────────────────────────────
        self.model = LGBMRegressor(**self.params)

        if X_valid is not None and y_valid is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[],
            )
            logger.info(f"[lgbm] 训练完成（带验证集）")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"[lgbm] 训练完成")

        self.is_fitted = True

        # ── 输出特征重要性 Top5 ──────────────────────────────────────
        imp = self.get_feature_importance()
        top5 = imp.head(5)
        logger.info(f"[lgbm] Top5 重要特征：\n{top5.to_string()}")

        return self

    # ────────────────────────────────────────────────────────────────
    # 预测
    # ────────────────────────────────────────────────────────────────

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        预测。

        Parameters
        ----------
        X_test : 特征 DataFrame，列名必须与训练时一致

        Returns
        -------
        pd.Series，预测值，index 与 X_test 一致
        """
        # ── 输入校验 ─────────────────────────────────────────────────
        self._assert_predict_inputs(X_test)

        # ── 列顺序对齐（防止顺序差异）────────────────────────────────
        X_aligned = X_test[self.feature_cols]

        # ── 预测 ─────────────────────────────────────────────────────
        preds = self.model.predict(X_aligned)

        # ── 输出契约 ─────────────────────────────────────────────────
        result = pd.Series(preds, index=X_test.index, name="pred_score")
        assert len(result) == len(X_test), \
            "[lgbm][断言失败] 预测输出长度与输入不一致"
        assert not np.isinf(result.values).any(), \
            "[lgbm][断言失败] 预测输出含 Inf"

        return result

    # ────────────────────────────────────────────────────────────────
    # 特征重要性
    # ────────────────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.Series:
        """
        返回特征重要性，按降序排列。

        Returns
        -------
        pd.Series，index=特征名，value=重要性分数
        """
        assert self.is_fitted, "[lgbm] 模型尚未训练"
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
            name="importance",
        ).sort_values(ascending=False)
        return importance

    # ════════════════════════════════════════════════════════════════
    # 内部契约校验
    # ════════════════════════════════════════════════════════════════

    def _assert_fit_inputs(
        self,
        X_train, y_train,
        X_valid, y_valid,
    ) -> None:
        # 类型
        assert isinstance(X_train, pd.DataFrame), \
            "[lgbm][断言失败] X_train 必须是 DataFrame"
        # 列数
        assert X_train.shape[1] > 0, \
            "[lgbm][断言失败] X_train 至少需要1列特征"
        # 行数一致
        assert len(X_train) == len(y_train), (
            f"[lgbm][断言失败] X_train ({len(X_train)}) 与 "
            f"y_train ({len(y_train)}) 长度不一致"
        )
        # X 中无 Inf
        if isinstance(X_train, pd.DataFrame):
            num_data = X_train.select_dtypes(include=[np.number]).values
            assert not np.isinf(num_data).any(), \
                "[lgbm][断言失败] X_train 含 Inf 值"

        # y 中无 NaN
        y_arr = np.asarray(y_train)
        assert not np.isnan(y_arr).any(), \
            "[lgbm][断言失败] y_train 含 NaN，模型无法训练"

        # 验证集校验
        if X_valid is not None:
            assert isinstance(X_valid, pd.DataFrame), \
                "[lgbm][断言失败] X_valid 必须是 DataFrame"
            assert list(X_valid.columns) == list(X_train.columns), \
                "[lgbm][断言失败] X_valid 列名必须与 X_train 一致"
            assert len(X_valid) == len(y_valid), \
                "[lgbm][断言失败] X_valid 与 y_valid 长度不一致"

    def _assert_predict_inputs(self, X_test) -> None:
        # 必须先训练
        assert self.is_fitted, \
            "[lgbm][断言失败] 模型尚未训练，请先调用 fit()"
        # 类型
        assert isinstance(X_test, pd.DataFrame), \
            "[lgbm][断言失败] X_test 必须是 DataFrame"
        # 列名集合一致
        train_set = set(self.feature_cols)
        test_set  = set(X_test.columns)
        missing   = train_set - test_set
        extra     = test_set  - train_set
        assert not missing, \
            f"[lgbm][断言失败] X_test 缺少训练时的列：{missing}"
        if extra:
            logger.warning(
                f"[lgbm] X_test 含训练时不存在的列（将被忽略）：{extra}"
            )
        # 长度
        assert len(X_test) > 0, \
            "[lgbm][断言失败] X_test 不能为空"
        # X_test 中 Inf 警告（不强制阻断，因为预测时遇到 Inf 通常 LGBM 能处理）
        num_data = X_test[self.feature_cols].select_dtypes(
            include=[np.number]
        ).values
        if np.isinf(num_data).any():
            logger.warning("[lgbm] X_test 含 Inf 值，可能导致预测异常")