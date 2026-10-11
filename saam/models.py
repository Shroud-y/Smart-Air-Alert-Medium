"""Models compared against the baselines."""
import lightgbm as lgb

from .dataset import varying_columns

# Chosen by a small search on validation (scripts/tune.py; README, "Tuning").
# One setting for all four models: small trees and large leaves, because
# positives are rare and neighbouring grid points are near-copies.
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


class Lgbm:
    """LightGBM on every feature that varies in the training rows."""

    def __init__(self, **overrides):
        self.params = {**LGBM_PARAMS, **overrides}

    def fit(self, X, y):
        self.cols_ = varying_columns(X)
        self.dropped_ = [c for c in X.columns if c not in self.cols_]
        self.model_ = lgb.LGBMClassifier(**self.params).fit(X[self.cols_], y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.cols_])[:, 1]

    def importance(self):
        """Share of total gain per feature, descending."""
        import pandas as pd

        g = pd.Series(self.model_.booster_.feature_importance("gain"), index=self.cols_)
        return (g / g.sum()).sort_values(ascending=False)
