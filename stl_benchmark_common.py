"""Shared algorithm adapters for the strong single-task PK benchmark."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


IMPLEMENTED_ALGORITHMS = {
    "dummy_mean", "ridge", "elasticnet", "knn", "random_forest", "extra_trees",
    "hist_gradient_boosting", "svr_rbf", "kernel_ridge_rbf", "xgboost", "lightgbm",
    "catboost", "mlp",
}
SCALED_ALGORITHMS = {"ridge", "elasticnet", "knn", "svr_rbf", "kernel_ridge_rbf", "mlp"}


def default_parameters(algorithm: str, small_budget: bool = True) -> dict:
    trees = 80 if small_budget else 300
    return {
        "dummy_mean": {},
        "ridge": {"alpha": 10.0},
        "elasticnet": {"alpha": 0.01, "l1_ratio": 0.25, "max_iter": 5000},
        "knn": {"n_neighbors": 7, "weights": "distance", "p": 2},
        "random_forest": {"n_estimators": trees, "max_features": 0.7, "min_samples_leaf": 2},
        "extra_trees": {"n_estimators": trees, "max_features": 0.7, "min_samples_leaf": 2},
        "hist_gradient_boosting": {"max_iter": 100 if small_budget else 250, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        "svr_rbf": {"C": 10.0, "gamma": "scale", "epsilon": 0.1, "cache_size": 2048},
        "kernel_ridge_rbf": {"alpha": 1.0, "kernel": "rbf", "gamma": 0.01},
        "xgboost": {"n_estimators": trees, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        "lightgbm": {"n_estimators": trees, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        "catboost": {"iterations": trees, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        "mlp": {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3,
                "max_iter": 150 if small_budget else 400, "early_stopping": True},
    }[algorithm]


def parameter_candidates(algorithm: str, maximum: int, seed: int,
                         profile: str = "stage_a") -> list[dict]:
    """Return a deterministic bounded subset; never an uncontrolled grid."""
    if maximum < 1:
        raise ValueError("maximum parameter candidates must be positive")
    if profile == "linear_stabilization":
        stabilization = {
            # alpha=1000 is the strongest pre-registered Ridge candidate and
            # was not included in the two-candidate first batch.
            "ridge": [{"alpha": 1000.0}],
            # Two meaningfully separated members of the frozen ElasticNet
            # space provide shrinkage plus sparsity without a post-hoc grid.
            "elasticnet": [
                {"alpha": 0.01, "l1_ratio": 0.25, "max_iter": 5000},
                {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000},
            ],
        }
        if algorithm not in stabilization:
            raise ValueError(f"linear_stabilization profile does not support {algorithm}")
        return stabilization[algorithm][:maximum]
    if profile != "stage_a":
        raise ValueError(f"Unknown candidate profile: {profile}")
    spaces = {
        "dummy_mean": [{}],
        "ridge": [{"alpha": value} for value in [0.1, 10.0, 1000.0]],
        "elasticnet": [
            {"alpha": 0.001, "l1_ratio": 0.1, "max_iter": 5000},
            {"alpha": 0.01, "l1_ratio": 0.25, "max_iter": 5000},
            {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000},
            {"alpha": 0.01, "l1_ratio": 0.75, "max_iter": 5000},
        ],
        "knn": [{"n_neighbors": n, "weights": weight, "p": 2} for n, weight in [(5, "distance"), (11, "distance"), (21, "uniform")]],
        "random_forest": [
            {"n_estimators": 120, "max_features": 0.5, "min_samples_leaf": 1},
            {"n_estimators": 120, "max_features": 0.7, "min_samples_leaf": 3},
            {"n_estimators": 180, "max_features": 1.0, "min_samples_leaf": 2},
            {"n_estimators": 180, "max_features": 0.3, "min_samples_leaf": 5},
        ],
        "extra_trees": [
            {"n_estimators": 120, "max_features": 0.5, "min_samples_leaf": 1},
            {"n_estimators": 120, "max_features": 0.7, "min_samples_leaf": 3},
            {"n_estimators": 180, "max_features": 1.0, "min_samples_leaf": 2},
            {"n_estimators": 180, "max_features": 0.3, "min_samples_leaf": 5},
        ],
        "hist_gradient_boosting": [
            {"max_iter": 120, "max_leaf_nodes": 15, "learning_rate": 0.05, "l2_regularization": 1.0},
            {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 1.0},
            {"max_iter": 180, "max_leaf_nodes": 31, "learning_rate": 0.03, "l2_regularization": 5.0},
            {"max_iter": 120, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 5.0},
        ],
        "svr_rbf": [
            {"C": c, "gamma": gamma, "epsilon": 0.1, "cache_size": 2048}
            for c, gamma in [(1.0, "scale"), (10.0, "scale"), (10.0, 0.01), (100.0, 0.001)]
        ],
        "kernel_ridge_rbf": [{"alpha": a, "kernel": "rbf", "gamma": g} for a, g in [(0.1, 0.001), (1.0, 0.01), (10.0, 0.01), (1.0, 0.1)]],
        "xgboost": [
            {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 180, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 120, "max_depth": 7, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.7},
            {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
        ],
        "lightgbm": [
            {"n_estimators": 120, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
            {"n_estimators": 180, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 20},
            {"n_estimators": 120, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50},
            {"n_estimators": 180, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50},
        ],
        "catboost": [
            {"iterations": 120, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
            {"iterations": 180, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
            {"iterations": 120, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
            {"iterations": 180, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
        ],
        "mlp": [
            {"hidden_layer_sizes": (128,), "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 200, "early_stopping": True},
            {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 3e-4, "max_iter": 250, "early_stopping": True},
            {"hidden_layer_sizes": (256, 128), "alpha": 1e-3, "learning_rate_init": 3e-4, "max_iter": 250, "early_stopping": True},
        ],
    }
    options = spaces[algorithm]
    if len(options) <= maximum:
        return options
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(options), size=maximum, replace=False).tolist())
    return [options[index] for index in chosen]


def make_estimator(algorithm: str, seed: int, threads: int, small_budget: bool = True,
                   parameters: dict | None = None):
    if algorithm not in IMPLEMENTED_ALGORITHMS:
        raise ValueError(f"Algorithm adapter is not implemented: {algorithm}")
    params = dict(default_parameters(algorithm, small_budget) if parameters is None else parameters)
    if algorithm == "dummy_mean":
        model = DummyRegressor(strategy="mean", **params)
    elif algorithm == "ridge":
        model = Ridge(solver="lsqr", tol=1e-6, **params)
    elif algorithm == "elasticnet":
        model = ElasticNet(random_state=seed, **params)
    elif algorithm == "knn":
        model = KNeighborsRegressor(n_jobs=threads, **params)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(random_state=seed, n_jobs=threads, **params)
    elif algorithm == "extra_trees":
        model = ExtraTreesRegressor(random_state=seed, n_jobs=threads, **params)
    elif algorithm == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=seed, **params)
    elif algorithm == "svr_rbf":
        model = SVR(kernel="rbf", **params)
    elif algorithm == "kernel_ridge_rbf":
        model = KernelRidge(**params)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                             n_jobs=threads, **params)
    elif algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=seed, n_jobs=threads, verbosity=-1, **params)
    elif algorithm == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_seed=seed, thread_count=threads, verbose=False,
                                  allow_writing_files=False, **params)
    else:
        model = MLPRegressor(random_state=seed, **params)
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if algorithm in SCALED_ALGORITHMS:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def feature_view(cache: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "ecfp4":
        return cache["ecfp4"]
    if name == "rdkit2d":
        return cache["rdkit2d"]
    if name == "ecfp4_rdkit2d":
        return np.concatenate([cache["ecfp4"], cache["rdkit2d"]], axis=1)
    raise ValueError(f"Unknown feature view: {name}")
