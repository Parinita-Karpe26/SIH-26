"""
train_model.py
==============================================================================
AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (Day 1-10)
Model training pipeline -- NCMRWF prototype

Trains a gradient-boosted tree classifier (XGBoost by default, LightGBM as an
alternative) that predicts ``bust_probability`` -- the probability that a
given grid point / lead time forecast will be a severe "bust" -- from the
engineered features produced by ``generate_data_and_features_real.py``
(real NCMRWF IMDAA reanalysis; see that file for how forecasts/busts are
defined).

------------------------------------------------------------------------------
WHY A CHRONOLOGICAL SPLIT ON VALID DATE (not a random row split)
------------------------------------------------------------------------------
Every grid point from the same day shares the same weather system, so a
random row split would let the model memorise that day's event. The split
is therefore by whole DAYS, ordered chronologically, with the latest days
held out.

The grouping key defaults to ``valid_time`` (the day being verified), not
``init_time``. With real data, forecasts issued on different days verify
against the SAME observation: e.g. a Day-3 forecast issued 07-07 and a
Day-1 forecast issued 07-09 are both scored against 07-10. Splitting on
``init_time`` would put the 07-10 observation into BOTH a training label and
a test label, and because lat/lon are features the model could memorise
where 07-10's rain fell. Splitting on ``valid_time`` guarantees every test
label is an observation the model never saw during training, while the test
set still covers every lead day. (``--split-key init_time`` restores the old
behaviour for the synthetic dataset.)

  1. Split by whole verification day, never by row.
  2. Hold out the most recent days for validation / test.
  3. Optionally (``--run-cv``) GroupKFold CV grouped by the same key.
  4. Report naive BASELINES on the same test set so the model's numbers can
     be read against "compared to what?".

------------------------------------------------------------------------------
ARTIFACT SCHEMA (models/bust_detector.pkl)
------------------------------------------------------------------------------
A single pickled dict, fully self-contained (no dependency on this file to
reload and use it -- see ``explainability.py``):
    {
      "model": <fitted xgboost.XGBClassifier or lightgbm.LGBMClassifier>,
      "model_type": "xgboost" | "lightgbm",
      "feature_cols": [str, ...]                # exact column order used for X
      "categorical_cols": [str, ...]             # subset of feature_cols
      "category_levels": {col: [category, ...]}  # training-time vocabulary
      "numeric_fill_values": {col: float}        # train medians, for missing values
      "categorical_fill_values": {col: str}      # train mode, for missing values
      "target_col": "is_bust",
      "operating_threshold": float,              # best-F1 threshold (chosen on val)
      "training_metadata": {...},
      "metrics": {"validation": {...}, "test": {...}},
    }

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python train_model.py
    python train_model.py --model-type lightgbm --run-cv
    python train_model.py --data data/processed_features_real.parquet \
        --output-model models/bust_detector.pkl

Dependencies: numpy, pandas, scikit-learn, xgboost and/or lightgbm.
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bust_detection.train")
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

TARGET_COL = "is_bust"
CATEGORICAL_COLS = ["regime", "region", "season", "lead_day_bucket"]
# Columns requiring ground truth (or raw timestamps not needed once lead_day /
# season / regime capture the temporal signal) -- excluded from model inputs.
NON_FEATURE_COLS = [
    "init_time", "valid_time",
    "Obs_T2m", "Obs_Rain", "temp_error", "rain_error", "error_magnitude", TARGET_COL,
]

# ===========================================================================
# 1. DATA LOADING / SPLITTING
# ===========================================================================


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {TARGET_COL, "init_time", "lead_day", "lat", "lon", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    return df


def get_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    # Any Obs_* column is ground truth on the valid day (e.g. Obs_T850 in the
    # S2S dataset) -- never a predictor, whatever the dataset.
    exclude = set(exclude or []) | {c for c in df.columns if c.startswith("Obs_")}
    unknown = exclude - set(df.columns)
    if unknown:
        logger.warning("--exclude-features names not in dataset (ignored): %s", sorted(unknown))
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in exclude]


def resolve_split_key(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    logger.warning("Split key '%s' not in dataset -- falling back to 'init_time'.", requested)
    return "init_time"


def year_split(
    df: pd.DataFrame, test_years: List[int], val_years: Optional[List[int]] = None,
    val_frac: float = 0.15, key: str = "valid_time",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out whole YEARS (by the year of ``key``): e.g. train 2011-2013,
    validate 2014, test 2015. Without --val-years, validation is the latest
    ``val_frac`` of the remaining days."""
    years = pd.to_datetime(df[key]).dt.year
    test_mask = years.isin(test_years)
    if not test_mask.any():
        raise ValueError(f"No rows in --test-years {test_years}; dataset years: {sorted(years.unique())}")
    if val_years:
        val_mask = years.isin(val_years)
        if not val_mask.any():
            raise ValueError(f"No rows in --val-years {val_years}; dataset years: {sorted(years.unique())}")
        rest = ~test_mask & ~val_mask
        return df[rest].reset_index(drop=True), df[val_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    rest = df[~test_mask]
    days = sorted(pd.to_datetime(rest[key]).unique())
    n_val = max(1, int(round(len(days) * val_frac)))
    if len(days) - n_val < 1:
        raise ValueError("Not enough days outside the test years to form train + validation.")
    val_days = set(days[-n_val:])
    val_mask = rest[key].isin(val_days)
    return rest[~val_mask].reset_index(drop=True), rest[val_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def chronological_case_split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, key: str = "valid_time"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by whole day (``key``, default ``valid_time``), ordered
    chronologically, so validation/test hold out entire, more-recent days
    rather than individual grid points from days the model has already seen."""
    unique_cases = sorted(pd.to_datetime(df[key]).unique())
    n = len(unique_cases)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough distinct '{key}' days ({n}) to carve out "
            f"train/val/test with val_frac={val_frac}, test_frac={test_frac}."
        )
    train_cases = set(unique_cases[:n_train])
    val_cases = set(unique_cases[n_train : n_train + n_val])
    test_cases = set(unique_cases[n_train + n_val :])

    train_df = df[df[key].isin(train_cases)].reset_index(drop=True)
    val_df = df[df[key].isin(val_cases)].reset_index(drop=True)
    test_df = df[df[key].isin(test_cases)].reset_index(drop=True)
    return train_df, val_df, test_df


# ===========================================================================
# 2. CATEGORICAL ENCODING (fixed vocabulary, stored in the artifact)
# ===========================================================================


def fit_categorical_encoding(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, List[str]]:
    """Build a fixed category->integer-code vocabulary from the FULL dataset
    (a structural property of the domain, not a statistic of the labels) so
    train/val/test/CV folds all encode categories identically regardless of
    which forecast cases happen to land in which split."""
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in categorical_cols}


def encode_categoricals(
    df: pd.DataFrame, feature_cols: List[str], categorical_cols: List[str], category_levels: Dict[str, List[str]]
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in categorical_cols:
        mapping = {cat: i for i, cat in enumerate(category_levels[c])}
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    return X


def compute_fill_values(
    train_df: pd.DataFrame, feature_cols: List[str], categorical_cols: List[str]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Training-set medians/modes, used by ``explainability.py`` to impute
    any features a caller doesn't supply for a single grid-point prediction."""
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    numeric_fill = train_df[numeric_cols].median(numeric_only=True).to_dict()
    categorical_fill = {c: str(train_df[c].astype(str).mode().iloc[0]) for c in categorical_cols}
    return numeric_fill, categorical_fill


# ===========================================================================
# 3. MODEL CONSTRUCTION / TRAINING
# ===========================================================================


def build_model(
    model_type: str,
    scale_pos_weight: float,
    seed: int,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 30,
):
    """Return an unfitted, imbalance-aware classifier. ``scale_pos_weight``
    (n_negative / n_positive on the TRAINING set) is the standard way both
    libraries down-weight the majority ("reliable") class so the rare bust
    class isn't ignored -- essential here since busts are typically <10%."""
    if model_type == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",  # PR-AUC-based early stopping, appropriate for imbalance
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        if not _HAS_LGB:
            raise ImportError("lightgbm is not installed. Install with: pip install lightgbm")
        depth = max_depth if max_depth and max_depth > 0 else -1
        num_leaves = min(2 ** depth - 1, 127) if depth > 0 else 63
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'xgboost' or 'lightgbm'.")


def train_with_early_stopping(
    model, model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray, early_stopping_rounds: int = 30,
):
    if model_type == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "lightgbm":
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)],
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")
    return model


def predict_proba_positive(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def get_best_iteration(model, model_type: str) -> Optional[int]:
    try:
        return int(model.best_iteration) if model_type == "xgboost" else int(model.best_iteration_)
    except Exception:
        return None


# ===========================================================================
# 4. METRICS
# ===========================================================================


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Search the precision-recall curve (computed on VALIDATION, never test)
    for the probability threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    f1s = f1s[:-1]  # precision/recall have one more point than thresholds
    best_idx = int(np.nanargmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold_used": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_positive": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def print_metrics_report(name: str, metrics: dict) -> None:
    print("-" * 74)
    print(name)
    print(f"  Brier Score        : {metrics['brier_score']:.4f}   (lower is better; 0 = perfect)")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.4f}   (more informative than ROC-AUC under class imbalance)")
    print(f"  F1-Score           : {metrics['f1_at_threshold']:.4f}   @ threshold={metrics['threshold_used']:.3f}")
    print(f"  Precision / Recall : {metrics['precision_at_threshold']:.4f} / {metrics['recall_at_threshold']:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"  n_bust / n_total   : {metrics['n_positive']:,} / {metrics['n_total']:,}")


# ===========================================================================
# 4b. NAIVE BASELINES ("compared to what?")
# ===========================================================================


def compute_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, dict]:
    """Score simple, no-ML bust predictors on the SAME held-out test rows.
    Each only uses information available from the training period or at
    issue time, so they are fair comparisons for the model."""
    y = test_df[TARGET_COL].values
    base_rate = float(train_df[TARGET_COL].mean())
    out: Dict[str, dict] = {}

    out["constant_train_base_rate"] = compute_classification_metrics(y, np.full(len(y), base_rate), threshold=0.5)

    # Historical bust rate for each (region, lead_day) in the training period
    lookup = train_df.groupby(["region", "lead_day"], observed=True)[TARGET_COL].mean()
    keys = pd.MultiIndex.from_arrays([test_df["region"].astype(str), test_df["lead_day"]])
    lookup.index = pd.MultiIndex.from_arrays([lookup.index.get_level_values(0).astype(str), lookup.index.get_level_values(1)])
    prob = lookup.reindex(keys).fillna(base_rate).values
    out["region_x_leadday_train_rate"] = compute_classification_metrics(y, prob, threshold=base_rate)

    # Per-grid-cell past-only verification history, if the dataset has it
    if "clim_hist_bust_rate" in test_df.columns:
        prob = test_df["clim_hist_bust_rate"].astype(float).fillna(base_rate).values
        out["past_only_cell_bust_rate"] = compute_classification_metrics(y, prob, threshold=base_rate)
    return out


def print_baselines(baselines: Dict[str, dict], model_metrics: dict) -> None:
    print("-" * 74)
    print("BASELINE COMPARISON on the same held-out test rows")
    print(f"  {'predictor':34s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'Brier':>8s}")
    for name, m in baselines.items():
        print(f"  {name:34s} {m['roc_auc']:8.4f} {m['pr_auc']:8.4f} {m['brier_score']:8.4f}")
    print(f"  {'>> trained model':34s} {model_metrics['roc_auc']:8.4f} {model_metrics['pr_auc']:8.4f} {model_metrics['brier_score']:8.4f}")


# ===========================================================================
# 5. REGION-WISE FORECAST CONFIDENCE INDEX
# ===========================================================================


def compute_confidence_index(df_slice: pd.DataFrame, bust_probability: np.ndarray) -> pd.DataFrame:
    """Confidence = (1 - bust_probability) * 100%, attached to each row's
    location / lead time / regime context for downstream reporting."""
    keep_cols = [c for c in ["init_time", "valid_time", "lead_day", "lat", "lon", "region", "regime", "season"] if c in df_slice.columns]
    out = df_slice[keep_cols].copy().reset_index(drop=True)
    out["bust_probability"] = bust_probability
    out["confidence_pct"] = (1.0 - bust_probability) * 100.0
    return out


def summarize_confidence(conf_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_region = (
        conf_df.groupby("region", observed=True)["confidence_pct"]
        .agg(["mean", "min", "count"])
        .rename(columns={"mean": "mean_confidence_pct", "min": "min_confidence_pct", "count": "n_points"})
        .sort_values("mean_confidence_pct")
    )
    by_region_leadday = conf_df.pivot_table(
        index="region", columns="lead_day", values="confidence_pct", aggfunc="mean", observed=True
    )
    by_lead_day = conf_df.groupby("lead_day", observed=True)["confidence_pct"].mean()
    return {"by_region": by_region, "by_region_leadday": by_region_leadday, "by_lead_day": by_lead_day}


# ===========================================================================
# 6. OPTIONAL: GROUPED CROSS-VALIDATION ROBUSTNESS CHECK
# ===========================================================================


def run_group_kfold_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    category_levels: Dict[str, List[str]],
    model_type: str,
    n_splits: int,
    seed: int,
    key: str = "valid_time",
    no_class_weight: bool = False,
) -> pd.DataFrame:
    """Cross-validate across several different held-out groups of days
    (never held-out rows) -- a robustness check on top of the single
    chronological split. NOTE: unlike the main split, folds are not in time
    order, so later days can train a model tested on earlier ones; treat CV
    as a stability check, not the headline number."""
    n_groups = df[key].nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"Need >= 2 distinct '{key}' days for CV; found {n_groups}.")
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[key].values

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups)):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        X_tr = encode_categoricals(tr_df, feature_cols, categorical_cols, category_levels)
        X_te = encode_categoricals(te_df, feature_cols, categorical_cols, category_levels)
        y_tr, y_te = tr_df[TARGET_COL].values, te_df[TARGET_COL].values
        spw = 1.0 if no_class_weight else (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)

        model = build_model(model_type, spw, seed, n_estimators=250)  # lighter for CV speed
        if model_type == "xgboost":
            model.set_params(early_stopping_rounds=None)
            model.fit(X_tr, y_tr, verbose=False)
        else:
            model.fit(X_tr, y_tr)

        prob = predict_proba_positive(model, X_te)
        m = compute_classification_metrics(y_te, prob, threshold=0.5)
        m["fold"] = fold + 1
        m["n_test_cases"] = int(te_df[key].nunique())
        rows.append(m)

    cv_df = pd.DataFrame(rows)[["fold", "n_test_cases", "roc_auc", "pr_auc", "brier_score", "f1_at_threshold"]]
    return cv_df


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    category_levels: Dict[str, List[str]],
    args: argparse.Namespace,
    key: str,
    n_folds: int,
    window_days: int,
) -> pd.DataFrame:
    """Rolling-origin ("walk-forward") evaluation: the latest ``window_days``
    days are the test window, the ``window_days`` before them validation, and
    everything earlier training; then the whole thing steps back one window
    and repeats. Every fold only ever trains on days BEFORE its test days, so
    this is the operationally honest estimate -- and averaging over several
    windows stops one unusual week from deciding the score."""
    days = sorted(pd.to_datetime(df[key]).unique())
    rows = []
    for fold in range(n_folds):
        end = len(days) - fold * window_days
        test_days, val_days = days[end - window_days:end], days[end - 2 * window_days:end - window_days]
        train_days = days[:end - 2 * window_days]
        if len(train_days) < window_days or len(val_days) == 0:
            logger.warning("Walk-forward: stopping after %d fold(s) -- not enough earlier days for fold %d.", fold, fold + 1)
            break
        tr = df[df[key].isin(train_days)]
        va = df[df[key].isin(val_days)]
        te = df[df[key].isin(test_days)]
        if va[TARGET_COL].nunique() < 2 or te[TARGET_COL].nunique() < 2:
            logger.warning("Walk-forward fold %d skipped: a split has only one class.", fold + 1)
            continue
        X_tr = encode_categoricals(tr, feature_cols, categorical_cols, category_levels)
        X_va = encode_categoricals(va, feature_cols, categorical_cols, category_levels)
        X_te = encode_categoricals(te, feature_cols, categorical_cols, category_levels)
        y_tr, y_va, y_te = tr[TARGET_COL].values, va[TARGET_COL].values, te[TARGET_COL].values
        spw = 1.0 if args.no_class_weight else (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)
        model = build_model(args.model_type, spw, args.seed, n_estimators=args.n_estimators, max_depth=args.max_depth,
                            learning_rate=args.learning_rate, early_stopping_rounds=args.early_stopping_rounds)
        model = train_with_early_stopping(model, args.model_type, X_tr, y_tr, X_va, y_va, args.early_stopping_rounds)
        thr, _ = find_best_f1_threshold(y_va, predict_proba_positive(model, X_va))
        m = compute_classification_metrics(y_te, predict_proba_positive(model, X_te), threshold=thr)
        base = compute_baselines(tr, te)
        best_base = max(base.values(), key=lambda b: b["pr_auc"])
        rows.append({
            "fold": fold + 1,
            "test_days": f"{pd.Timestamp(test_days[0]).date()}..{pd.Timestamp(test_days[-1]).date()}",
            "bust_rate": float(y_te.mean()),
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier_score": m["brier_score"],
            "best_baseline_pr_auc": best_base["pr_auc"],
            "pr_auc_lift_vs_baseline": m["pr_auc"] / max(best_base["pr_auc"], 1e-9),
        })
        logger.info("Walk-forward fold %d (%s): ROC %.3f | PR %.3f (best baseline %.3f) | Brier %.4f",
                    fold + 1, rows[-1]["test_days"], m["roc_auc"], m["pr_auc"], best_base["pr_auc"], m["brier_score"])
    return pd.DataFrame(rows)


# ===========================================================================
# 7. ARTIFACT PERSISTENCE
# ===========================================================================


def save_artifact(artifact: dict, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model artifact saved to %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


# ===========================================================================
# 8. ORCHESTRATION
# ===========================================================================


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the NCMRWF forecast-bust-detection classifier.")
    p.add_argument("--data", type=str, default="data/processed_features_real.parquet")
    p.add_argument("--output-model", type=str, default="models/bust_detector.pkl")
    p.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm"])
    p.add_argument("--split-key", type=str, default="valid_time", choices=["valid_time", "init_time"],
                   help="Day column the chronological split is grouped on (see module docstring).")
    p.add_argument("--test-years", type=int, nargs="*", default=[],
                   help="Hold out these whole years as the test set (e.g. 2015). Recommended for multi-year S2S data.")
    p.add_argument("--val-years", type=int, nargs="*", default=[],
                   help="Whole years for validation (e.g. 2014). Needs --test-years.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument("--exclude-features", nargs="*", default=[],
                   help="Extra columns to leave out of the model, e.g. --exclude-features lat lon "
                        "to test whether skill survives without memorising location.")
    p.add_argument("--no-class-weight", action="store_true",
                   help="Train without up-weighting busts. Probabilities then stay close to real bust "
                        "frequencies (much better Brier / confidence %%); rare-event ranking is barely affected "
                        "because the operating threshold is still tuned on validation.")
    p.add_argument("--walk-forward", type=int, default=0, metavar="N",
                   help="Also run N-fold walk-forward (rolling-origin) evaluation -- the most honest score. Try 3.")
    p.add_argument("--run-cv", action="store_true", help="Also run GroupKFold CV across train+val as a robustness check")
    p.add_argument("--cv-folds", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()

    logger.info("Loading dataset from %s ...", args.data)
    df = load_data(args.data)
    feature_cols = get_feature_columns(df, args.exclude_features)
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    logger.info("Using %d predictor features (target=%s excluded, along with Obs_*/*_error diagnostics).", len(feature_cols), TARGET_COL)

    category_levels = fit_categorical_encoding(df, categorical_cols)

    key = resolve_split_key(df, args.split_key)
    if args.test_years:
        train_df, val_df, test_df = year_split(df, args.test_years, args.val_years, args.val_frac, key=key)
        logger.info("Year-based split: test years %s, validation %s", args.test_years, args.val_years or f"last {args.val_frac:.0%} of remaining days")
    else:
        if args.val_years:
            raise ValueError("--val-years needs --test-years as well.")
        train_df, val_df, test_df = chronological_case_split(df, args.val_frac, args.test_frac, key=key)
    for name, part, note in [("train", train_df, ""), ("val", val_df, ""), ("test", test_df, "  [held-out FUTURE days]")]:
        logger.info(
            "Chronological split on %s -> %-5s: %d day(s) / %s rows (%s -> %s), lead days %s, bust rate %.2f%%%s",
            key, name, part[key].nunique(), f"{len(part):,}", part[key].min(), part[key].max(),
            sorted(part["lead_day"].unique().tolist()), 100 * part[TARGET_COL].mean(), note,
        )
    for name, part in [("validation", val_df), ("test", test_df)]:
        if part[TARGET_COL].nunique() < 2:
            raise ValueError(f"The {name} split has only one class -- add more days of data or change --val-frac/--test-frac.")

    numeric_fill, categorical_fill = compute_fill_values(train_df, feature_cols, categorical_cols)

    X_train = encode_categoricals(train_df, feature_cols, categorical_cols, category_levels)
    X_val = encode_categoricals(val_df, feature_cols, categorical_cols, category_levels)
    X_test = encode_categoricals(test_df, feature_cols, categorical_cols, category_levels)
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    n_pos, n_neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    scale_pos_weight = 1.0 if args.no_class_weight else n_neg / max(n_pos, 1)
    logger.info(
        "Training class balance: %d busts / %d reliable (%.2f%% positive) -> scale_pos_weight=%.2f",
        n_pos, n_neg, 100 * n_pos / len(y_train), scale_pos_weight,
    )

    if args.run_cv:
        logger.info("Running %d-fold GroupKFold CV (grouped by forecast case) across train+val as a robustness check...", args.cv_folds)
        cv_df = run_group_kfold_cv(
            pd.concat([train_df, val_df], ignore_index=True), feature_cols, categorical_cols,
            category_levels, args.model_type, args.cv_folds, args.seed, key=key, no_class_weight=args.no_class_weight,
        )
        print(cv_df.to_string(index=False))
        print(f"CV mean PR-AUC={cv_df['pr_auc'].mean():.4f} (+/-{cv_df['pr_auc'].std():.4f}), "
              f"mean ROC-AUC={cv_df['roc_auc'].mean():.4f} (+/-{cv_df['roc_auc'].std():.4f})")

    wf_df = None
    if args.walk_forward:
        window = max(1, test_df[key].nunique())
        logger.info("Running %d-fold walk-forward evaluation (%d-day windows)...", args.walk_forward, window)
        wf_df = run_walk_forward(df, feature_cols, categorical_cols, category_levels, args, key, args.walk_forward, window)
        if len(wf_df):
            print("=" * 74)
            print("WALK-FORWARD EVALUATION (each fold trains only on earlier days)")
            print(wf_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"MEAN over {len(wf_df)} windows: ROC-AUC {wf_df['roc_auc'].mean():.3f} (+/-{wf_df['roc_auc'].std(ddof=0):.3f}) | "
                  f"PR-AUC {wf_df['pr_auc'].mean():.3f} (+/-{wf_df['pr_auc'].std(ddof=0):.3f}) | "
                  f"Brier {wf_df['brier_score'].mean():.4f} | PR lift vs best baseline x{wf_df['pr_auc_lift_vs_baseline'].mean():.2f}")
            print("=" * 74)

    logger.info("Training final %s classifier (early stopping on validation set)...", args.model_type)
    model = build_model(
        args.model_type, scale_pos_weight, args.seed,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, early_stopping_rounds=args.early_stopping_rounds,
    )
    model = train_with_early_stopping(model, args.model_type, X_train, y_train, X_val, y_val, args.early_stopping_rounds)
    best_iter = get_best_iteration(model, args.model_type)
    if best_iter is not None:
        logger.info("Early stopping selected iteration %d / %d.", best_iter, args.n_estimators)

    val_prob = predict_proba_positive(model, X_val)
    best_threshold, best_f1_val = find_best_f1_threshold(y_val, val_prob)
    logger.info("Best-F1 operating threshold chosen on VALIDATION set: %.3f (val F1=%.4f)", best_threshold, best_f1_val)

    val_metrics_default = compute_classification_metrics(y_val, val_prob, threshold=0.5)
    val_metrics_best = compute_classification_metrics(y_val, val_prob, threshold=best_threshold)

    test_prob = predict_proba_positive(model, X_test)
    test_metrics_default = compute_classification_metrics(y_test, test_prob, threshold=0.5)
    test_metrics_best = compute_classification_metrics(y_test, test_prob, threshold=best_threshold)

    print("=" * 74)
    print(f"MODEL EVALUATION -- {args.model_type.upper()}")
    print_metrics_report("VALIDATION (threshold=0.50)", val_metrics_default)
    print_metrics_report(f"VALIDATION (best-F1 threshold={best_threshold:.3f})", val_metrics_best)
    print_metrics_report("TEST / held-out FUTURE events (threshold=0.50)", test_metrics_default)
    print_metrics_report(f"TEST / held-out FUTURE events (best-F1 threshold={best_threshold:.3f})", test_metrics_best)
    baselines = compute_baselines(train_df, test_df)
    print_baselines(baselines, test_metrics_best)
    print("=" * 74)

    conf_df = compute_confidence_index(test_df, test_prob)
    summaries = summarize_confidence(conf_df)
    print("Region-wise Forecast Confidence Index on held-out test events (mean %, lower = higher bust risk):")
    print(summaries["by_region"].to_string(float_format=lambda x: f"{x:.2f}"))
    print("=" * 74)

    models_dir = Path(args.output_model).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    summaries["by_region"].to_csv(models_dir / "region_confidence_index.csv")
    summaries["by_region_leadday"].to_csv(models_dir / "region_leadday_confidence_index.csv")
    logger.info("Region confidence index CSVs written to %s", models_dir)

    artifact = {
        "model": model,
        "model_type": args.model_type,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "category_levels": category_levels,
        "numeric_fill_values": numeric_fill,
        "categorical_fill_values": categorical_fill,
        "target_col": TARGET_COL,
        "operating_threshold": best_threshold,
        "training_metadata": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": args.model_type,
            "data_path": args.data,
            "split_key": key,
            "test_years": args.test_years,
            "val_years": args.val_years,
            "n_train_cases": int(train_df[key].nunique()),
            "n_val_cases": int(val_df[key].nunique()),
            "n_test_cases": int(test_df[key].nunique()),
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "train_date_range": [str(train_df[key].min()), str(train_df[key].max())],
            "test_date_range": [str(test_df[key].min()), str(test_df[key].max())],
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": best_iter,
            "hyperparameters": {
                "n_estimators": args.n_estimators, "max_depth": args.max_depth,
                "learning_rate": args.learning_rate, "early_stopping_rounds": args.early_stopping_rounds,
            },
        },
        "metrics": {
            "validation": {"threshold_0.5": val_metrics_default, "best_f1_threshold": val_metrics_best},
            "test": {"threshold_0.5": test_metrics_default, "best_f1_threshold": test_metrics_best},
            "baselines_on_test": baselines,
            "walk_forward": None if wf_df is None else wf_df.to_dict(orient="records"),
        },
    }
    save_artifact(artifact, args.output_model)

    logger.info("Pipeline complete in %.1f s.", time.perf_counter() - t0)


if __name__ == "__main__":
    main()