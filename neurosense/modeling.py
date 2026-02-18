"""LOSO cross-validation with MiniRocket + SVM pipeline and random hyper-parameter search."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from scipy.stats import loguniform

from .config import NeuroSenseConfig


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    subject_id: str
    quadrant: str
    accuracy: float
    f1_macro: float
    confusion_matrix: np.ndarray
    decision_probabilities: np.ndarray   # predicted proba for test samples
    best_params: dict


@dataclass
class LOSOResults:
    folds: List[FoldResult] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return per-quadrant mean ± std of accuracy and F1."""
        records = []
        if not self.folds:
            return pd.DataFrame(columns=["quadrant", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "n_folds"])
        for quadrant in set(f.quadrant for f in self.folds):
            q_folds = [f for f in self.folds if f.quadrant == quadrant]
            accs = [f.accuracy for f in q_folds]
            f1s = [f.f1_macro for f in q_folds]
            records.append(
                {
                    "quadrant": quadrant,
                    "accuracy_mean": np.mean(accs),
                    "accuracy_std": np.std(accs),
                    "f1_mean": np.mean(f1s),
                    "f1_std": np.std(f1s),
                    "n_folds": len(q_folds),
                }
            )
        return pd.DataFrame(records).sort_values("quadrant").reset_index(drop=True)

    def per_subject_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "subject_id": f.subject_id,
                    "quadrant": f.quadrant,
                    "accuracy": f.accuracy,
                    "f1_macro": f.f1_macro,
                }
                for f in self.folds
            ]
        )

    def mean_decision_probability(self) -> Dict[str, float]:
        """Mean max predicted probability per subject (for flagging)."""
        per_subject: Dict[str, List[float]] = {}
        for f in self.folds:
            probs = f.decision_probabilities
            mean_p = float(np.max(probs, axis=1).mean()) if probs.ndim == 2 else float(probs.mean())
            per_subject.setdefault(f.subject_id, []).append(mean_p)
        return {sid: np.mean(vals) for sid, vals in per_subject.items()}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _make_scaler(name: str):
    return {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}[name]


def build_pipeline(params: Optional[dict] = None) -> Pipeline:
    """
    Build the MiniRocket → Scaler → SVM pipeline.
    params keys: scaler, svm_C
    MiniRocket fitting happens separately in the LOSO loop (raw epochs needed).
    Here the pipeline operates on pre-extracted features.
    """
    params = params or {}
    scaler_name = params.get("scaler", "standard")
    svm_C = params.get("svm_C", 1.0)

    return Pipeline(
        [
            ("scaler", _make_scaler(scaler_name)),
            ("svm", SVC(kernel="rbf", C=svm_C, probability=True)),
        ]
    )


def build_search_space(cfg: NeuroSenseConfig) -> dict:
    return {
        "scaler": ["minmax", "standard", "robust"],
        "svm_C": loguniform(1e-2, 1e3),
    }


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------

def _subject_split(subject_ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Return list of (train_idx, test_idx, test_subject_id) for LOSO."""
    unique = list(dict.fromkeys(subject_ids))  # preserve order, dedupe
    sid_arr = np.array(subject_ids)
    splits = []
    for sid in unique:
        test_idx = np.where(sid_arr == sid)[0]
        train_idx = np.where(sid_arr != sid)[0]
        if len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx, sid))
    return splits


def loso_evaluate(
    feature_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    cfg: NeuroSenseConfig,
) -> LOSOResults:
    """
    Leave-One-Subject-Out evaluation for each quadrant.

    feature_datasets: quadrant → (X, y, subject_ids)
    """
    results = LOSOResults()

    for quadrant, (X, y, subject_ids) in feature_datasets.items():
        if len(X) == 0:
            warnings.warn(f"No data for quadrant {quadrant}; skipping.")
            continue

        splits = _subject_split(subject_ids)
        if not splits:
            warnings.warn(f"Could not create LOSO splits for quadrant {quadrant}.")
            continue

        for train_idx, test_idx, test_sid in splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                warnings.warn(
                    f"Only one class in training set for {quadrant} / {test_sid}; skipping fold."
                )
                continue

            # Inner hyper-parameter search
            search_space = build_search_space(cfg)
            pipeline = build_pipeline()

            # Wrap pipeline for RandomizedSearchCV — pass params with prefixes
            param_dist = {
                "scaler": search_space["scaler"],
                "svm__C": search_space["svm_C"],
            }

            # Build a searchable pipeline
            inner_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", probability=True)),
                ]
            )
            param_dist_inner = {
                "scaler": [MinMaxScaler(), StandardScaler(), RobustScaler()],
                "svm__C": loguniform(1e-2, 1e3),
            }

            cv_folds = min(cfg.cv_inner, len(X_train))
            search = RandomizedSearchCV(
                inner_pipeline,
                param_dist_inner,
                n_iter=min(cfg.n_random_search_iter, 10),  # cap for speed
                cv=max(2, cv_folds),
                scoring="accuracy",
                random_state=cfg.random_seed,
                n_jobs=-1,
                refit=True,
            )
            try:
                search.fit(X_train, y_train)
                best_clf = search.best_estimator_
                best_params = search.best_params_
            except Exception as e:
                warnings.warn(f"RandomizedSearchCV failed for {quadrant}/{test_sid}: {e}. Using default params.")
                best_clf = build_pipeline()
                best_clf.fit(X_train, y_train)
                best_params = {}

            y_pred = best_clf.predict(X_test)
            try:
                proba = best_clf.predict_proba(X_test)
            except Exception:
                proba = np.column_stack([1 - (y_pred == 1).astype(float), (y_pred == 1).astype(float)])

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            results.folds.append(
                FoldResult(
                    subject_id=test_sid,
                    quadrant=quadrant,
                    accuracy=acc,
                    f1_macro=f1,
                    confusion_matrix=cm,
                    decision_probabilities=proba,
                    best_params=best_params,
                )
            )

    return results
