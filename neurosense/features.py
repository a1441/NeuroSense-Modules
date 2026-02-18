"""MiniRocket feature extraction for NeuroSense sub-epochs."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import NeuroSenseConfig
from .preprocessing import PreprocessedData


def _get_minirocket():
    """Import MiniRocketMultivariate, raising a clear error if unavailable."""
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except ImportError:
        raise ImportError(
            "sktime is required for MiniRocket features. "
            "Install with: pip install sktime[all_extras]"
        )


def extract_minirocket(
    sub_epochs: np.ndarray,
    cfg: NeuroSenseConfig,
    num_kernels: int = 10_000,
    max_dilations_per_kernel: int = 32,
    fit_transform: bool = True,
    fitted_transform=None,
) -> Tuple[np.ndarray, object]:
    """
    Apply MiniRocketMultivariate to a set of sub-epochs.

    Parameters
    ----------
    sub_epochs : (n_instances, n_channels, n_timepoints)  — sktime panel format
    cfg        : pipeline config
    num_kernels, max_dilations_per_kernel : MiniRocket hyper-parameters
    fit_transform : if True, fit on sub_epochs then transform; else use fitted_transform
    fitted_transform : pre-fitted transformer (for test-fold transform)

    Returns
    -------
    features     : (n_instances, n_features)
    transformer  : fitted MiniRocketMultivariate instance
    """
    MiniRocketMultivariate = _get_minirocket()

    # sktime expects a 3-D array (n_instances, n_columns, n_timepoints)
    if sub_epochs.ndim != 3:
        raise ValueError(f"sub_epochs must be 3-D, got shape {sub_epochs.shape}")

    if fit_transform:
        transformer = MiniRocketMultivariate(
            num_kernels=num_kernels,
            max_dilations_per_kernel=max_dilations_per_kernel,
            random_state=cfg.random_seed,
        )
        features = transformer.fit_transform(sub_epochs)
    else:
        if fitted_transform is None:
            raise ValueError("fitted_transform must be provided when fit_transform=False")
        transformer = fitted_transform
        features = transformer.transform(sub_epochs)

    return np.asarray(features), transformer


def _gather_quadrant_sub_epochs(
    preprocessed: PreprocessedData,
    roi_window: Tuple[float, float],
    quadrant: str,
    cfg: NeuroSenseConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect stimulus sub-epochs that fall within roi_window for a given quadrant,
    returning (X, y) where y is binary (1=target quadrant, 0=other).
    """
    ratings = preprocessed.ratings
    n_trials = preprocessed.n_trials
    n_sub = preprocessed.stimulus_sub_epochs.shape[0] // n_trials

    win_start_i = int(roi_window[0] / cfg.epoch_duration)
    win_end_i = int(roi_window[1] / cfg.epoch_duration)

    target_mask = (ratings["quadrant"].values[:n_trials] == quadrant)

    X_list, y_list = [], []
    for t in range(n_trials):
        block = preprocessed.stimulus_sub_epochs[t * n_sub: (t + 1) * n_sub]
        window_block = block[win_start_i: win_end_i]
        if len(window_block) == 0:
            continue
        for epoch in window_block:
            X_list.append(epoch)
            y_list.append(1 if target_mask[t] else 0)

    if not X_list:
        return np.empty((0, preprocessed.stimulus_sub_epochs.shape[1], 1)), np.array([])

    return np.stack(X_list), np.array(y_list)


def build_feature_dataset(
    preprocessed_list: List[PreprocessedData],
    roi_maps: List[Dict[str, Tuple[Tuple[float, float], object]]],
    cfg: NeuroSenseConfig,
    num_kernels: int = 10_000,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    For each quadrant, gather ROI sub-epochs from all subjects,
    extract MiniRocket features, and return labelled datasets.

    Returns
    -------
    dict: quadrant → (X, y, subject_ids)
        X : (n_total_instances, n_features)
        y : (n_total_instances,)  binary labels
        subject_ids : list of length n_total_instances (for LOSO splitting)
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

    for quadrant in cfg.quadrant_names:
        all_epochs, all_y, all_sids = [], [], []

        for preprocessed, roi_map in zip(preprocessed_list, roi_maps):
            window, _ = roi_map[quadrant]
            X_raw, y = _gather_quadrant_sub_epochs(preprocessed, window, quadrant, cfg)
            if len(X_raw) == 0:
                continue
            all_epochs.append(X_raw)
            all_y.append(y)
            all_sids.extend([preprocessed.subject_id] * len(X_raw))

        if not all_epochs:
            warnings.warn(f"No data collected for quadrant {quadrant}")
            continue

        X_raw_all = np.concatenate(all_epochs, axis=0)
        y_all = np.concatenate(all_y, axis=0)

        # Fit MiniRocket once on all data (LOSO will refit per fold in modeling.py)
        X_feat, _ = extract_minirocket(X_raw_all, cfg, num_kernels=num_kernels)
        result[quadrant] = (X_feat, y_all, all_sids)

    return result
