"""ROI time-sliding window selection per emotion quadrant using SlidingEstimator."""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import NeuroSenseConfig
from .preprocessing import PreprocessedData

# MNE decoding — import lazily to avoid heavy cost at module load
_mne_decoding_available = True
try:
    from mne.decoding import SlidingEstimator, cross_val_multiscore
except ImportError:
    _mne_decoding_available = False
    warnings.warn("mne.decoding not available; ROI sliding will use a simple fallback.")


def _window_sub_epochs(
    stimulus_sub_epochs: np.ndarray,
    n_trials: int,
    window: Tuple[float, float],
    epoch_duration: float,
    sfreq: float,
) -> np.ndarray:
    """
    Extract sub-epochs that fall within [window[0], window[1]] seconds from stimulus onset.

    stimulus_sub_epochs: (n_trials * n_sub_per_trial, n_ch, n_times)
    Returns: (n_selected_sub_epochs, n_ch, n_times)
    """
    n_sub_per_trial = stimulus_sub_epochs.shape[0] // n_trials
    win_start_idx = int(window[0] / epoch_duration)
    win_end_idx = int(window[1] / epoch_duration)

    selected = []
    for t in range(n_trials):
        base = t * n_sub_per_trial
        for w in range(win_start_idx, min(win_end_idx, n_sub_per_trial)):
            selected.append(stimulus_sub_epochs[base + w])

    if not selected:
        return np.empty((0, stimulus_sub_epochs.shape[1], stimulus_sub_epochs.shape[2]))
    return np.stack(selected)


def select_roi_window(
    preprocessed: PreprocessedData,
    quadrant: str,
    cfg: NeuroSenseConfig,
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    For each candidate ROI window, build a binary classification dataset
    (target quadrant sub-epochs vs baseline sub-epochs), run SlidingEstimator,
    and pick the window with the highest mean accuracy above the 95th-percentile threshold.

    Returns
    -------
    best_window : (start, end) tuple in seconds
    accuracy_array : 1-D accuracy scores for the best window
    """
    ratings = preprocessed.ratings
    if "quadrant" not in ratings.columns:
        raise ValueError("ratings DataFrame must have a 'quadrant' column")

    target_trial_mask = (ratings["quadrant"] == quadrant).values
    n_trials = preprocessed.n_trials

    # Trim mask to actual number of trials extracted
    if len(target_trial_mask) > n_trials:
        target_trial_mask = target_trial_mask[:n_trials]
    elif len(target_trial_mask) < n_trials:
        # Pad with False
        pad = np.zeros(n_trials - len(target_trial_mask), dtype=bool)
        target_trial_mask = np.concatenate([target_trial_mask, pad])

    target_idx = np.where(target_trial_mask)[0]
    baseline_idx = np.where(~target_trial_mask)[0]

    if len(target_idx) < 2:
        warnings.warn(f"Quadrant {quadrant}: fewer than 2 target trials — using first window.")
        return cfg.roi_windows[0], np.array([0.5])

    n_sub = preprocessed.stimulus_sub_epochs.shape[0] // n_trials
    best_window = cfg.roi_windows[0]
    best_score = -np.inf
    best_acc = np.array([0.5])

    for window in cfg.roi_windows:
        stim_sub = _window_sub_epochs(
            preprocessed.stimulus_sub_epochs, n_trials, window,
            cfg.epoch_duration, preprocessed.sfreq,
        )
        # Use baseline sub-epochs (all) as negative class
        bl_sub = preprocessed.baseline_sub_epochs

        # Build binary dataset per trial
        X_pos_list, X_neg_list = [], []
        for t_idx in target_idx:
            base = t_idx * (stim_sub.shape[0] // max(len(target_idx), 1))
            # Simpler: gather windows proportionally
        # Rebuild cleanly
        n_sub_stim = preprocessed.stimulus_sub_epochs.shape[0] // n_trials
        n_sub_bl = preprocessed.baseline_sub_epochs.shape[0] // n_trials
        win_start_i = int(window[0] / cfg.epoch_duration)
        win_end_i = int(window[1] / cfg.epoch_duration)

        X_pos, X_neg = [], []
        for t in range(n_trials):
            block = preprocessed.stimulus_sub_epochs[t * n_sub_stim: (t + 1) * n_sub_stim]
            bl_block = preprocessed.baseline_sub_epochs[t * n_sub_bl: (t + 1) * n_sub_bl]
            window_block = block[win_start_i: win_end_i]
            if target_trial_mask[t]:
                X_pos.append(window_block)
            else:
                X_neg.append(bl_block[:len(window_block)])

        if not X_pos or not X_neg:
            continue

        X_p = np.concatenate(X_pos, axis=0)  # (N_pos, n_ch, n_times)
        X_n = np.concatenate(X_neg, axis=0)
        n_min = min(len(X_p), len(X_n))
        X_p, X_n = X_p[:n_min], X_n[:n_min]
        X = np.concatenate([X_p, X_n], axis=0)
        y = np.concatenate([np.ones(n_min), np.zeros(n_min)])

        if _mne_decoding_available and X.shape[2] > 1:
            clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
            sliding = SlidingEstimator(clf, scoring="accuracy", n_jobs=1, verbose=False)
            try:
                scores = cross_val_multiscore(sliding, X, y, cv=min(3, n_min), n_jobs=1)
                acc = scores.mean(axis=0)  # mean over CV folds → (n_times,)
            except Exception as e:
                warnings.warn(f"SlidingEstimator failed for window {window}: {e}")
                acc = np.array([0.5])
        else:
            # Fallback: flatten and use simple CV accuracy
            from sklearn.model_selection import cross_val_score
            X_flat = X.reshape(len(X), -1)
            clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
            try:
                acc = cross_val_score(clf, X_flat, y, cv=min(3, n_min), scoring="accuracy")
            except Exception:
                acc = np.array([0.5])

        threshold_95 = np.percentile(acc, 95)
        above = acc[acc > threshold_95] if len(acc) > 1 else acc
        mean_score = above.mean() if len(above) > 0 else acc.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_window = window
            best_acc = acc

    return best_window, best_acc


def run_roi_for_all_quadrants(
    preprocessed: PreprocessedData,
    cfg: NeuroSenseConfig,
) -> Dict[str, Tuple[Tuple[float, float], np.ndarray]]:
    """
    Run ROI window selection for each of the 4 Russell quadrants.

    Returns
    -------
    dict mapping quadrant name → (best_window, accuracy_array)
    """
    results = {}
    for quadrant in cfg.quadrant_names:
        window, acc = select_roi_window(preprocessed, quadrant, cfg)
        results[quadrant] = (window, acc)
    return results
