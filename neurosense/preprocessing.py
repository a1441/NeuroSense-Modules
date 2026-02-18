"""EEG preprocessing: filter, ringing reduction, outlier removal, sub-epoch splitting."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import NeuroSenseConfig
from .data_loader import SubjectData

mne.set_log_level("WARNING")


@dataclass
class PreprocessedData:
    subject_id: str
    baseline_epochs: np.ndarray     # (n_trials, n_channels, baseline_samples)
    stimulus_epochs: np.ndarray     # (n_trials, n_channels, stimulus_samples)
    baseline_sub_epochs: np.ndarray # (n_trials * n_sub, n_channels, sub_samples)
    stimulus_sub_epochs: np.ndarray # (n_trials * n_sub, n_channels, sub_samples)
    n_trials: int
    sfreq: float
    channel_names: List[str]
    ratings: object                 # DataFrame passthrough


# ---------------------------------------------------------------------------
# Step 1 — Build MNE Raw from SubjectData
# ---------------------------------------------------------------------------

def _to_mne_raw(subject: SubjectData, cfg: NeuroSenseConfig) -> mne.io.RawArray:
    info = mne.create_info(
        ch_names=subject.channel_names,
        sfreq=subject.sfreq,
        ch_types=cfg.ch_types,
    )
    # MNE expects (n_channels, n_times) and data in Volts
    data_v = subject.eeg_data.T * 1e-6
    raw = mne.io.RawArray(data_v, info, verbose=False)

    # Standard 10-20 montage — use only what's available
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


# ---------------------------------------------------------------------------
# Step 2 — FIR bandpass filter
# ---------------------------------------------------------------------------

def apply_fir_filter(raw: mne.io.Raw, cfg: NeuroSenseConfig) -> mne.io.Raw:
    l_freq, h_freq = cfg.bandpass
    return raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_window=cfg.fir_window,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Step 3 — Ringing / muscle artifact reduction (meegkit, with fallback)
# ---------------------------------------------------------------------------

def reduce_ringing_artifacts(raw: mne.io.Raw, cfg: NeuroSenseConfig) -> mne.io.Raw:
    try:
        from meegkit.utils import covariances as _  # noqa: F401 — check availability
        # meegkit doesn't expose a single "ringing" function; use DSS denoising
        try:
            from meegkit.dss import dss_line
            data = raw.get_data()  # (n_ch, n_times)
            # dss_line expects (n_times, n_ch, n_epochs) — reshape, apply, reshape back
            data_t = data.T[:, :, np.newaxis]
            result = dss_line(data_t, fline=50.0, sfreq=raw.info["sfreq"])
            cleaned = result[0]
            raw_clean = raw.copy()
            raw_clean._data = cleaned[:, :, 0].T
            return raw_clean
        except Exception as e:
            warnings.warn(f"meegkit dss_line failed ({e}); skipping ringing reduction.")
            return raw
    except ImportError:
        warnings.warn("meegkit not available; skipping ringing artifact reduction.")
        return raw


# ---------------------------------------------------------------------------
# Step 4 — KNN-based ocular outlier removal
# ---------------------------------------------------------------------------

def remove_ocular_outliers(data: np.ndarray, cfg: NeuroSenseConfig) -> np.ndarray:
    """
    Detect and interpolate outlier samples using KNN distance thresholding.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)

    Returns
    -------
    np.ndarray, same shape, with outlier samples linearly interpolated.
    """
    n_ch, n_times = data.shape
    X = data.T  # (n_times, n_ch) — each time-point is one sample

    nbrs = NearestNeighbors(n_neighbors=cfg.knn_n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = distances[:, -1]  # distance to k-th neighbour

    threshold = np.percentile(k_dist, 100 * (1 - cfg.knn_contamination))
    outlier_mask = k_dist > threshold

    if not np.any(outlier_mask):
        return data

    cleaned = data.copy()
    indices = np.arange(n_times)
    for ch in range(n_ch):
        good = ~outlier_mask
        cleaned[ch] = np.interp(indices, indices[good], data[ch, good])

    return cleaned


# ---------------------------------------------------------------------------
# Step 5 — Epoch extraction
# ---------------------------------------------------------------------------

def build_epochs(
    data: np.ndarray,
    timestamps: np.ndarray,
    markers_df,
    sfreq: float,
    cfg: NeuroSenseConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract baseline and stimulus epochs for each trial marker.

    Returns
    -------
    baseline_epochs : (n_trials, n_ch, baseline_samples)
    stimulus_epochs : (n_trials, n_ch, stimulus_samples)
    """
    baseline_samps = int(cfg.baseline_duration * sfreq)
    stim_samps = int(cfg.stimulus_duration * sfreq)
    t0 = timestamps[0]
    n_total = data.shape[1]

    baselines, stimuli = [], []
    for _, row in markers_df.iterrows():
        onset_s = float(row["timestamp"]) - t0
        onset_idx = int(onset_s * sfreq)

        # Baseline: the `baseline_duration` seconds immediately before stimulus onset
        bl_start = onset_idx - baseline_samps
        bl_end = onset_idx

        # Stimulus: `stimulus_duration` seconds starting at onset
        st_start = onset_idx
        st_end = onset_idx + stim_samps

        if bl_start < 0 or st_end > n_total:
            continue  # skip trials with out-of-bounds windows

        baselines.append(data[:, bl_start:bl_end])
        stimuli.append(data[:, st_start:st_end])

    if not baselines:
        raise ValueError("No valid epochs could be extracted. Check marker timestamps vs recording length.")

    return np.stack(baselines), np.stack(stimuli)


# ---------------------------------------------------------------------------
# Step 6 — Split epochs into non-overlapping sub-epochs
# ---------------------------------------------------------------------------

def split_into_sub_epochs(
    epochs: np.ndarray,
    sfreq: float,
    cfg: NeuroSenseConfig,
) -> np.ndarray:
    """
    Slice each epoch into non-overlapping windows of `epoch_duration` seconds.

    Parameters
    ----------
    epochs : (n_trials, n_ch, n_times)

    Returns
    -------
    sub_epochs : (n_trials * n_windows, n_ch, window_samples)
    """
    n_trials, n_ch, n_times = epochs.shape
    win_samps = int(cfg.epoch_duration * sfreq)
    n_windows = n_times // win_samps

    sub_list = []
    for trial in epochs:
        for w in range(n_windows):
            sub_list.append(trial[:, w * win_samps : (w + 1) * win_samps])

    return np.stack(sub_list)  # (n_trials * n_windows, n_ch, win_samps)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess_subject(subject: SubjectData, cfg: NeuroSenseConfig) -> PreprocessedData:
    """
    Full preprocessing chain:
      1. Build MNE Raw
      2. Apply FIR bandpass
      3. Ringing reduction
      4. KNN ocular outlier removal
      5. Build baseline + stimulus epochs
      6. Split into sub-epochs
    """
    raw = _to_mne_raw(subject, cfg)
    raw = apply_fir_filter(raw, cfg)
    raw = reduce_ringing_artifacts(raw, cfg)

    data = raw.get_data()  # (n_ch, n_times), Volts → back to µV for consistency
    data = data * 1e6
    data = remove_ocular_outliers(data, cfg)

    baseline_epochs, stimulus_epochs = build_epochs(
        data, subject.timestamps, subject.markers, subject.sfreq, cfg
    )

    baseline_sub = split_into_sub_epochs(baseline_epochs, subject.sfreq, cfg)
    stimulus_sub = split_into_sub_epochs(stimulus_epochs, subject.sfreq, cfg)

    return PreprocessedData(
        subject_id=subject.subject_id,
        baseline_epochs=baseline_epochs,
        stimulus_epochs=stimulus_epochs,
        baseline_sub_epochs=baseline_sub,
        stimulus_sub_epochs=stimulus_sub,
        n_trials=len(baseline_epochs),
        sfreq=subject.sfreq,
        channel_names=subject.channel_names,
        ratings=subject.ratings,
    )
