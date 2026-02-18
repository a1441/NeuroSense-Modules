"""Data loading utilities for NeuroSense XDF files and ratings CSVs."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyxdf

from .config import NeuroSenseConfig, assign_quadrant


@dataclass
class SubjectData:
    """Raw data container for one recording session."""
    subject_id: str
    eeg_data: np.ndarray          # shape (n_samples, n_channels)
    sfreq: float
    channel_names: List[str]
    timestamps: np.ndarray        # LSL timestamps, shape (n_samples,)
    markers: pd.DataFrame         # columns: timestamp, label  (may be empty)
    ratings: pd.DataFrame         # columns: video_id, valence, arousal, dominance, liking, familiarity, quadrant


# ---------------------------------------------------------------------------
# XDF loading
# ---------------------------------------------------------------------------

def _find_stream(streams: list, stream_type: str) -> Optional[dict]:
    """Return first stream matching *stream_type* (case-insensitive)."""
    for s in streams:
        if s["info"].get("type", [""])[0].lower() == stream_type.lower():
            return s
    return None


def load_neurosense_subject(
    xdf_path: str | Path,
    ratings_path: Optional[str | Path],
    cfg: NeuroSenseConfig,
    subject_id: Optional[str] = None,
) -> SubjectData:
    """
    Load one subject from an XDF file.

    Parameters
    ----------
    xdf_path:      Path to the .xdf recording.
    ratings_path:  Path to the ratings CSV, or None for synthetic/missing ratings.
    cfg:           Pipeline configuration.
    subject_id:    Optional label; inferred from filename if not given.

    Returns
    -------
    SubjectData
    """
    xdf_path = Path(xdf_path)
    if subject_id is None:
        subject_id = xdf_path.stem

    streams, _ = pyxdf.load_xdf(str(xdf_path))

    # --- EEG stream ---
    eeg_stream = _find_stream(streams, "EEG")
    if eeg_stream is None:
        raise ValueError(f"No EEG stream found in {xdf_path}")

    eeg_data = eeg_stream["time_series"].astype(np.float64)   # (n_samples, n_ch)
    timestamps = eeg_stream["time_stamps"]
    sfreq_reported = float(eeg_stream["info"].get("nominal_srate", [cfg.sfreq])[0])

    # Parse channel labels from XDF header
    try:
        ch_info = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        raw_labels = [c["label"][0] for c in ch_info]
    except (KeyError, IndexError, TypeError):
        raw_labels = [f"ch{i}" for i in range(eeg_data.shape[1])]

    # Keep only the 4 NeuroSense channels; drop AUX etc.
    wanted = list(cfg.channels)
    keep_idx = [i for i, lbl in enumerate(raw_labels) if lbl in wanted]
    if not keep_idx:
        warnings.warn(
            f"None of {wanted} found in {raw_labels}; using first {len(wanted)} channels.",
            stacklevel=2,
        )
        keep_idx = list(range(min(len(wanted), eeg_data.shape[1])))
        channel_names = raw_labels[: len(keep_idx)]
    else:
        channel_names = [raw_labels[i] for i in keep_idx]

    eeg_data = eeg_data[:, keep_idx]

    # --- Marker stream (optional) ---
    marker_stream = _find_stream(streams, "Markers")
    if marker_stream is None:
        marker_stream = _find_stream(streams, "Marker")
    if marker_stream is not None:
        m_labels = [str(m[0]) if m else "" for m in marker_stream["time_series"]]
        markers = pd.DataFrame(
            {"timestamp": marker_stream["time_stamps"], "label": m_labels}
        )
    else:
        warnings.warn(
            f"No marker stream in {xdf_path}. Trials will be inferred from recording length.",
            stacklevel=2,
        )
        markers = _infer_trial_markers(timestamps, cfg)

    # --- Ratings ---
    if ratings_path is not None and Path(ratings_path).exists():
        ratings = _load_ratings(Path(ratings_path), subject_id)
    else:
        ratings = _synthetic_ratings(len(markers), cfg)

    return SubjectData(
        subject_id=subject_id,
        eeg_data=eeg_data,
        sfreq=sfreq_reported,
        channel_names=channel_names,
        timestamps=timestamps,
        markers=markers,
        ratings=ratings,
    )


def _infer_trial_markers(timestamps: np.ndarray, cfg: NeuroSenseConfig) -> pd.DataFrame:
    """
    When no marker stream exists, divide the recording into equal-length trials
    of (baseline_duration + stimulus_duration) seconds.
    """
    trial_len = cfg.baseline_duration + cfg.stimulus_duration
    t_start = timestamps[0]
    t_end = timestamps[-1]
    n_trials = int((t_end - t_start) // trial_len)
    if n_trials == 0:
        n_trials = 1
    onsets = [t_start + i * trial_len for i in range(n_trials)]
    return pd.DataFrame({"timestamp": onsets, "label": [f"trial_{i}" for i in range(n_trials)]})


def _load_ratings(ratings_path: Path, subject_id: str) -> pd.DataFrame:
    """
    Load ratings CSV.  Expected columns:
      user_id, video_id, valence, arousal, dominance, liking, familiarity
    Filters to the given subject_id and adds a 'quadrant' column.
    """
    df = pd.read_csv(ratings_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Filter to subject if column present
    if "user_id" in df.columns:
        subject_rows = df[df["user_id"].astype(str) == str(subject_id)]
        if subject_rows.empty:
            warnings.warn(f"No ratings found for subject {subject_id}; using all rows.")
        else:
            df = subject_rows.reset_index(drop=True)

    if "valence" in df.columns and "arousal" in df.columns:
        df["quadrant"] = df.apply(
            lambda r: assign_quadrant(r["valence"], r["arousal"]), axis=1
        )
    return df.reset_index(drop=True)


def _synthetic_ratings(n_trials: int, cfg: NeuroSenseConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    valence = rng.uniform(1, 9, n_trials)
    arousal = rng.uniform(1, 9, n_trials)
    quadrants = [assign_quadrant(v, a) for v, a in zip(valence, arousal)]
    return pd.DataFrame(
        {
            "video_id": range(n_trials),
            "valence": valence,
            "arousal": arousal,
            "dominance": rng.uniform(1, 9, n_trials),
            "liking": rng.uniform(1, 9, n_trials),
            "familiarity": rng.uniform(1, 9, n_trials),
            "quadrant": quadrants,
        }
    )


# ---------------------------------------------------------------------------
# External labels
# ---------------------------------------------------------------------------

def load_external_labels(labels_path: str | Path) -> pd.DataFrame:
    """
    Load expert/external quadrant labels CSV.
    Expected columns: video_id, quadrant, valence_ext, arousal_ext
    """
    df = pd.read_csv(labels_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Synthetic subject (for unit tests — no real data needed)
# ---------------------------------------------------------------------------

def make_synthetic_subject(
    cfg: NeuroSenseConfig,
    n_trials: int = 10,
    seed: int = 42,
) -> SubjectData:
    """
    Generate a fully synthetic SubjectData with gaussian noise EEG.
    Safe to use in unit tests without any real recording.
    """
    rng = np.random.default_rng(seed)
    trial_len = cfg.baseline_duration + cfg.stimulus_duration
    # Prepend one baseline worth of buffer so trial 0 has room for its baseline epoch
    pre_roll = cfg.baseline_duration
    total_samples = int((pre_roll + n_trials * trial_len) * cfg.sfreq)
    n_ch = len(cfg.channels)

    eeg_data = rng.standard_normal((total_samples, n_ch)).astype(np.float64) * 10.0  # µV-scale
    timestamps = np.arange(total_samples) / cfg.sfreq

    # Markers at stimulus onset (after the baseline pre-roll)
    onsets = [pre_roll + i * trial_len for i in range(n_trials)]
    markers = pd.DataFrame(
        {"timestamp": onsets, "label": [f"trial_{i}" for i in range(n_trials)]}
    )

    ratings = _synthetic_ratings(n_trials, cfg)

    return SubjectData(
        subject_id=f"sub_{seed:03d}",
        eeg_data=eeg_data,
        sfreq=cfg.sfreq,
        channel_names=list(cfg.channels),
        timestamps=timestamps,
        markers=markers,
        ratings=ratings,
    )
