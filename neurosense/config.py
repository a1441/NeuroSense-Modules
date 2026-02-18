"""NeuroSenseConfig — single source of truth for all pipeline parameters."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class NeuroSenseConfig:
    # --- Device / recording ---
    sfreq: float = 256.0
    channels: Tuple[str, ...] = ("AF7", "AF8", "TP9", "TP10")
    ch_types: str = "eeg"
    reference: str = "average"          # MNE average reference

    # --- Bandpass filter ---
    bandpass: Tuple[float, float] = (1.0, 45.0)
    fir_window: str = "hamming"

    # --- KNN ocular outlier removal ---
    knn_contamination: float = 0.1      # fraction of samples treated as outliers
    knn_n_neighbors: int = 5

    # --- Trial / epoch structure ---
    epoch_duration: float = 5.0         # sub-epoch length in seconds
    baseline_duration: float = 5.0      # baseline period before stimulus
    stimulus_duration: float = 60.0     # stimulus (video) duration in seconds

    # --- ROI time-sliding windows (start, end) in seconds relative to stimulus onset ---
    roi_windows: Tuple[Tuple[float, float], ...] = (
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
    )

    # --- Circumplex / label ---
    n_quadrants: int = 4
    quadrant_names: Tuple[str, ...] = ("HVHA", "LVHA", "LVLA", "HVLA")

    # --- Hyper-parameter search ---
    n_random_search_iter: int = 50
    cv_inner: int = 3
    top_n_models: int = 30

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Participant flagging ---
    flag_percentile: float = 25.0       # flag if STD or mean-prob below this percentile


# Russell Circumplex quadrant definitions
# key -> (valence_sign, arousal_sign)  where +1 = high, -1 = low
RUSSELL_QUADRANTS = {
    "HVHA": (+1, +1),   # High Valence, High Arousal  (happy, excited)
    "LVHA": (-1, +1),   # Low Valence,  High Arousal  (angry, fearful)
    "LVLA": (-1, -1),   # Low Valence,  Low Arousal   (sad, bored)
    "HVLA": (+1, -1),   # High Valence, Low Arousal   (calm, relaxed)
}


def assign_quadrant(valence: float, arousal: float) -> str:
    """Map a (valence, arousal) pair on a 1-9 SAM scale to a Russell quadrant."""
    mid = 5.0
    v_sign = +1 if valence >= mid else -1
    a_sign = +1 if arousal >= mid else -1
    for name, (vs, as_) in RUSSELL_QUADRANTS.items():
        if vs == v_sign and as_ == a_sign:
            return name
    raise ValueError(f"Could not assign quadrant for valence={valence}, arousal={arousal}")
