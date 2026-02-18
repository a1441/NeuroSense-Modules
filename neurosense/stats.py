"""Statistical analyses: label reliability (Pearson r) and participant flagging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .config import NeuroSenseConfig
from .modeling import LOSOResults


@dataclass
class ReliabilityResult:
    valence_r: float
    valence_p: float
    arousal_r: float
    arousal_p: float

    def __str__(self) -> str:
        return (
            f"Label reliability:\n"
            f"  Valence : r={self.valence_r:.3f}  p={self.valence_p:.4f}\n"
            f"  Arousal : r={self.arousal_r:.3f}  p={self.arousal_p:.4f}"
        )


def compute_label_reliability(
    self_assessment_df: pd.DataFrame,
    external_labels_df: pd.DataFrame,
) -> ReliabilityResult:
    """
    Pearson correlation between participant self-assessments and external (expert) labels.

    self_assessment_df : columns include video_id, valence, arousal
    external_labels_df : columns include video_id, valence_ext, arousal_ext
    """
    merged = self_assessment_df.merge(external_labels_df, on="video_id", how="inner")
    if len(merged) < 3:
        return ReliabilityResult(
            valence_r=float("nan"),
            valence_p=float("nan"),
            arousal_r=float("nan"),
            arousal_p=float("nan"),
        )

    # Average self-assessments per video across participants first
    agg = merged.groupby("video_id").agg(
        valence_mean=("valence", "mean"),
        arousal_mean=("arousal", "mean"),
        valence_ext=("valence_ext", "first"),
        arousal_ext=("arousal_ext", "first"),
    ).reset_index()

    val_r, val_p = pearsonr(agg["valence_mean"], agg["valence_ext"])
    aro_r, aro_p = pearsonr(agg["arousal_mean"], agg["arousal_ext"])

    return ReliabilityResult(
        valence_r=float(val_r),
        valence_p=float(val_p),
        arousal_r=float(aro_r),
        arousal_p=float(aro_p),
    )


def compute_participant_flags(
    self_assessment_df: pd.DataFrame,
    loso_results: LOSOResults,
    cfg: NeuroSenseConfig,
) -> pd.DataFrame:
    """
    Flag participants whose ratings or model confidence are unusually low.

    Flagging criteria (either triggers):
      - STD of valence, arousal, or dominance across videos < 25th percentile
      - Mean max decision probability from LOSO < 25th percentile

    Returns a DataFrame with one row per subject.
    """
    # Per-participant rating variability
    id_col = "user_id" if "user_id" in self_assessment_df.columns else None
    if id_col is None:
        # No user_id — treat whole df as one participant
        self_assessment_df = self_assessment_df.copy()
        self_assessment_df["user_id"] = "p0"
        id_col = "user_id"

    variability = (
        self_assessment_df.groupby(id_col)
        .agg(
            valence_std=("valence", "std"),
            arousal_std=("arousal", "std"),
            dominance_std=("dominance", "std") if "dominance" in self_assessment_df.columns else ("valence", "std"),
        )
        .reset_index()
    )
    variability.rename(columns={id_col: "subject_id"}, inplace=True)

    # Mean decision probability from LOSO
    mean_prob = loso_results.mean_decision_probability()
    variability["mean_decision_prob"] = variability["subject_id"].map(mean_prob).fillna(np.nan)

    # Flagging thresholds (25th percentile)
    p = cfg.flag_percentile
    v_thresh = np.nanpercentile(variability["valence_std"], p)
    a_thresh = np.nanpercentile(variability["arousal_std"], p)
    d_thresh = np.nanpercentile(variability["dominance_std"], p)
    prob_thresh = np.nanpercentile(variability["mean_decision_prob"].dropna(), p) if variability["mean_decision_prob"].notna().any() else 0.0

    variability["flagged"] = (
        (variability["valence_std"] < v_thresh)
        | (variability["arousal_std"] < a_thresh)
        | (variability["dominance_std"] < d_thresh)
        | (variability["mean_decision_prob"] < prob_thresh)
    )

    return variability


def print_results_table(loso_results: LOSOResults, flags: Optional[pd.DataFrame] = None) -> None:
    """Print a formatted summary table to stdout."""
    summary = loso_results.summary()
    print("\n" + "=" * 60)
    print("LOSO CLASSIFICATION RESULTS")
    print("=" * 60)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if flags is not None:
        n_flagged = flags["flagged"].sum()
        print(f"\nParticipant flags: {n_flagged}/{len(flags)} flagged")
        if n_flagged > 0:
            print(flags[flags["flagged"]][["subject_id", "valence_std", "arousal_std", "mean_decision_prob"]].to_string(index=False))
    print("=" * 60 + "\n")
