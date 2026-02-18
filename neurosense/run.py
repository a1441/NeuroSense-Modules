"""
NeuroSense pipeline orchestrator.

Usage
-----
Synthetic smoke run:
  python -m neurosense.run --use-synthetic --output results/

Real data:
  python -m neurosense.run \\
      --data-root /path/to/xdf/files \\
      --ratings   /path/to/ratings.csv \\
      --labels    /path/to/external_labels.csv \\
      --output    results/

Single XDF file:
  python -m neurosense.run \\
      --xdf-file  /path/to/sub-P001_eeg.xdf \\
      --output    results/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mne")


def _save_confusion_matrices(loso_results, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    from itertools import groupby
    from .modeling import FoldResult

    # Aggregate CM per quadrant
    quadrants = sorted({f.quadrant for f in loso_results.folds})
    for q in quadrants:
        folds = [f for f in loso_results.folds if f.quadrant == q]
        agg_cm = sum(f.confusion_matrix for f in folds)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(agg_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Quadrant {q}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(cm_dir / f"cm_{q}.png", dpi=150)
        plt.close(fig)


def export_results(loso_results, reliability, flags, roi_maps, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject metrics
    per_subj = loso_results.per_subject_df()
    per_subj.to_csv(output_dir / "per_subject_metrics.csv", index=False)

    # Overall summary
    summary = loso_results.summary()
    summary.to_json(output_dir / "overall_summary.json", orient="records", indent=2)

    # ROI windows
    roi_records = []
    for subj_idx, roi_map in enumerate(roi_maps):
        for quadrant, (window, _) in roi_map.items():
            roi_records.append(
                {"subject": subj_idx, "quadrant": quadrant, "window_start": window[0], "window_end": window[1]}
            )
    pd.DataFrame(roi_records).to_csv(output_dir / "roi_windows.csv", index=False)

    # Participant flags
    if flags is not None:
        flags.to_csv(output_dir / "participant_flags.csv", index=False)

    # Label reliability
    if reliability is not None:
        rel_dict = {
            "valence_r": reliability.valence_r,
            "valence_p": reliability.valence_p,
            "arousal_r": reliability.arousal_r,
            "arousal_p": reliability.arousal_p,
        }
        with open(output_dir / "label_reliability.json", "w") as fh:
            json.dump(rel_dict, fh, indent=2)

    # Confusion matrices
    try:
        _save_confusion_matrices(loso_results, output_dir)
    except Exception as e:
        print(f"  [warn] Could not save confusion matrix plots: {e}")

    print(f"Results saved to: {output_dir.resolve()}")


def main(
    data_root: Optional[str] = None,
    xdf_file: Optional[str] = None,
    ratings_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    output_dir: str = "results",
    use_synthetic: bool = False,
    n_synthetic_subjects: int = 3,
    n_synthetic_trials: int = 10,
) -> None:
    from .config import NeuroSenseConfig
    from .data_loader import (
        load_neurosense_subject,
        load_external_labels,
        make_synthetic_subject,
    )
    from .preprocessing import preprocess_subject
    from .roi_timesliding import run_roi_for_all_quadrants
    from .features import build_feature_dataset
    from .modeling import loso_evaluate
    from .stats import (
        compute_label_reliability,
        compute_participant_flags,
        print_results_table,
    )

    cfg = NeuroSenseConfig()
    output_path = Path(output_dir)

    print("=" * 55)
    print("  NeuroSense EEG Pipeline")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # 1. Load subjects
    # -----------------------------------------------------------------------
    subjects = []

    if use_synthetic:
        print(f"  Mode: SYNTHETIC ({n_synthetic_subjects} subjects, {n_synthetic_trials} trials each)")
        for i in range(n_synthetic_subjects):
            subjects.append(make_synthetic_subject(cfg, n_trials=n_synthetic_trials, seed=cfg.random_seed + i))
    elif xdf_file:
        print(f"  Mode: SINGLE FILE — {xdf_file}")
        subjects.append(load_neurosense_subject(xdf_file, ratings_path, cfg))
    elif data_root:
        print(f"  Mode: DIRECTORY — {data_root}")
        xdf_files = sorted(Path(data_root).rglob("*.xdf"))
        if not xdf_files:
            raise FileNotFoundError(f"No .xdf files found under {data_root}")
        for xdf in xdf_files:
            try:
                subjects.append(load_neurosense_subject(xdf, ratings_path, cfg))
            except Exception as e:
                print(f"  [warn] Skipping {xdf.name}: {e}")
    else:
        raise ValueError("Provide --use-synthetic, --xdf-file, or --data-root")

    print(f"  Loaded {len(subjects)} subject(s)")

    # -----------------------------------------------------------------------
    # 2. Preprocess
    # -----------------------------------------------------------------------
    print("  Preprocessing...")
    preprocessed_list = []
    for subj in subjects:
        try:
            preprocessed_list.append(preprocess_subject(subj, cfg))
        except Exception as e:
            print(f"  [warn] Preprocessing failed for {subj.subject_id}: {e}")

    if not preprocessed_list:
        raise RuntimeError("All subjects failed preprocessing.")

    print(f"  Preprocessed {len(preprocessed_list)} subject(s)")

    # -----------------------------------------------------------------------
    # 3. ROI time-sliding
    # -----------------------------------------------------------------------
    print("  Running ROI window selection...")
    roi_maps = []
    for p in preprocessed_list:
        roi_maps.append(run_roi_for_all_quadrants(p, cfg))

    # -----------------------------------------------------------------------
    # 4. Feature extraction
    # -----------------------------------------------------------------------
    print("  Extracting MiniRocket features...")
    feature_datasets = build_feature_dataset(preprocessed_list, roi_maps, cfg)

    # -----------------------------------------------------------------------
    # 5. LOSO evaluation
    # -----------------------------------------------------------------------
    print("  Running LOSO evaluation...")
    loso_results = loso_evaluate(feature_datasets, cfg)

    if not loso_results.folds:
        print("  [warn] No LOSO folds completed (need >= 2 subjects with data).")
        print("  Exiting without results export.")
        return

    # -----------------------------------------------------------------------
    # 6. Stats
    # -----------------------------------------------------------------------
    reliability = None
    flags = None

    # Combine all ratings
    all_ratings = pd.concat([p.ratings for p in preprocessed_list], ignore_index=True)

    if labels_path and Path(labels_path).exists():
        external_labels = load_external_labels(labels_path)
        if "video_id" in all_ratings.columns and "video_id" in external_labels.columns:
            reliability = compute_label_reliability(all_ratings, external_labels)
            print(f"  {reliability}")

    if "valence" in all_ratings.columns and "arousal" in all_ratings.columns:
        flags = compute_participant_flags(all_ratings, loso_results, cfg)

    # -----------------------------------------------------------------------
    # 7. Print + export
    # -----------------------------------------------------------------------
    print_results_table(loso_results, flags)
    export_results(loso_results, reliability, flags, roi_maps, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroSense EEG emotion classification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", help="Directory containing .xdf files (recursive search)")
    parser.add_argument("--xdf-file", help="Path to a single .xdf file")
    parser.add_argument("--ratings", dest="ratings_path", help="Path to ratings CSV")
    parser.add_argument("--labels", dest="labels_path", help="Path to external labels CSV")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--use-synthetic", action="store_true", help="Run with synthetic data")
    parser.add_argument("--n-subjects", type=int, default=3, help="Number of synthetic subjects")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials per synthetic subject")

    args = parser.parse_args()
    main(
        data_root=args.data_root,
        xdf_file=args.xdf_file,
        ratings_path=args.ratings_path,
        labels_path=args.labels_path,
        output_dir=args.output,
        use_synthetic=args.use_synthetic,
        n_synthetic_subjects=args.n_subjects,
        n_synthetic_trials=args.n_trials,
    )


if __name__ == "__main__":
    cli_main()
