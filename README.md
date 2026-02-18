# NeuroSense EEG Pipeline

End-to-end Python implementation of the methodology from:

> *"NeuroSense: A Novel EEG Dataset Utilizing Low-Cost, Sparse Electrode Devices for Emotion Exploration"* — IEEE Access

Processes 4-channel Muse EEG recordings → Russell Circumplex emotion quadrant classification using **MiniRocket + SVM** with **Leave-One-Subject-Out** cross-validation.

---

## Setup

```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
```

---

## Usage

### Synthetic smoke run (no real data needed)
```bash
python -m neurosense.run --use-synthetic --output results/
```

### Single XDF file
```bash
python -m neurosense.run \
  --xdf-file path/to/sub-P001_eeg.xdf \
  --output results/
```

### Full dataset
```bash
python -m neurosense.run \
  --data-root  path/to/xdf_files/ \
  --ratings    path/to/ratings.csv \
  --labels     path/to/external_labels.csv \
  --output     results/
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| Config | `config.py` | Frozen dataclass with all parameters |
| Load | `data_loader.py` | XDF → SubjectData; synthetic data generator |
| Preprocess | `preprocessing.py` | FIR filter → ringing reduction → KNN outlier removal → epoch extraction |
| ROI | `roi_timesliding.py` | SlidingEstimator selects best 5-s window per quadrant |
| Features | `features.py` | MiniRocketMultivariate on ROI sub-epochs |
| Model | `modeling.py` | LOSO × RandomizedSearchCV (SVM + scaler) |
| Stats | `stats.py` | Pearson reliability, participant flagging |
| Run | `run.py` | Orchestrator + CLI |

---

## Output artifacts (`results/`)

| File | Contents |
|------|----------|
| `per_subject_metrics.csv` | Accuracy, F1 per subject per quadrant |
| `overall_summary.json` | Mean ± std per quadrant |
| `confusion_matrices/cm_*.png` | Aggregated confusion matrix per quadrant |
| `roi_windows.csv` | Best ROI window per subject per quadrant |
| `participant_flags.csv` | Rating STD, mean probability, flagged status |
| `label_reliability.json` | Pearson r/p for valence and arousal |

---

## Device / data format

- **Headset**: Muse (4 channels: TP9, AF7, AF8, TP10 at 256 Hz)
- **Recording**: LabRecorder → XDF format
- **Ratings**: CSV with columns `user_id, video_id, valence, arousal, dominance, liking, familiarity` (1–9 SAM scale)
- **External labels**: CSV with columns `video_id, quadrant, valence_ext, arousal_ext`
