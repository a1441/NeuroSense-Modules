# NeuroSense EEG Pipeline — Project Context

## What this project is
End-to-end Python implementation of the methodology from the IEEE Access paper
"NeuroSense: A Novel EEG Dataset Utilizing Low-Cost, Sparse Electrode Devices for
Emotion Exploration". Processes 4-channel Muse EEG data → emotion quadrant classification
using MiniRocket + SVM with LOSO cross-validation.

## Repository
- GitHub: https://github.com/a1441/NeuroSense-Modules
- Default branch: `main`

## Environment
- Location: `D:\Claude\Neuro\`
- Python venv: `D:\Claude\Neuro\.venv\` (Python 3.12)
- Activate: `D:\Claude\Neuro\.venv\Scripts\activate`
- Run tests: `D:\Claude\Neuro\.venv\Scripts\pytest tests/ -v`
- Run pipeline: `D:\Claude\Neuro\.venv\Scripts\python -m neurosense.run --help`

## Sample data
- XDF: `d:\NeuroStuff\LabRecorder\Files\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf`
- Device: Muse headset (Muse-2889)
- EEG stream: Stream index 3, type="EEG", 5 channels, 256 Hz, ~198 s
  - Channels: ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
  - Shape: (50688, 5) — drop 'Right AUX', keep first 4
- Other streams: Accelerometer (3ch@52Hz), Gyroscope (3ch@52Hz), PPG (3ch@64Hz)
- NO marker stream in this recording — treat as continuous / create synthetic markers

## Package versions installed (key)
- mne, meegkit, scikit-learn, sktime[all_extras], pyxdf, numpy, scipy
- pandas, matplotlib, seaborn, joblib, pytest

## Module map
```
neurosense/
  __init__.py          # version only
  config.py            # NeuroSenseConfig frozen dataclass + RUSSELL_QUADRANTS
  data_loader.py       # load_neurosense_subject(), make_synthetic_subject(), SubjectData
  preprocessing.py     # FIR filter, ringing reduction, KNN ocular removal, sub-epoch split
  roi_timesliding.py   # SlidingEstimator per quadrant → best 5-s ROI window
  features.py          # MiniRocketMultivariate feature extraction
  modeling.py          # build_pipeline(), loso_evaluate(), LOSOResults
  stats.py             # label reliability (Pearson r), participant flags
  run.py               # main() orchestrator + argparse CLI
tests/
  test_pipeline.py     # smoke tests using make_synthetic_subject()
results/               # output artifacts (gitignored)
```

## Key design decisions
- `NeuroSenseConfig` is a frozen dataclass — pass `cfg` everywhere, no globals
- XDF loading: use `pyxdf.load_xdf()`, find stream by `type=="EEG"`
- Missing marker stream: graceful fallback — divide recording into equal-length trials
- meegkit ringing reduction: wrapped in try/except, no-op fallback with warning
- KNN outlier removal: `sklearn.neighbors.NearestNeighbors`, interpolate outlier samples
- MiniRocket input shape: `(n_instances, n_channels, n_timepoints)` — sktime convention
- LOSO: `sklearn.model_selection.LeaveOneOut` outer loop, `RandomizedSearchCV` inner
- Results saved to `results/` dir (CSV, JSON, PNG confusion matrices)

## Russell Circumplex quadrants
- Q1 HVHA: High Valence, High Arousal (happy, excited)
- Q2 LVHA: Low Valence, High Arousal (angry, fearful)
- Q3 LVLA: Low Valence, Low Arousal (sad, bored)
- Q4 HVLA: High Valence, Low Arousal (calm, relaxed)

## Status
- [x] Project scaffold created
- [x] .venv created and all packages installed
- [x] .gitignore, pyproject.toml written
- [x] neurosense/__init__.py, tests/__init__.py written
- [ ] config.py
- [ ] data_loader.py
- [ ] preprocessing.py
- [ ] roi_timesliding.py
- [ ] features.py
- [ ] modeling.py
- [ ] stats.py
- [ ] run.py
- [ ] tests/test_pipeline.py
- [ ] README.md
