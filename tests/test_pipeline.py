"""Smoke tests for the NeuroSense pipeline using fully synthetic data."""
import numpy as np
import pytest

from neurosense.config import NeuroSenseConfig, assign_quadrant, RUSSELL_QUADRANTS
from neurosense.data_loader import make_synthetic_subject
from neurosense.preprocessing import preprocess_subject
from neurosense.roi_timesliding import run_roi_for_all_quadrants
from neurosense.features import extract_minirocket, build_feature_dataset
from neurosense.modeling import loso_evaluate, LOSOResults
from neurosense.stats import compute_participant_flags


CFG = NeuroSenseConfig()
N_TRIALS = 8   # small for speed


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_frozen():
    cfg = NeuroSenseConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.sfreq = 512  # type: ignore


def test_assign_quadrant_all_four():
    assert assign_quadrant(7.0, 7.0) == "HVHA"
    assert assign_quadrant(3.0, 7.0) == "LVHA"
    assert assign_quadrant(3.0, 3.0) == "LVLA"
    assert assign_quadrant(7.0, 3.0) == "HVLA"


# ---------------------------------------------------------------------------
# data_loader
# ---------------------------------------------------------------------------

def test_synthetic_subject_shape():
    subj = make_synthetic_subject(CFG, n_trials=N_TRIALS)
    # Data includes a pre-roll buffer of baseline_duration before trial 0
    trial_len = CFG.baseline_duration + CFG.stimulus_duration
    n_samples = int((CFG.baseline_duration + N_TRIALS * trial_len) * CFG.sfreq)
    assert subj.eeg_data.shape == (n_samples, len(CFG.channels))
    assert len(subj.markers) == N_TRIALS
    assert len(subj.ratings) == N_TRIALS


def test_synthetic_ratings_have_quadrant():
    subj = make_synthetic_subject(CFG, n_trials=N_TRIALS)
    assert "quadrant" in subj.ratings.columns
    assert set(subj.ratings["quadrant"]).issubset(set(CFG.quadrant_names))


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preprocessed():
    subj = make_synthetic_subject(CFG, n_trials=N_TRIALS)
    return preprocess_subject(subj, CFG)


def test_preprocessing_completes(preprocessed):
    assert preprocessed is not None
    assert preprocessed.n_trials >= 1


def test_epoch_shapes(preprocessed):
    n_ch = len(CFG.channels)
    bl_samps = int(CFG.baseline_duration * CFG.sfreq)
    st_samps = int(CFG.stimulus_duration * CFG.sfreq)
    n = preprocessed.n_trials
    assert preprocessed.baseline_epochs.shape == (n, n_ch, bl_samps)
    assert preprocessed.stimulus_epochs.shape == (n, n_ch, st_samps)


def test_sub_epoch_shapes(preprocessed):
    n_ch = len(CFG.channels)
    win_samps = int(CFG.epoch_duration * CFG.sfreq)
    n_sub_bl = int(CFG.baseline_duration / CFG.epoch_duration)
    n_sub_st = int(CFG.stimulus_duration / CFG.epoch_duration)
    n = preprocessed.n_trials
    assert preprocessed.baseline_sub_epochs.shape == (n * n_sub_bl, n_ch, win_samps)
    assert preprocessed.stimulus_sub_epochs.shape == (n * n_sub_st, n_ch, win_samps)


# ---------------------------------------------------------------------------
# ROI window selection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def roi_maps(preprocessed):
    return run_roi_for_all_quadrants(preprocessed, CFG)


def test_roi_returns_all_quadrants(roi_maps):
    assert set(roi_maps.keys()) == set(CFG.quadrant_names)


def test_roi_windows_are_valid(roi_maps):
    valid_windows = set(CFG.roi_windows)
    for q, (window, acc) in roi_maps.items():
        assert window in valid_windows, f"Invalid window {window} for quadrant {q}"
        assert isinstance(acc, np.ndarray)
        assert len(acc) > 0


# ---------------------------------------------------------------------------
# MiniRocket features
# ---------------------------------------------------------------------------

def test_minirocket_feature_shape(preprocessed):
    sub_epochs = preprocessed.stimulus_sub_epochs[:4]  # first 4 sub-epochs
    features, transformer = extract_minirocket(sub_epochs, CFG, num_kernels=100)
    assert features.shape[0] == 4
    assert features.shape[1] > 0  # some features extracted


# ---------------------------------------------------------------------------
# LOSO with 3 synthetic subjects
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loso_results_fixture():
    subjects = [make_synthetic_subject(CFG, n_trials=N_TRIALS, seed=i) for i in range(3)]
    preprocessed_list = [preprocess_subject(s, CFG) for s in subjects]
    roi_maps_list = [run_roi_for_all_quadrants(p, CFG) for p in preprocessed_list]
    feature_datasets = build_feature_dataset(
        preprocessed_list, roi_maps_list, CFG, num_kernels=100
    )
    return loso_evaluate(feature_datasets, CFG)


def test_loso_completes(loso_results_fixture):
    assert isinstance(loso_results_fixture, LOSOResults)


def test_loso_has_folds(loso_results_fixture):
    assert len(loso_results_fixture.folds) > 0


def test_loso_accuracy_in_range(loso_results_fixture):
    for fold in loso_results_fixture.folds:
        assert 0.0 <= fold.accuracy <= 1.0, f"Accuracy out of range: {fold.accuracy}"


def test_loso_summary_shape(loso_results_fixture):
    summary = loso_results_fixture.summary()
    assert "quadrant" in summary.columns
    assert "accuracy_mean" in summary.columns
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_participant_flags(loso_results_fixture):
    subjects = [make_synthetic_subject(CFG, n_trials=N_TRIALS, seed=i) for i in range(3)]
    all_ratings = __import__("pandas").concat(
        [s.ratings for s in subjects], ignore_index=True
    )
    # Add user_id column
    for i, s in enumerate(subjects):
        s.ratings["user_id"] = s.subject_id
    all_ratings = __import__("pandas").concat(
        [s.ratings for s in subjects], ignore_index=True
    )
    flags = compute_participant_flags(all_ratings, loso_results_fixture, CFG)
    assert "flagged" in flags.columns
    assert len(flags) > 0
