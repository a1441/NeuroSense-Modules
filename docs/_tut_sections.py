"""HTML content sections for generate_tutorial.py — imported at render time."""
from __future__ import annotations


def build_sections(B, CFG, subjects, preprocessed, roi_maps, feat_ds, loso,
                   summary_df, per_subj_df, flags, NT,
                   fig_html, callout, code, term, schema_table, df_table,
                   section_wrap, subsec, have_now, checklist, mono, shape_badge):

    BL_SAMPS  = int(CFG.baseline_duration * CFG.sfreq)
    ST_SAMPS  = int(CFG.stimulus_duration  * CFG.sfreq)
    SUB_SAMPS = int(CFG.epoch_duration     * CFG.sfreq)
    N_SUB_ST  = int(CFG.stimulus_duration  / CFG.epoch_duration)
    N_SUB_BL  = int(CFG.baseline_duration  / CFG.epoch_duration)

    # ── S0: Mental Model ─────────────────────────────────────────────────────
    S_MENTAL = section_wrap("mental", "", "One-Page Mental Model (TL;DR)",
        subsec("What problem we solve",
            "<p>You have recordings from a cheap 4-electrode EEG headset (Muse). "
            "Each recording captures brain activity while a participant watches emotion-evoking videos. "
            "Afterward they rate each video on <strong>valence</strong> (pleasant &harr; unpleasant) "
            "and <strong>arousal</strong> (excited &harr; calm) on a 1&ndash;9 SAM scale.</p>"
            "<p>The question: <em>can the raw EEG signal alone predict which emotional quadrant "
            "the participant was in?</em> A working classifier could enable real-time passive "
            "emotion monitoring without explicit self-reports.</p>",
        ),
        subsec("What goes in / what comes out",
            schema_table(
                ["Input", "Format", "Example"],
                [
                    ["EEG recording", "XDF file", "sub-P001_ses-S001_task-Default_run-001_eeg.xdf"],
                    ["Participant ratings", "CSV", "user_id=P001, video_id=3, valence=7.2, arousal=3.1"],
                    ["Expert labels (optional)", "CSV", "video_id=3, valence_ext=6.8, arousal_ext=2.9"],
                ],
            ),
            schema_table(
                ["Output", "Format", "Answers the question"],
                [
                    ["per_subject_metrics.csv", "CSV", "How accurate was the model for each person?"],
                    ["overall_summary.json",    "JSON","Mean accuracy across all subjects per quadrant"],
                    ["confusion_matrices/",     "PNG", "Which quadrants got confused with which?"],
                    ["roi_windows.csv",         "CSV", "Which 5-s window of the video was most informative?"],
                    ["participant_flags.csv",   "CSV", "Which participants gave noisy/flat responses?"],
                    ["label_reliability.json",  "JSON","Do self-reports agree with expert labels?"],
                ],
            ),
        ),
        subsec("The 7 pipeline stages",
            fig_html(B["PIPELINE"], "Pipeline overview: XDF file in, classification results out"),
        ),
        subsec("Common rookie mistakes",
            """<div class="mistake-grid">
              <div class="mistake-card"><div class="m-title">Using the AUX channel</div>
                <p>The Muse outputs 5 channels: TP9, AF7, AF8, TP10 + Right AUX.
                AUX is ambient light/movement, not brain signal. The loader drops it automatically
                by matching against <code>cfg.channels</code>.</p></div>
              <div class="mistake-card"><div class="m-title">No pre-roll buffer</div>
                <p>Trial 0 starts at t=0. The 5-s baseline window would need t=&minus;5&ndash;0,
                which is before the recording. The loader adds a 5-s pre-roll so every trial
                has valid baseline data.</p></div>
              <div class="mistake-card"><div class="m-title">Scaling before LOSO split</div>
                <p>Fitting a StandardScaler on the full dataset before LOSO leaks test-subject
                statistics into training. Always fit the scaler <em>inside</em> the fold
                on training subjects only.</p></div>
              <div class="mistake-card"><div class="m-title">Running LOSO with 1 subject</div>
                <p>LOSO needs &ge;2 subjects. Single-subject runs work through preprocessing
                and feature extraction but skip LOSO with a clear warning message.</p></div>
              <div class="mistake-card"><div class="m-title">Treating this as 4-class</div>
                <p>Each quadrant is a <em>binary</em> classifier (target vs all others).
                Four separate models are trained. Chance level is <strong>50%</strong>,
                not 25%.</p></div>
              <div class="mistake-card"><div class="m-title">Ignoring the 95th-pct ROI rule</div>
                <p>The ROI window is not picked by highest raw accuracy alone &mdash;
                it must exceed the 95th percentile of scores across time. This filters
                out windows where accuracy is uniformly mediocre.</p></div>
            </div>""",
        ),
    )

    # ── S1: Data Model ───────────────────────────────────────────────────────
    s0 = subjects[0]
    n_total_samples = s0.eeg_data.shape[0]
    S_DATA = section_wrap("datamodel", "", "Data Model &amp; Schemas",
        subsec("SubjectData — the core container",
            "<p>Every XDF file becomes a <code>SubjectData</code> dataclass. "
            "All pipeline stages accept this object:</p>",
            schema_table(
                ["Field", "Type", "Shape / Value", "Meaning"],
                [
                    ["subject_id",    "str",        f'"{s0.subject_id}"',
                     "Unique participant label"],
                    ["eeg_data",      "np.ndarray", f"({n_total_samples}, 4) float64",
                     "Raw EEG in &micro;V — rows=time samples, cols=channels"],
                    ["sfreq",         "float",      f"{CFG.sfreq}",
                     "Sampling rate (Hz)"],
                    ["channel_names", "List[str]",  str(list(CFG.channels)),
                     "Channel labels in column order"],
                    ["timestamps",    "np.ndarray", f"({n_total_samples},) float64",
                     "LSL wall-clock timestamps"],
                    ["markers",       "DataFrame",  f"({NT} rows) timestamp, label",
                     "Trial onset events (auto-inferred if no marker stream)"],
                    ["ratings",       "DataFrame",  f"({NT} rows) see below",
                     "Self-assessment scores per trial"],
                ],
            ),
            callout("info", "Per-sample vs per-trial vs per-window",
                "<strong>Per-sample:</strong> every row of eeg_data — at 256 Hz "
                "one row every 3.9 ms.<br>"
                "<strong>Per-trial:</strong> one row in markers/ratings — "
                f"one video = one trial ({NT} trials total).<br>"
                "<strong>Per-window:</strong> one 5-second sub-epoch carved "
                "from the 60-s stimulus period."),
        ),
        subsec("Ratings table schema",
            schema_table(
                ["Column", "Type", "Range", "Description"],
                [
                    ["user_id",     "str",   "&mdash;",  "Participant identifier"],
                    ["video_id",    "int",   "0&ndash;N","Stimulus video index"],
                    ["valence",     "float", "1&ndash;9","Pleasantness (1=unpleasant, 9=pleasant)"],
                    ["arousal",     "float", "1&ndash;9","Activation (1=calm, 9=excited)"],
                    ["dominance",   "float", "1&ndash;9","Perceived control"],
                    ["liking",      "float", "1&ndash;9","How much the participant liked the video"],
                    ["familiarity", "float", "1&ndash;9","Whether they had seen the video before"],
                    ["quadrant",    "str",   "HVHA/LVHA/LVLA/HVLA",
                     "Computed: valence&ge;5&rarr;HV, arousal&ge;5&rarr;HA"],
                ],
            ),
            "<p><strong>Three example rows:</strong></p>",
            schema_table(
                ["user_id","video_id","valence","arousal","dominance","quadrant"],
                [
                    ["P001","3","7.2","3.1","6.0","HVLA"],
                    ["P001","7","2.8","7.9","2.1","LVHA"],
                    ["P002","3","6.8","2.9","5.5","HVLA"],
                ],
            ),
            callout("info", "Quadrant assignment rule",
                "valence &ge; 5 &rarr; High Valence (HV) &nbsp;|&nbsp; "
                "valence &lt; 5 &rarr; Low Valence (LV)<br>"
                "arousal &ge; 5 &rarr; High Arousal (HA) &nbsp;|&nbsp; "
                "arousal &lt; 5 &rarr; Low Arousal (LA)<br>"
                "Midpoint is 5.0. Ties go to High."),
        ),
        subsec("External labels table",
            "<p>Optional expert/crowdsourced annotations per video "
            "(not per participant):</p>",
            schema_table(
                ["Column","Type","Description"],
                [
                    ["video_id","int","Matches video_id in ratings"],
                    ["quadrant","str","Expert-assigned quadrant"],
                    ["valence_ext","float","Expert valence"],
                    ["arousal_ext","float","Expert arousal"],
                ],
            ),
        ),
        subsec("Russell Circumplex — the label space",
            "<p>Emotions are positioned on a 2D circle where X = valence and "
            "Y = arousal. Cutting at the midpoint creates four quadrants:</p>",
            fig_html(B["RUSSELL"],
                "Russell Circumplex Model: X-axis = valence, Y-axis = arousal. "
                "Each quadrant groups related emotions."),
            schema_table(
                ["Quadrant","V","A","Example emotions","Code label"],
                [
                    ["High Valence, High Arousal","&ge;5","&ge;5","Happy, excited","HVHA"],
                    ["Low Valence,  High Arousal","&lt;5","&ge;5","Angry, fearful","LVHA"],
                    ["Low Valence,  Low Arousal", "&lt;5","&lt;5","Sad, bored",    "LVLA"],
                    ["High Valence, Low Arousal", "&ge;5","&lt;5","Calm, relaxed", "HVLA"],
                ],
            ),
        ),
        have_now(
            "Understood: SubjectData shape is (N_samples, 4) for EEG; ratings is (N_trials, 8)",
            "Understood: quadrant labels are computed from valence &amp; arousal thresholded at 5",
            "Understood: this is 4 separate binary classifiers, not one 4-class model",
        ),
    )

    # ── S2: Step 1 — Load ────────────────────────────────────────────────────
    S_LOAD = section_wrap("step1", "1", "Load — XDF Files &amp; Markers",
        subsec("What is an XDF file?",
            "<p>XDF (Extensible Data Format) is the standard output of LabRecorder. "
            "It stores multiple simultaneous data streams — EEG, markers, accelerometer, "
            "PPG — each with its own timestamps. The file is hierarchical: "
            "stream 0 may be accelerometer, stream 3 may be EEG. "
            "The loader finds the EEG stream by looking for "
            "<code>type == \"EEG\"</code> regardless of order.</p>",
        ),
        subsec("Input",
            schema_table(
                ["Parameter","Type","Example"],
                [
                    ["xdf_path","str / Path","d:/NeuroStuff/.../sub-P001_eeg.xdf"],
                    ["ratings_path","str / Path (optional)","data/ratings.csv"],
                    ["cfg","NeuroSenseConfig","default config"],
                ],
            ),
        ),
        subsec("What the code does — step by step",
            code("""
                from neurosense.data_loader import load_neurosense_subject
                from neurosense.config import NeuroSenseConfig

                cfg = NeuroSenseConfig()
                subject = load_neurosense_subject(
                    xdf_path="sub-P001_eeg.xdf",
                    ratings_path="ratings.csv",   # or None
                    cfg=cfg,
                )

                # Inspect
                print(subject.eeg_data.shape)     # (N, 4) float64
                print(subject.sfreq)              # 256.0
                print(subject.channel_names)      # ['AF7','AF8','TP9','TP10']
                print(subject.markers.head(3))    # timestamp | label
                print(subject.ratings.head(3))    # video_id | valence | ...
            """),
            "<ol>"
            "<li><code>pyxdf.load_xdf()</code> reads all streams from disk.</li>"
            "<li>Find the stream with <code>type==\"EEG\"</code>.</li>"
            "<li>Parse channel labels from XDF header; keep only "
            "<code>cfg.channels</code> = [AF7, AF8, TP9, TP10], drop Right AUX.</li>"
            "<li>Look for a stream with <code>type==\"Markers\"</code>. "
            "If not found, call <code>_infer_trial_markers()</code> which divides "
            "the recording into equal 65-s blocks (5 s baseline + 60 s stimulus).</li>"
            "<li>Load ratings CSV, filter to this subject_id, compute quadrant column.</li>"
            "</ol>",
        ),
        subsec("Marker inference — what it does when markers are missing",
            callout("warn", "Most consumer Muse recordings have no marker stream",
                "LabRecorder records the headset but does not automatically mark "
                "when each video started. The loader detects this and infers "
                "trial onsets by dividing the recording length by the configured "
                "trial duration (65 s). This is an approximation &mdash; if you "
                "have a separate event log, convert it to a CSV and pass it via "
                "<code>ratings_path</code>."),
            code("""
                # Marker inference logic (simplified)
                trial_len = cfg.baseline_duration + cfg.stimulus_duration  # 5 + 60 = 65 s
                n_trials  = int((t_end - t_start) / trial_len)
                onsets    = [t_start + i * trial_len for i in range(n_trials)]
                # Returns DataFrame with columns: timestamp, label
            """),
        ),
        subsec("Graphs at this step",
            "<p><strong>Graph 1 — Raw EEG traces per channel (first 10 s).</strong> "
            "Verify that all 4 channels have signal and are on a reasonable "
            "amplitude scale (&minus;50 to +50 &micro;V typical for Muse).</p>",
            fig_html(B["RAWEEG"],
                "Raw EEG: 4 channels, first 10 s. Expect amplitude ~10-50 µV, "
                "no flat lines, no channel dominated by a single artefact."),
            "<p><strong>Graph 2 — Marker timeline.</strong> "
            "Verify marker positions cover the recording without overlap and "
            "that each baseline window fits before its stimulus.</p>",
            fig_html(B["MARKERS"],
                "Marker timeline: grey=baseline, teal=stimulus, yellow dashes=trial onset. "
                "Each teal block should be exactly 60 s wide."),
        ),
        subsec("Shape checks",
            code("""
                # After loading, always verify:
                assert subject.eeg_data.ndim == 2
                assert subject.eeg_data.shape[1] == 4,        "Expected 4 channels"
                assert subject.sfreq == 256.0,                 "Expected 256 Hz"
                assert len(subject.markers) >= 1,              "No trials found"
                assert "quadrant" in subject.ratings.columns,  "Missing quadrant column"

                # Amplitude sanity
                amp = subject.eeg_data.std()
                assert 0.5 < amp < 200,  f"Suspicious amplitude std: {amp:.1f} µV"
            """),
        ),
        callout("gate", "Verification gate",
            "<strong>Expected after Step 1:</strong><br>"
            "eeg_data shape: (N, 4) where N &gt; 10,000 for a 1-minute recording<br>"
            "sfreq = 256.0<br>"
            "channel_names = ['AF7','AF8','TP9','TP10']<br>"
            "markers rows: 1&ndash;40 (depends on recording length)<br>"
            "ratings rows = markers rows (or a warning is printed)"),
        have_now(
            f"<code>subject.eeg_data</code>: shape (N, 4), dtype float64, units µV",
            "<code>subject.markers</code>: DataFrame with trial onset timestamps",
            "<code>subject.ratings</code>: DataFrame with valence, arousal, quadrant columns",
            "Synthetic alternative: <code>make_synthetic_subject(cfg, n_trials=12)</code>",
        ),
    )

    # ── S3: Step 2 — Preprocess ──────────────────────────────────────────────
    S_PREPROC = section_wrap("step2", "2", "Preprocess — Filter, Denoise, Remove Artefacts",
        "<p>Raw EEG contains several types of noise that must be removed before any "
        "analysis. This step applies three sequential cleaning operations, each "
        "targeting a different noise source.</p>",

        subsec("Why clean EEG at all?",
            "<p>EEG signal from a Muse headset has three main noise sources:</p>"
            "<ol>"
            "<li><strong>DC drift / very-low-frequency noise</strong> &mdash; slow electrode "
            "baseline wander. Unrelated to brain activity. Removed by the high-pass cutoff.</li>"
            "<li><strong>High-frequency EMG / muscle artefacts</strong> &mdash; jaw clenching, "
            "head movements. Removed by the low-pass cutoff.</li>"
            "<li><strong>50/60 Hz mains line noise</strong> &mdash; electrical hum from power "
            "outlets coupling into the electrode cables. Removed by DSS.</li>"
            "</ol>",
        ),

        subsec("Sub-step 2a — FIR bandpass filter (1&ndash;45 Hz, Hamming window)",
            callout("math", "What is a FIR filter?",
                "<strong>FIR</strong> = Finite Impulse Response. It is a weighted moving average "
                "applied to the signal. The weights (filter coefficients) are designed so that "
                "frequencies outside the pass-band are attenuated to near zero.<br><br>"
                "<strong>Hamming window</strong>: the coefficients are tapered with a bell-shaped "
                "Hamming envelope so the filter does not create sharp transitions in the frequency "
                "domain (which would cause ringing artefacts in the time domain).<br><br>"
                "<strong>Why 1&ndash;45 Hz?</strong> "
                "EEG brain rhythms of interest (delta=1&ndash;4, theta=4&ndash;8, "
                "alpha=8&ndash;13, beta=13&ndash;30, gamma=30&ndash;45 Hz) all fit "
                "within this band. The Muse samples at 256 Hz, so the Nyquist limit is 128 Hz, "
                "but muscle artefacts above 45 Hz are removed here."),
            code("""
                from neurosense.preprocessing import apply_fir_filter

                raw_filtered = apply_fir_filter(raw_mne, cfg)
                # cfg.bandpass = (1.0, 45.0)
                # cfg.fir_window = "hamming"
                # Under the hood: raw.filter(l_freq=1.0, h_freq=45.0,
                #                            method='fir', fir_window='hamming')
            """),
            "<p><strong>PSD (Power Spectral Density) before vs after:</strong> "
            "The PSD is a plot of signal power vs frequency. After filtering you should see "
            "a cliff-edge at 1 Hz and 45 Hz &mdash; power outside these bounds drops to near "
            "zero. This is the primary sanity check for the filter.</p>",
            fig_html(B["PSD"],
                "PSD before (red) and after FIR filter (green). "
                "Power below 1 Hz (DC drift) and above 45 Hz (HF noise) should be eliminated. "
                "The y-axis is log scale — a factor of 100 in power is 20 dB."),
        ),

        subsec("Sub-step 2b — DSS line noise removal at 50 Hz",
            callout("math", "What is DSS?",
                "<strong>DSS</strong> = Denoising Source Separation. It is an extension "
                "of PCA that finds spatial filters (combinations of channels) which maximise "
                "the ratio of <em>periodic noise</em> power to total power. "
                "Once found, those filters are projected out of the data.<br><br>"
                "For 50 Hz (European mains frequency): the algorithm fits a periodic component "
                "at exactly 50 Hz and removes it. In North America you would use 60 Hz.<br><br>"
                "<strong>Fallback:</strong> if meegkit is not installed or DSS fails, "
                "the code issues a warning and skips this step without crashing. "
                "The FIR filter (which has a 45 Hz cutoff) already attenuates 50 Hz "
                "to some extent, so the fallback is not catastrophic."),
            code("""
                from neurosense.preprocessing import reduce_ringing_artifacts
                raw_clean = reduce_ringing_artifacts(raw_filtered, cfg)
                # Uses meegkit.dss.dss_line(..., fline=50.0)
                # Prints: "Power of components removed by DSS: 0.34"
                # If meegkit unavailable: UserWarning then raw returned unchanged
            """),
            callout("gate", "Verification gate — 50 Hz spike",
                "Plot the PSD after DSS. If you see a spike at exactly 50 Hz, "
                "DSS did not work (likely fell back). You can verify by checking the "
                "log output: look for <em>Power of components removed by DSS: X.XX</em>. "
                "If X &lt; 0.01, the removal was minimal."),
        ),

        subsec("Sub-step 2c — KNN outlier removal",
            callout("math", "What is KNN outlier detection?",
                "Each time-sample is a 4-dimensional point (one value per channel). "
                "For each sample we compute the distance to its k-th nearest neighbour "
                "in this 4D space (k=5 by default).<br><br>"
                "Samples that are far from all neighbours are likely artefacts "
                "(eye-blinks, electrode pops, movement). We flag the top 10% "
                "by k-distance (<code>contamination=0.10</code>) and replace "
                "them with <strong>linear interpolation</strong> from the surrounding "
                "clean samples.<br><br>"
                "Why interpolate instead of delete? Deleting would change the signal length "
                "and break the time-frequency structure that MiniRocket relies on."),
            code("""
                from neurosense.preprocessing import remove_ocular_outliers

                # data shape: (4, N_times) — channels first
                data_clean = remove_ocular_outliers(data, cfg)
                # cfg.knn_contamination = 0.10  (flag top 10% by k-distance)
                # cfg.knn_n_neighbors  = 5
            """),
            fig_html(B["KNN"],
                "Top: EEG channel with outlier samples flagged in red (10% of all samples). "
                "Bottom: k-distance curve — samples above the threshold (red dashed) "
                "are replaced by linear interpolation."),
            fig_html(B["PREPROC_TD"],
                "Time-domain comparison of the three sub-steps. "
                "Top: raw + simulated 50 Hz line noise. "
                "Middle: after FIR bandpass (HF noise gone). "
                "Bottom: after DSS (50 Hz hum removed)."),
        ),

        callout("gate", "Verification gate — preprocessing",
            "<strong>Check after this step:</strong><br>"
            "Signal amplitude std should still be in a plausible EEG range "
            "(roughly 1&ndash;50 µV). If it drops to near 0, the filter may have "
            "removed your signal. If it spikes to thousands, something went wrong.<br>"
            "No NaN/Inf values: <code>assert not np.isnan(data_clean).any()</code><br>"
            "Shape unchanged: still (4, N_times)"),
        have_now(
            "Filtered EEG: shape (4, N_times), bandpassed 1–45 Hz, line noise reduced",
            "PSD plot showing cliff at 1 Hz and 45 Hz",
            "Outlier samples interpolated (expect ~10% of samples touched)",
        ),
    )

    # ── S4: Step 3 — Epoching ────────────────────────────────────────────────
    S_EPOCH = section_wrap("step3", "3", "Epoching &amp; Sub-epoch Splitting",
        "<p>After cleaning, the continuous EEG is cut into labelled windows aligned "
        "to trial onsets. This converts the time-series into a dataset of fixed-length "
        "segments that can be fed to a classifier.</p>",

        subsec("Baseline epoch vs stimulus epoch",
            schema_table(
                ["Epoch type","Start (relative to onset)","Duration","Samples @ 256 Hz","Purpose"],
                [
                    ["Baseline",  f"&minus;{int(CFG.baseline_duration)} s",
                     f"{int(CFG.baseline_duration)} s", str(BL_SAMPS),
                     "Resting-state reference before each video"],
                    ["Stimulus",  "0 s (onset)", f"{int(CFG.stimulus_duration)} s",
                     str(ST_SAMPS), "Brain response during video watching"],
                ],
            ),
            callout("info", "Why extract a baseline?",
                "Brain activity varies between people and sessions. "
                "A 5-second rest period before each trial captures the "
                "participant's background brain state. You can subtract or "
                "normalise by the baseline to focus on stimulus-driven changes. "
                "In this pipeline, baseline sub-epochs also serve as the "
                "<em>negative class</em> in the ROI window selection step."),
        ),
        subsec("Sub-epoch splitting",
            f"<p>The {int(CFG.stimulus_duration)}-second stimulus epoch is cut into "
            f"non-overlapping {int(CFG.epoch_duration)}-second windows, giving "
            f"<strong>{N_SUB_ST} sub-epochs per trial</strong>. "
            f"Each sub-epoch is {SUB_SAMPS} samples long "
            f"({int(CFG.epoch_duration)} s &times; {int(CFG.sfreq)} Hz).</p>",
            code(f"""
                from neurosense.preprocessing import preprocess_subject

                preprocessed = preprocess_subject(subject, cfg)

                # Shape reference:
                # baseline_epochs:      ({'{'}n_trials{'}'}, 4, {BL_SAMPS})
                # stimulus_epochs:      ({'{'}n_trials{'}'}, 4, {ST_SAMPS})
                # baseline_sub_epochs:  ({'{'}n_trials{'}'}*{N_SUB_BL}, 4, {SUB_SAMPS})
                # stimulus_sub_epochs:  ({'{'}n_trials{'}'}*{N_SUB_ST}, 4, {SUB_SAMPS})

                p = preprocess_subject(subject, cfg)
                print(p.baseline_epochs.shape)      # (N_trials, 4, {BL_SAMPS})
                print(p.stimulus_epochs.shape)      # (N_trials, 4, {ST_SAMPS})
                print(p.stimulus_sub_epochs.shape)  # (N_trials*{N_SUB_ST}, 4, {SUB_SAMPS})
            """),
            fig_html(B["EPOCHS"],
                f"Trial structure: grey=baseline (5 s), coloured blocks=5-s sub-epoch "
                f"windows W1–W{N_SUB_ST} inside the 60-s stimulus. "
                "Yellow dashes = stimulus onset markers."),
        ),
        subsec("Shape verification",
            code(f"""
                n = preprocessed.n_trials
                assert preprocessed.baseline_epochs.shape      == (n, 4, {BL_SAMPS})
                assert preprocessed.stimulus_epochs.shape      == (n, 4, {ST_SAMPS})
                assert preprocessed.baseline_sub_epochs.shape  == (n * {N_SUB_BL}, 4, {SUB_SAMPS})
                assert preprocessed.stimulus_sub_epochs.shape  == (n * {N_SUB_ST}, 4, {SUB_SAMPS})
            """),
        ),
        callout("gate", "Verification gate — shapes",
            f"If you get fewer trials than expected (e.g. 7 instead of 8), "
            f"one trial is out of bounds &mdash; the baseline window extends before "
            f"the recording start. The pre-roll buffer in the loader prevents this "
            f"for synthetic data; for real data check that the first trial onset is "
            f"at least {int(CFG.baseline_duration)} s from the start of the recording."),
        have_now(
            f"baseline_sub_epochs: shape (n_trials&times;{N_SUB_BL}, 4, {SUB_SAMPS})",
            f"stimulus_sub_epochs: shape (n_trials&times;{N_SUB_ST}, 4, {SUB_SAMPS})",
            "Every sub-epoch is correctly aligned to a trial and labelled with its quadrant",
        ),
    )

    # ── S5: Step 4 — ROI ─────────────────────────────────────────────────────
    S_ROI = section_wrap("step4", "4", "ROI Time-Sliding Window Selection",
        "<p>Not all 60 seconds of a video produce equally discriminative EEG. "
        "The first few seconds may be orienting responses; the middle might be peak engagement. "
        "<strong>ROI selection</strong> finds the most informative 5-second window "
        "per quadrant using a sliding classifier.</p>",

        subsec("Candidate windows",
            schema_table(
                ["Window","Time range","Sub-epoch indices"],
                [
                    ["W1", "0–5 s",   "0"],
                    ["W2", "5–10 s",  "1"],
                    ["W3", "10–15 s", "2"],
                    ["W4", "15–20 s", "3"],
                    ["W5", "20–25 s", "4"],
                ],
            ),
        ),
        subsec("How the window is selected",
            "<ol>"
            "<li>For each of the 5 candidate windows, extract the sub-epochs "
            "of trials in the <em>target quadrant</em> (label=1) and "
            "match them with the same number of baseline sub-epochs (label=0).</li>"
            "<li>Train an <code>mne.decoding.SlidingEstimator</code> "
            "&mdash; a StandardScaler + RBF-SVM applied independently at every "
            "time point within the sub-epoch &mdash; using 3-fold cross-validation.</li>"
            "<li>Compute the 95th percentile of the resulting accuracy array.</li>"
            "<li>Filter to scores above that percentile; take their mean.</li>"
            "<li>Pick the window whose mean-above-95th-pct is highest.</li>"
            "</ol>",
            callout("math", "What is a SlidingEstimator?",
                "Imagine slicing the 1280-sample sub-epoch into 1280 individual time points. "
                "At each time point you fit a separate SVM on the 4-channel feature vector. "
                "This gives an accuracy value per time point, showing <em>when</em> within "
                "the 5-second window the two classes are most separable. "
                "The 95th-percentile threshold ensures we only count the best moments, "
                "not the average &mdash; this makes the selection more robust to noise."),
            callout("warn", "Double-dipping risk",
                "ROI selection uses the same subjects that later go into LOSO. "
                "This could bias the ROI choice toward the test subject. "
                "Mitigation: ROI selection runs per <em>participant separately</em> "
                "(each subject selects their own best window), so the LOSO test fold "
                "never directly influences the window chosen for training folds. "
                "<strong>Ideal practice</strong> (not yet implemented): select ROI "
                "on train folds only inside each LOSO split."),
        ),
        subsec("Outputs",
            code("""
                from neurosense.roi_timesliding import run_roi_for_all_quadrants

                roi_map = run_roi_for_all_quadrants(preprocessed, cfg)
                # Returns: dict[quadrant -> (window_tuple, acc_array)]

                for q, (window, acc) in roi_map.items():
                    print(f"{q}: best window {window[0]:.0f}-{window[1]:.0f}s  "
                          f"mean acc = {acc.mean():.3f}")
            """),
        ),
        subsec("Graphs",
            fig_html(B["ROI"],
                "Left: for each subject and time window, how many quadrants chose that window. "
                "Right: best mean accuracy achieved per window. "
                "Healthy variation across subjects is expected — different people respond "
                "differently at different times in the video."),
        ),
        callout("gate", "Verification gate — ROI always picks W1",
            "If every subject/quadrant picks window W1 (0&ndash;5 s), likely causes:<br>"
            "1. All windows have identical accuracy (e.g. if you have too few trials for "
            "reliable CV). With only 1&ndash;2 target-quadrant trials the SVM cannot train.<br>"
            "2. SlidingEstimator fell back to flat cross-validation. Check warnings.<br>"
            "<strong>Fix:</strong> increase <code>n_trials</code>, or lower "
            "<code>cv_inner</code> to 2 if sample counts are small."),
        have_now(
            "roi_map: dict mapping each quadrant to its best (start, end) time window",
            "Accuracy arrays per window, showing time-point-wise discriminability",
            "ROI heatmap showing window selection per subject",
        ),
    )

    # ── S6: Step 5 — MiniRocket ──────────────────────────────────────────────
    S_FEAT = section_wrap("step5", "5", "MiniRocket Feature Extraction",
        "<p>MiniRocket converts each 5-second sub-epoch into a fixed-length numeric feature "
        "vector that the SVM can learn from. It is the fastest accurate time-series "
        "classification transform available as of 2023.</p>",

        subsec("Intuition: what random kernels do",
            callout("math", "MiniRocket in plain English",
                "A <strong>kernel</strong> is a short pattern template (e.g. 9 samples long). "
                "It is slid along the time series and at each position we compute the "
                "dot product (similarity). This gives a 1-D activation sequence.<br><br>"
                "MiniRocket uses <strong>10,000 random kernels</strong> with random lengths, "
                "dilations, and channel combinations. For each kernel it computes one feature: "
                "the <strong>PPV (Proportion of Positive Values)</strong> &mdash; the fraction "
                "of time the activation was positive.<br><br>"
                "Because the kernels are random, there is no training for them. "
                "The SVM that follows learns which of the 10,000 PPV values are "
                "predictive of the emotion quadrant.<br><br>"
                "<strong>Why &times;2?</strong> Each kernel also computes a dilated version, "
                "doubling the feature count: 10,000 kernels &times; 2 = <strong>20,000 features</strong> "
                "per sub-epoch."),
        ),
        subsec("Input/output shapes",
            schema_table(
                ["Variable","Shape","Description"],
                [
                    ["sub_epochs (input)",
                     f"(n_instances, 4, {SUB_SAMPS})",
                     "Stack of sub-epochs from the ROI window"],
                    ["features (output)",
                     "(n_instances, 20,000)",
                     "One PPV feature per kernel per instance"],
                ],
            ),
            callout("info", "sktime panel format",
                "MiniRocketMultivariate expects the input as a 3-D array: "
                "<code>(n_instances, n_channels, n_timepoints)</code>. "
                "This is the standard <em>panel</em> format used by sktime. "
                "axis 0 = one sub-epoch, axis 1 = EEG channel, axis 2 = time sample."),
        ),
        subsec("Code",
            code(f"""
                from neurosense.features import extract_minirocket, build_feature_dataset

                # Single set of sub-epochs
                # sub_epochs shape: (n_instances, 4, {SUB_SAMPS})
                features, transformer = extract_minirocket(
                    sub_epochs, cfg, num_kernels=10_000
                )
                print(features.shape)  # (n_instances, 20000)

                # Full dataset: gather ROI sub-epochs for all subjects + quadrants
                feature_datasets = build_feature_dataset(
                    preprocessed_list, roi_maps, cfg, num_kernels=10_000
                )
                # Returns: dict[quadrant -> (X, y, subject_ids)]
                # X shape:           (total_instances, 20000)
                # y shape:           (total_instances,)  — 0=other, 1=target quadrant
                # subject_ids:       list[str] length total_instances
            """),
        ),
        subsec("Graphs",
            fig_html(B["FEAT"] if B.get("FEAT") else "",
                "Left: distribution of feature variance across 20,000 features — "
                "a healthy spread means the kernels are capturing diverse patterns. "
                "Centre: class balance — ~25% target, ~75% other (since each quadrant "
                "is roughly 1/4 of trials). Right: cumulative variance explained.") if B.get("FEAT") else "",
        ),
        callout("gate", "Verification gate",
            "features.shape[1] should be <code>num_kernels * 2</code> (e.g. 20,000 for 10K kernels).<br>"
            "No NaN/Inf: <code>assert np.isfinite(features).all()</code><br>"
            "Feature variance should not be uniformly near zero: "
            "<code>assert features.var(axis=0).mean() &gt; 1e-6</code>"),
        have_now(
            "Feature matrix X: shape (n_total_instances, 20000)",
            "Label vector y: shape (n_total_instances,) — binary 0/1",
            "subject_ids list: same length as X, used by LOSO to split train/test",
        ),
    )

    # ── S7: Step 6 — LOSO ────────────────────────────────────────────────────
    S_LOSO = section_wrap("step6", "6", "LOSO SVM Classification",
        "<p>With features extracted, we train an SVM and measure how well it generalises "
        "to <em>unseen participants</em> using Leave-One-Subject-Out (LOSO) "
        "cross-validation.</p>",

        subsec("What is LOSO and why does it matter?",
            callout("math", "Data leakage and why LOSO prevents it",
                "EEG signals are highly personal. If you train on sub-epochs from "
                "participant A and test on other sub-epochs from participant A, "
                "the model can memorise that person's brain patterns and report "
                "inflated accuracy &mdash; this is <strong>data leakage</strong>.<br><br>"
                "LOSO fixes this: in each fold, every sub-epoch from <em>one</em> participant "
                "is held out as the test set. The model is trained on all other participants. "
                "This measures how well the model works on a <strong>new person it has "
                "never seen</strong> &mdash; the real-world use case."),
        ),
        subsec("Outer loop — LOSO",
            code("""
                # Pseudocode of the outer loop
                for test_subject in all_subjects:
                    X_train = features from all OTHER subjects
                    y_train = labels  from all OTHER subjects
                    X_test  = features from test_subject
                    y_test  = labels  from test_subject

                    best_model = inner_search(X_train, y_train)
                    y_pred     = best_model.predict(X_test)
                    record accuracy, F1, confusion matrix
            """),
        ),
        subsec("Inner loop — RandomizedSearchCV",
            callout("math", "What is RandomizedSearchCV?",
                "Instead of trying all combinations of hyper-parameters "
                "(which would be thousands of SVM fits), RandomizedSearchCV "
                "randomly samples 50 combinations from a defined search space "
                "and evaluates each with 3-fold cross-validation on the training set. "
                "The best-performing combination is then used to train the final model.<br><br>"
                "<strong>What is tuned:</strong><br>"
                "&bull; <strong>SVM C</strong>: the penalty for misclassifying training points. "
                "Small C = soft margin (tolerates errors, less overfit). "
                "Large C = hard margin (fits training data tightly, may overfit). "
                "Sampled log-uniformly from 0.01 to 1000.<br>"
                "&bull; <strong>Scaler</strong>: MinMaxScaler, StandardScaler, or RobustScaler. "
                "Affects how features are normalised before the SVM sees them."),
            code("""
                from neurosense.modeling import loso_evaluate

                loso_results = loso_evaluate(feature_datasets, cfg)
                # cfg.n_random_search_iter = 50  (combinations tried)
                # cfg.cv_inner             = 3   (inner CV folds)

                print(loso_results.summary())
                # quadrant  accuracy_mean  accuracy_std  f1_mean  f1_std  n_folds
                print(loso_results.per_subject_df())
                # subject_id  quadrant  accuracy  f1_macro
            """),
        ),
        subsec("Binary one-vs-rest per quadrant",
            "<p>There are 4 quadrants but classification is not 4-class. "
            "Instead, 4 separate binary classifiers are trained:</p>"
            "<ul>"
            "<li><strong>HVHA classifier</strong>: HVHA trials (y=1) vs all other trials (y=0)</li>"
            "<li><strong>LVHA classifier</strong>: LVHA trials (y=1) vs all other trials (y=0)</li>"
            "<li><strong>LVLA classifier</strong>: LVLA trials (y=1) vs all other trials (y=0)</li>"
            "<li><strong>HVLA classifier</strong>: HVLA trials (y=1) vs all other trials (y=0)</li>"
            "</ul>"
            "<p>This matches the paper's approach and is more robust when classes "
            "are imbalanced (typically ~25% target, ~75% other).</p>",
            callout("warn", "Chance level is 50%, not 25%",
                "Because it is binary, a random classifier achieves 50% accuracy. "
                "Accuracy above 55% starts to be interesting. "
                "Accuracy above 65% is strong for EEG emotion recognition."),
        ),
        subsec("Outputs and how to read them",
            df_table(summary_df) if len(summary_df) else "<p><em>Need &ge;2 subjects to produce LOSO results.</em></p>",
            schema_table(
                ["Column","Meaning","Good range"],
                [
                    ["accuracy_mean","Mean accuracy across LOSO folds","0.55&ndash;0.80"],
                    ["accuracy_std","Std dev across folds — measures consistency","Low is better, &lt;0.15 good"],
                    ["f1_mean","Macro F1 — accounts for class imbalance","0.50&ndash;0.75"],
                    ["n_folds","Number of LOSO folds = number of subjects","Should equal N_subjects"],
                ],
            ),
        ),
        subsec("Graphs",
            fig_html(B["LOSO"] if B.get("LOSO") else "",
                "Left: accuracy heatmap per subject per quadrant — look for subjects "
                "who are consistently low (candidate for flagging). "
                "Right: mean accuracy vs chance level (0.50 dashed line).") if B.get("LOSO") else
            "<p><em>LOSO plots require &ge;2 subjects.</em></p>",
            fig_html(B["CM"] if B.get("CM") else "",
                "Aggregated confusion matrices (summed over all LOSO folds). "
                "Rows = actual class, columns = predicted class. "
                "Ideally the diagonal dominates. Off-diagonal cells show "
                "which classes are confused with each other.") if B.get("CM") else "",
        ),
        callout("gate", "Verification gate — accuracy too good",
            "If accuracy is >90% on EEG emotion data, something is likely wrong:<br>"
            "1. Data leakage: scaler or MiniRocket fit on all data before splitting.<br>"
            "2. Subject IDs not unique: all subjects have the same ID so LOSO cannot split.<br>"
            "3. Trivial class imbalance: if one quadrant has 95% of trials, "
            "predicting always 'other' gives 95% accuracy. Check class balance.<br>"
            "<strong>Sanity check:</strong> dummy baseline using "
            "<code>DummyClassifier(strategy='most_frequent')</code> "
            "should give accuracy &lt; 80% for reasonable class balance."),
        have_now(
            "LOSOResults with one FoldResult per (subject, quadrant) combination",
            "summary_df: mean &plusmn; std accuracy and F1 per quadrant",
            "per_subject_df: row per subject showing which they struggled on",
            "Confusion matrices showing class-level errors",
        ),
    )

    # ── S8: Step 7 — Stats ───────────────────────────────────────────────────
    S_STATS = section_wrap("step7", "7", "Statistics &amp; Participant Flags",

        subsec("Label reliability — do self-reports agree with experts?",
            "<p>Before trusting the classification results, verify that the emotion labels "
            "themselves are reliable. This is done by computing the Pearson correlation "
            "between participant self-assessments (SAM scale) and expert/external labels "
            "for the same videos.</p>",
            callout("math", "Pearson r in plain English",
                "Pearson r measures linear correlation: r=1 means perfect positive agreement, "
                "r=0 means no relationship, r=&minus;1 means perfect inverse. "
                "We also report p-value: p&lt;0.05 means the correlation is unlikely by chance. "
                "Before computing r, we <em>average self-assessments per video across participants</em> "
                "to get a stable per-video mean, then correlate with the expert rating. "
                "Averaging first prevents individual outliers from dominating."),
            code("""
                from neurosense.stats import compute_label_reliability

                reliability = compute_label_reliability(self_assessment_df, external_labels_df)
                print(reliability)
                # Label reliability:
                #   Valence : r=0.742  p=0.0031
                #   Arousal : r=0.681  p=0.0089
            """),
            callout("gate", "Verification gate — near-zero reliability",
                "If r &lt; 0.3 or p &gt; 0.05:<br>"
                "1. Check that video_id columns match between tables (join on video_id).<br>"
                "2. Check that rating scales are compatible (both 1&ndash;9, same direction).<br>"
                "3. Genuine low reliability: participants may have interpreted the scale "
                "differently from the experts. Consider excluding videos with high "
                "inter-rater disagreement."),
        ),

        subsec("Participant flags — quality control",
            "<p>Some participants give unreliable data. There are two independent signals:</p>"
            "<ol>"
            "<li><strong>Low rating variability</strong>: a participant who rated every video "
            "the same score (std near 0) was probably not paying attention or "
            "pressing buttons randomly.</li>"
            "<li><strong>Low model confidence</strong>: if the SVM assigned low decision "
            "probabilities to all trials for a participant, the EEG was not informative "
            "(possibly due to bad electrode contact or excessive movement).</li>"
            "</ol>",
            code("""
                from neurosense.stats import compute_participant_flags

                flags = compute_participant_flags(all_ratings_df, loso_results, cfg)
                # cfg.flag_percentile = 25.0  (flag if below 25th percentile)

                print(flags[['subject_id','valence_std','arousal_std',
                             'mean_decision_prob','flagged']])
            """),
            schema_table(
                ["Column","Meaning","Flag trigger"],
                [
                    ["valence_std",
                     "Std of valence ratings across all videos",
                     "&lt; 25th percentile"],
                    ["arousal_std",
                     "Std of arousal ratings across all videos",
                     "&lt; 25th percentile"],
                    ["dominance_std",
                     "Std of dominance ratings",
                     "&lt; 25th percentile"],
                    ["mean_decision_prob",
                     "Mean of max(SVM probability) across all test sub-epochs",
                     "&lt; 25th percentile"],
                    ["flagged",
                     "True if ANY of the above triggered",
                     "&mdash;"],
                ],
            ),
            fig_html(B["FLAGS"] if B.get("FLAGS") else "",
                "Bar chart: valence STD per subject (left) and mean SVM probability (right). "
                "Red bars are flagged participants. Dashed line is the 25th-percentile threshold.") if B.get("FLAGS") else "",
            callout("info", "What to do with flagged participants",
                "Flagging does not auto-exclude. It surfaces candidates for manual review. "
                "Typical actions: re-examine EEG traces for persistent artefacts, "
                "check if the participant reported difficulty following instructions, "
                "re-run the LOSO analysis with and without flagged subjects "
                "to check robustness."),
        ),

        have_now(
            "label_reliability.json: Pearson r/p for valence and arousal",
            "participant_flags.csv: per-subject rating STD, decision probability, flagged column",
            "Identified participants who may need manual review",
        ),
    )

    # ── S9: Outputs ──────────────────────────────────────────────────────────
    S_OUTPUTS = section_wrap("outputs", "", "Outputs &amp; How to Interpret Each File",
        schema_table(
            ["File","Columns / Keys","What decisions it supports"],
            [
                ["per_subject_metrics.csv",
                 "subject_id, quadrant, accuracy, f1_macro",
                 "Identify best/worst performing subjects and quadrants"],
                ["overall_summary.json",
                 "quadrant, accuracy_mean, accuracy_std, f1_mean, f1_std, n_folds",
                 "Report publishable results; compare across conditions"],
                ["confusion_matrices/cm_HVHA.png",
                 "(image) actual vs predicted, 2x2",
                 "Diagnose whether the model confuses target with other"],
                ["roi_windows.csv",
                 "subject, quadrant, window_start, window_end",
                 "Check if stimulus onset or middle is most informative"],
                ["participant_flags.csv",
                 "subject_id, valence_std, arousal_std, mean_decision_prob, flagged",
                 "Quality control: candidates for exclusion from analysis"],
                ["label_reliability.json",
                 "valence_r, valence_p, arousal_r, arousal_p",
                 "Validate that emotion labels are trustworthy before classification"],
            ],
        ),
        callout("ok", "If you only read two files",
            "<strong>1. overall_summary.json</strong> &mdash; tells you if classification "
            "worked at all (accuracy_mean &gt; 0.55 per quadrant = signal present).<br>"
            "<strong>2. participant_flags.csv</strong> &mdash; tells you if any participants "
            "are compromising the average (flagged=True)."),
    )

    # ── S10: Config ──────────────────────────────────────────────────────────
    S_CONFIG = section_wrap("config", "", "Configuration Deep-Dive",
        "<p><code>NeuroSenseConfig</code> is a <strong>frozen dataclass</strong> &mdash; "
        "all parameters in one place. Frozen means you cannot accidentally mutate it "
        "mid-pipeline. To customise, use <code>dataclasses.replace()</code>:</p>",
        code("""
            from neurosense.config import NeuroSenseConfig
            from dataclasses import replace

            cfg = NeuroSenseConfig()               # all defaults
            cfg_60hz = replace(cfg, sfreq=256.0)   # same, explicit
            cfg_fast  = replace(cfg,
                n_random_search_iter=10,           # faster tuning
                cv_inner=2,                        # fewer inner folds
            )
        """),
        schema_table(
            ["Parameter","Default","Plain-English meaning","When to change"],
            [
                ["sfreq","256.0","Sampling rate of the EEG device (Hz)",
                 "If your Muse reports 250 Hz or you resample"],
                ["channels","[AF7,AF8,TP9,TP10]","Which EEG channels to keep",
                 "If your device has different electrode names"],
                ["bandpass","(1.0, 45.0)","FIR filter pass-band in Hz",
                 "Lower to (0.5, 40) for slower rhythms; raise to (1, 100) if no muscle artefact"],
                ["knn_contamination","0.10","Fraction of samples treated as outliers",
                 "Lower to 0.05 if you trust data quality; raise to 0.20 for noisy sessions"],
                ["epoch_duration","5.0 s","Length of each sub-epoch",
                 "Never change without revalidating all downstream shapes"],
                ["stimulus_duration","60.0 s","Expected video length",
                 "Must match your actual video lengths"],
                ["roi_windows","(0-5),(5-10),…","Candidate ROI time windows",
                 "Add more windows if videos are longer; shift if response is known to be late"],
                ["n_random_search_iter","50","Hyper-parameter combinations to try",
                 "Lower to 10 for fast debugging; raise to 200 for final results"],
                ["cv_inner","3","Inner cross-validation folds",
                 "Lower to 2 if training set is very small (<20 instances per fold)"],
                ["flag_percentile","25.0","Threshold for participant flagging",
                 "Raise to 33 for stricter QC; lower to 10 for lenient QC"],
                ["random_seed","42","Controls all random operations",
                 "Change to test result stability across seeds"],
            ],
        ),
        subsec("Safe defaults vs research tweaks",
            schema_table(
                ["Scenario","Config change","Reason"],
                [
                    ["Fast debugging run",
                     "n_random_search_iter=5, cv_inner=2, num_kernels=100",
                     "Full run can take 30+ min; small values finish in seconds"],
                    ["60 Hz country (North America)",
                     "Not in config &mdash; edit preprocessing.py fline=60.0",
                     "DSS targets the mains frequency; 50 Hz is European standard"],
                    ["Longer videos (e.g. 3-min clips)",
                     "stimulus_duration=180.0, add roi_windows up to (55-60)",
                     "More sub-epochs per trial improves statistics"],
                ],
            ),
        ),
    )

    # ── S11: End-to-end recipes ──────────────────────────────────────────────
    S_RECIPES = section_wrap("recipes", "", "End-to-End Run Recipes",
        subsec("Recipe 1 — Synthetic smoke test",
            "<p>Run this first to verify the installation works. No real data needed.</p>",
            code("""
                python -m neurosense.run \\
                    --use-synthetic \\
                    --n-subjects 4 \\
                    --n-trials  12 \\
                    --output    results/
            """),
            term("""
                =======================================================
                  NeuroSense EEG Pipeline
                =======================================================
                  Mode: SYNTHETIC (4 subjects, 12 trials each)
                  Loaded 4 subject(s)
                  Preprocessing...
                  Preprocessed 4 subject(s)
                  Running ROI window selection...
                  Extracting MiniRocket features...
                  Running LOSO evaluation...

                LOSO CLASSIFICATION RESULTS
                ============================================================
                 quadrant  accuracy_mean  accuracy_std  f1_mean  n_folds
                     HVHA          0.583         0.144    0.510        4
                     HVLA          0.542         0.128    0.482        4
                     LVHA          0.567         0.161    0.501        4
                     LVLA          0.558         0.139    0.496        4

                Results saved to: results/
            """),
            callout("ok", "What success looks like",
                "All 4 quadrants complete. Accuracy near 0.50&ndash;0.60 is correct for "
                "random synthetic data (no real EEG patterns, so slightly above chance). "
                "Files appear in results/."),
        ),
        subsec("Recipe 2 — Single real XDF file",
            "<p>Works with one recording. LOSO is skipped (needs &ge;2 subjects) "
            "but all preprocessing and feature extraction runs.</p>",
            code("""
                python -m neurosense.run \\
                    --xdf-file "d:/NeuroStuff/.../sub-P001_eeg.xdf" \\
                    --output   results/
            """),
        ),
        subsec("Recipe 3 — Full multi-subject dataset",
            "<p>Point to a folder; the loader recursively finds all .xdf files.</p>",
            code("""
                # Required folder structure:
                # data/
                #   sub-P001/ses-S001/eeg/*.xdf
                #   sub-P002/ses-S001/eeg/*.xdf
                # ratings.csv (user_id, video_id, valence, arousal, ...)
                # labels.csv  (video_id, quadrant, valence_ext, arousal_ext)

                python -m neurosense.run \\
                    --data-root data/ \\
                    --ratings   ratings.csv \\
                    --labels    labels.csv \\
                    --output    results/
            """),
        ),
        subsec("Minimal debugging workflow — stop after each step",
            code("""
                from neurosense.config import NeuroSenseConfig
                from neurosense.data_loader import make_synthetic_subject
                from neurosense.preprocessing import preprocess_subject
                from neurosense.roi_timesliding import run_roi_for_all_quadrants
                from neurosense.features import build_feature_dataset
                from neurosense.modeling import loso_evaluate

                cfg = NeuroSenseConfig()

                # --- Step 1: Load ---
                subject = make_synthetic_subject(cfg, n_trials=12, seed=0)
                print("EEG shape:", subject.eeg_data.shape)
                print("Ratings:\\n", subject.ratings.head(3))
                input("Step 1 OK? Press Enter...")

                # --- Step 2-3: Preprocess ---
                prep = preprocess_subject(subject, cfg)
                print("Stim sub-epochs:", prep.stimulus_sub_epochs.shape)
                input("Step 2-3 OK? Press Enter...")

                # --- Step 4: ROI ---
                roi = run_roi_for_all_quadrants(prep, cfg)
                for q,(w,a) in roi.items():
                    print(f"  {q}: window={w}, mean_acc={a.mean():.3f}")
                input("Step 4 OK? Press Enter...")

                # --- Step 5: Features ---
                fd = build_feature_dataset([prep],[roi],cfg,num_kernels=500)
                for q,(X,y,_) in fd.items():
                    print(f"  {q}: X={X.shape}, y={y.shape}")
                input("Step 5 OK? Press Enter...")

                # --- Step 6: LOSO (needs ≥2 subjects) ---
                # Make 3 subjects total
                subjects = [make_synthetic_subject(cfg, n_trials=12, seed=i) for i in range(3)]
                preps    = [preprocess_subject(s, cfg) for s in subjects]
                rois     = [run_roi_for_all_quadrants(p, cfg) for p in preps]
                fd3      = build_feature_dataset(preps, rois, cfg, num_kernels=500)
                results  = loso_evaluate(fd3, cfg)
                print(results.summary())
            """),
        ),
    )

    # ── S12: Appendix ────────────────────────────────────────────────────────
    S_APPENDIX = section_wrap("appendix", "", "Appendix — Method Mapping to the Paper",
        subsec("Paper concept &rarr; code function &rarr; output artifact",
            schema_table(
                ["Paper concept","Module","Function","Output"],
                [
                    ["EEG acquisition (Muse)",
                     "data_loader.py","load_neurosense_subject()","SubjectData"],
                    ["Participant affect ratings (SAM)",
                     "data_loader.py","_load_ratings()","ratings DataFrame"],
                    ["Russell Circumplex mapping",
                     "config.py","assign_quadrant()","quadrant column in ratings"],
                    ["Bandpass filter 1&ndash;45 Hz",
                     "preprocessing.py","apply_fir_filter()","filtered MNE Raw"],
                    ["Line noise reduction",
                     "preprocessing.py","reduce_ringing_artifacts()","cleaned Raw"],
                    ["Artefact removal",
                     "preprocessing.py","remove_ocular_outliers()","cleaned ndarray"],
                    ["Baseline &amp; stimulus epoching",
                     "preprocessing.py","build_epochs()","baseline_epochs, stimulus_epochs"],
                    ["5-s sub-epoch segmentation",
                     "preprocessing.py","split_into_sub_epochs()","sub_epochs ndarray"],
                    ["ROI time-sliding window",
                     "roi_timesliding.py","select_roi_window()","best (t0,t1) window"],
                    ["MiniRocket feature extraction",
                     "features.py","extract_minirocket()","(n_instances, 20000) array"],
                    ["LOSO cross-validation",
                     "modeling.py","loso_evaluate()","LOSOResults"],
                    ["Hyper-parameter optimisation",
                     "modeling.py","RandomizedSearchCV (inner)","best_params per fold"],
                    ["Label reliability",
                     "stats.py","compute_label_reliability()","label_reliability.json"],
                    ["Participant quality flags",
                     "stats.py","compute_participant_flags()","participant_flags.csv"],
                ],
            ),
        ),
        subsec("Implementation notes vs typical EEG pipelines",
            schema_table(
                ["Aspect","This implementation","Typical alternative"],
                [
                    ["Reference electrode",
                     "Average reference (MNE default)",
                     "Linked mastoids or Cz reference"],
                    ["ICA for artefact removal",
                     "Not implemented — uses KNN interpolation instead",
                     "ICA is standard for clinical EEG but requires &ge;8 channels"],
                    ["Epoch rejection threshold",
                     "KNN distance (implicit)",
                     "Peak-to-peak amplitude threshold (e.g. &gt;100 µV reject)"],
                    ["Feature extraction",
                     "MiniRocket (time-domain random kernels)",
                     "Hand-crafted band-power features (alpha, theta, etc.)"],
                    ["ROI selection timing",
                     "Per participant, uses full data (potential minor leakage)",
                     "ROI selection inside LOSO fold on training data only"],
                    ["Classification",
                     "Binary one-vs-rest per quadrant, RBF-SVM",
                     "Multi-class SVM, Random Forest, or deep learning"],
                ],
            ),
            callout("warn", "Aspects not specified in the current report",
                "1. Whether the paper uses absolute or relative power features &mdash; "
                "this implementation uses raw time-domain MiniRocket features.<br>"
                "2. Whether baseline normalisation is applied before feature extraction "
                "&mdash; currently it is not; sub-epoch features are extracted raw.<br>"
                "3. Exact DSS parameters (number of components removed) &mdash; "
                "the implementation uses meegkit defaults."),
        ),
    )

    return [S_MENTAL, S_DATA, S_LOAD, S_PREPROC, S_EPOCH,
            S_ROI, S_FEAT, S_LOSO, S_STATS, S_OUTPUTS,
            S_CONFIG, S_RECIPES, S_APPENDIX]
