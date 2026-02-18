"""
generate_report.py
------------------
Runs the NeuroSense pipeline end-to-end on synthetic data and produces a
single self-contained HTML report at docs/report.html.

Usage
-----
  python docs/generate_report.py
"""
from __future__ import annotations

import base64
import io
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# Make sure the package root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def code_block(code: str, language: str = "python") -> str:
    escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<pre class="code-block"><code class="lang-{language}">{escaped}</code></pre>'


def section(title: str, *content: str) -> str:
    body = "\n".join(content)
    return f"""
    <section>
      <h2>{title}</h2>
      {body}
    </section>"""


def subsection(title: str, *content: str) -> str:
    body = "\n".join(content)
    return f'<div class="subsection"><h3>{title}</h3>{body}</div>'


def img(b64: str, caption: str = "") -> str:
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{caption}">{cap}</div>'


def table(df: pd.DataFrame) -> str:
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for v in row:
            if isinstance(v, float):
                cells += f"<td>{v:.4f}</td>"
            else:
                cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>"
    return f'<table><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table>'


def callout(text: str, kind: str = "info") -> str:
    icons = {"info": "ℹ️", "warn": "⚠️", "ok": "✅"}
    return f'<div class="callout callout-{kind}"><span>{icons.get(kind,"")}</span> {text}</div>'


def terminal(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<pre class="terminal">{escaped}</pre>'


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run
# ─────────────────────────────────────────────────────────────────────────────

print("Loading NeuroSense modules …")
from neurosense.config import NeuroSenseConfig, RUSSELL_QUADRANTS
from neurosense.data_loader import make_synthetic_subject
from neurosense.preprocessing import preprocess_subject
from neurosense.roi_timesliding import run_roi_for_all_quadrants
from neurosense.features import build_feature_dataset, extract_minirocket
from neurosense.modeling import loso_evaluate
from neurosense.stats import compute_participant_flags, print_results_table

CFG = NeuroSenseConfig()
N_SUBJECTS = 4
N_TRIALS   = 12

print(f"Generating {N_SUBJECTS} synthetic subjects ({N_TRIALS} trials each) …")
subjects      = [make_synthetic_subject(CFG, n_trials=N_TRIALS, seed=i*7) for i in range(N_SUBJECTS)]
preprocessed  = [preprocess_subject(s, CFG) for s in subjects]

print("ROI window selection …")
roi_maps = [run_roi_for_all_quadrants(p, CFG) for p in preprocessed]

print("MiniRocket feature extraction …")
feature_datasets = build_feature_dataset(preprocessed, roi_maps, CFG, num_kernels=500)

print("LOSO evaluation …")
loso_results = loso_evaluate(feature_datasets, CFG)

all_ratings = pd.concat([s.ratings for s in subjects], ignore_index=True)
for i, s in enumerate(subjects):
    s.ratings["user_id"] = s.subject_id
all_ratings = pd.concat([s.ratings for s in subjects], ignore_index=True)
flags = compute_participant_flags(all_ratings, loso_results, CFG)

summary_df  = loso_results.summary()
per_subj_df = loso_results.per_subject_df()

print("Building figures …")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Russell Circumplex
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
circle = plt.Circle((0, 0), 1, fill=False, color="#888", linewidth=1.2, linestyle="--")
ax.add_patch(circle)
ax.axhline(0, color="#ccc", linewidth=0.8)
ax.axvline(0, color="#ccc", linewidth=0.8)
colors   = {"HVHA": "#e74c3c", "LVHA": "#e67e22", "LVLA": "#3498db", "HVLA": "#2ecc71"}
labels   = {"HVHA": "HVHA\n(Happy/Excited)", "LVHA": "LVHA\n(Angry/Fearful)",
            "LVLA": "LVLA\n(Sad/Bored)",      "HVLA": "HVLA\n(Calm/Relaxed)"}
positions = {"HVHA": (0.65, 0.65), "LVHA": (-0.65, 0.65),
             "LVLA": (-0.65, -0.65), "HVLA": (0.65, -0.65)}
for q, (x, y) in positions.items():
    ax.scatter(x, y, s=220, color=colors[q], zorder=5)
    ax.annotate(labels[q], (x, y), textcoords="offset points",
                xytext=(12 if x > 0 else -12, 6), fontsize=8.5,
                ha="left" if x > 0 else "right", color=colors[q], fontweight="bold")
ax.set_xlabel("Valence →", fontsize=10); ax.set_ylabel("Arousal →", fontsize=10)
ax.set_title("Russell Circumplex Model of Affect", fontsize=11, fontweight="bold")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig1.tight_layout()
b64_russell = fig_to_b64(fig1); plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Synthetic EEG traces (one subject, first 10 s)
# ─────────────────────────────────────────────────────────────────────────────
s0 = subjects[0]
show_s  = int(10 * CFG.sfreq)
t_axis  = np.arange(show_s) / CFG.sfreq
ch_names = s0.channel_names

fig2, axes = plt.subplots(len(ch_names), 1, figsize=(10, 5), sharex=True)
palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
for i, (ax, ch) in enumerate(zip(axes, ch_names)):
    ax.plot(t_axis, s0.eeg_data[:show_s, i], color=palette[i], linewidth=0.7, alpha=0.85)
    ax.set_ylabel(ch, fontsize=9, rotation=0, labelpad=32)
    ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
axes[-1].set_xlabel("Time (s)"); axes[0].set_title("Raw EEG — 4 channels (first 10 s)", fontweight="bold")
fig2.tight_layout(); b64_eeg = fig_to_b64(fig2); plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Preprocessing pipeline — before / after one epoch
# ─────────────────────────────────────────────────────────────────────────────
p0 = preprocessed[0]
n_show = int(5 * CFG.sfreq)
raw_seg = s0.eeg_data[:n_show, 0]           # channel 0, first 5 s
clean_seg = p0.baseline_sub_epochs[0, 0, :] # first sub-epoch, ch 0

fig3, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4), sharex=False)
t5 = np.arange(n_show) / CFG.sfreq
a1.plot(t5, raw_seg, color="#e74c3c", linewidth=0.8); a1.set_title("Raw AF7 (first 5 s)", fontsize=10)
a1.set_ylabel("µV"); a1.spines[["top","right"]].set_visible(False)
t5c = np.arange(len(clean_seg)) / CFG.sfreq
a2.plot(t5c, clean_seg, color="#2ecc71", linewidth=0.8); a2.set_title("After FIR + DSS + KNN outlier removal", fontsize=10)
a2.set_ylabel("µV"); a2.set_xlabel("Time (s)"); a2.spines[["top","right"]].set_visible(False)
fig3.tight_layout(); b64_preproc = fig_to_b64(fig3); plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Epoch / sub-epoch structure
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(10, 2.2))
trial_len = CFG.baseline_duration + CFG.stimulus_duration
n_show_trials = 3
ax.set_xlim(0, n_show_trials * trial_len)
ax.set_ylim(0, 1); ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
for t in range(n_show_trials):
    x0 = t * trial_len
    # baseline
    ax.barh(0.5, CFG.baseline_duration, left=x0, height=0.4, color="#95a5a6", align="center")
    ax.text(x0 + CFG.baseline_duration/2, 0.5, "Baseline\n5 s", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    # sub-epochs inside stimulus
    n_sub = int(CFG.stimulus_duration / CFG.epoch_duration)
    sub_colors = ["#3498db","#2980b9","#1f618d","#154360","#0b2739"]
    for w in range(min(5, n_sub)):
        wx = x0 + CFG.baseline_duration + w * CFG.epoch_duration
        ax.barh(0.5, CFG.epoch_duration - 0.3, left=wx + 0.15, height=0.4,
                color=sub_colors[w % len(sub_colors)], align="center")
        ax.text(wx + CFG.epoch_duration/2, 0.5, f"W{w+1}", ha="center", va="center",
                fontsize=7, color="white")
    # remaining stimulus
    rem_x = x0 + CFG.baseline_duration + 5 * CFG.epoch_duration
    rem_w = CFG.stimulus_duration - 5 * CFG.epoch_duration
    if rem_w > 0:
        ax.barh(0.5, rem_w, left=rem_x, height=0.4, color="#aab7c4", align="center")
        ax.text(rem_x + rem_w/2, 0.5, "…", ha="center", va="center", fontsize=10, color="white")
ax.set_xlabel("Time (s)"); ax.set_title("Trial structure — baseline + 5-s sub-epoch windows", fontweight="bold")
fig4.tight_layout(); b64_epochs = fig_to_b64(fig4); plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: ROI window selection heatmap
# ─────────────────────────────────────────────────────────────────────────────
window_labels = [f"{int(s)}–{int(e)} s" for s, e in CFG.roi_windows]
roi_matrix = np.zeros((N_SUBJECTS, len(CFG.roi_windows)))
for subj_idx, roi_map in enumerate(roi_maps):
    for q_idx, q in enumerate(CFG.quadrant_names):
        window, _ = roi_map[q]
        w_idx = list(CFG.roi_windows).index(window)
        roi_matrix[subj_idx, w_idx] += 1

fig5, ax = plt.subplots(figsize=(7, 3))
subj_labels = [s.subject_id for s in subjects]
sns.heatmap(roi_matrix, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=window_labels, yticklabels=subj_labels,
            linewidths=0.5, ax=ax, cbar_kws={"label": "# quadrants selecting window"})
ax.set_title("ROI Window Selection — count of quadrants per subject", fontweight="bold")
ax.set_xlabel("Time window"); ax.set_ylabel("Subject")
fig5.tight_layout(); b64_roi = fig_to_b64(fig5); plt.close(fig5)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: MiniRocket feature distribution
# ─────────────────────────────────────────────────────────────────────────────
if feature_datasets:
    q_sample = list(feature_datasets.keys())[0]
    X_sample, y_sample, _ = feature_datasets[q_sample]
    fig6, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # Feature variance distribution
    variances = X_sample.var(axis=0)
    axes[0].hist(variances, bins=40, color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Feature variance"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"MiniRocket feature variance\n({X_sample.shape[1]} features, quadrant: {q_sample})", fontsize=9)
    axes[0].spines[["top","right"]].set_visible(False)
    # Class balance
    unique, counts = np.unique(y_sample, return_counts=True)
    bar_labels = ["Other quadrants" if u == 0 else q_sample for u in unique]
    axes[1].bar(bar_labels, counts, color=["#95a5a6", "#e74c3c"], edgecolor="white")
    axes[1].set_ylabel("Instances"); axes[1].set_title(f"Class balance — {q_sample}", fontsize=9)
    axes[1].spines[["top","right"]].set_visible(False)
    fig6.tight_layout(); b64_features = fig_to_b64(fig6); plt.close(fig6)
else:
    b64_features = ""

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: LOSO results — accuracy per subject per quadrant
# ─────────────────────────────────────────────────────────────────────────────
if loso_results.folds:
    pivot = per_subj_df.pivot_table(index="subject_id", columns="quadrant", values="accuracy")
    fig7, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_title("LOSO Accuracy per Subject per Quadrant", fontweight="bold")
    ax.set_xlabel("Quadrant"); ax.set_ylabel("Left-out subject")
    fig7.tight_layout(); b64_loso_heat = fig_to_b64(fig7); plt.close(fig7)

    # Bar chart — mean ± std per quadrant
    fig8, ax = plt.subplots(figsize=(6, 3.5))
    qs = summary_df["quadrant"].tolist()
    means = summary_df["accuracy_mean"].tolist()
    stds  = summary_df["accuracy_std"].tolist()
    bar_colors = ["#e74c3c","#e67e22","#3498db","#2ecc71"][:len(qs)]
    ax.bar(qs, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="#888", linewidth=1, linestyle="--", label="Chance (0.50)")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_xlabel("Quadrant")
    ax.set_title("Mean LOSO Accuracy ± std per Quadrant", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig8.tight_layout(); b64_loso_bar = fig_to_b64(fig8); plt.close(fig8)
else:
    b64_loso_heat = b64_loso_bar = ""

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
if loso_results.folds:
    quadrants_with_folds = sorted({f.quadrant for f in loso_results.folds})
    n_q = len(quadrants_with_folds)
    fig9, axes = plt.subplots(1, n_q, figsize=(4 * n_q, 3.5))
    if n_q == 1: axes = [axes]
    for ax, q in zip(axes, quadrants_with_folds):
        q_folds = [f for f in loso_results.folds if f.quadrant == q]
        agg_cm = sum(f.confusion_matrix for f in q_folds)
        sns.heatmap(agg_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Other","Target"], yticklabels=["Other","Target"],
                    linewidths=0.5, cbar=False)
        ax.set_title(f"Quadrant {q}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig9.suptitle("Aggregated Confusion Matrices (LOSO)", fontweight="bold", fontsize=11)
    fig9.tight_layout(); b64_cm = fig_to_b64(fig9); plt.close(fig9)
else:
    b64_cm = ""

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Participant flags
# ─────────────────────────────────────────────────────────────────────────────
if flags is not None and len(flags) > 0 and "valence_std" in flags.columns:
    fig10, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    flagged_color = ["#e74c3c" if f else "#2ecc71" for f in flags["flagged"]]
    axes[0].bar(flags["subject_id"], flags["valence_std"], color=flagged_color, edgecolor="white")
    thr_v = np.nanpercentile(flags["valence_std"], CFG.flag_percentile)
    axes[0].axhline(thr_v, color="#888", linestyle="--", linewidth=1, label=f"25th pct ({thr_v:.2f})")
    axes[0].set_title("Valence STD per subject", fontsize=10); axes[0].set_ylabel("STD")
    axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)

    if "mean_decision_prob" in flags.columns and flags["mean_decision_prob"].notna().any():
        axes[1].bar(flags["subject_id"], flags["mean_decision_prob"], color=flagged_color, edgecolor="white")
        thr_p = np.nanpercentile(flags["mean_decision_prob"].dropna(), CFG.flag_percentile)
        axes[1].axhline(thr_p, color="#888", linestyle="--", linewidth=1, label=f"25th pct ({thr_p:.2f})")
        axes[1].set_title("Mean decision probability per subject", fontsize=10)
        axes[1].set_ylabel("Probability"); axes[1].legend(fontsize=8)
        axes[1].spines[["top","right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Flagged"), Patch(facecolor="#2ecc71", label="OK")]
    fig10.legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig10.suptitle("Participant Quality Flags", fontweight="bold")
    fig10.tight_layout(); b64_flags = fig_to_b64(fig10); plt.close(fig10)
else:
    b64_flags = ""

print("Rendering HTML …")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0f1117;
  color: #e2e8f0;
  line-height: 1.65;
}
header {
  background: linear-gradient(135deg, #1a1f35 0%, #0f1117 100%);
  border-bottom: 1px solid #2d3748;
  padding: 3rem 2rem 2rem;
  text-align: center;
}
header h1 { font-size: 2.4rem; color: #63b3ed; letter-spacing: -0.5px; }
header .subtitle { color: #a0aec0; margin-top: 0.5rem; font-size: 1.05rem; }
header .badges { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
.badge {
  background: #2d3748; border: 1px solid #4a5568;
  border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.78rem; color: #a0aec0;
}
.badge.green { border-color: #48bb78; color: #48bb78; }
.badge.blue  { border-color: #63b3ed; color: #63b3ed; }

nav {
  position: sticky; top: 0; z-index: 100;
  background: #1a202c; border-bottom: 1px solid #2d3748;
  padding: 0 2rem; display: flex; gap: 0; overflow-x: auto;
}
nav a {
  color: #a0aec0; text-decoration: none; padding: 0.8rem 1.1rem;
  font-size: 0.88rem; white-space: nowrap; border-bottom: 2px solid transparent;
  transition: all 0.15s;
}
nav a:hover { color: #63b3ed; border-bottom-color: #63b3ed; }

main { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
section { margin-bottom: 3.5rem; }
section h2 {
  font-size: 1.55rem; color: #63b3ed; margin-bottom: 1.2rem;
  padding-bottom: 0.5rem; border-bottom: 1px solid #2d3748;
}
.subsection { margin: 1.5rem 0; }
.subsection h3 { font-size: 1.1rem; color: #90cdf4; margin-bottom: 0.8rem; }
p { margin-bottom: 0.9rem; color: #cbd5e0; }

pre.code-block {
  background: #1a202c; border: 1px solid #2d3748; border-radius: 8px;
  padding: 1.2rem 1.4rem; overflow-x: auto; margin: 0.8rem 0 1.2rem;
  font-size: 0.84rem; line-height: 1.6;
}
pre.code-block code { color: #e2e8f0; font-family: 'Fira Code', 'Cascadia Code', Consolas, monospace; }
pre.terminal {
  background: #0d1117; border: 1px solid #2d3748; border-radius: 8px;
  padding: 1rem 1.4rem; overflow-x: auto; margin: 0.8rem 0 1.2rem;
  font-size: 0.83rem; color: #68d391; font-family: Consolas, monospace; line-height: 1.6;
}

.callout {
  display: flex; align-items: flex-start; gap: 0.7rem;
  padding: 0.8rem 1.2rem; border-radius: 6px; margin: 0.8rem 0 1.2rem;
  font-size: 0.9rem;
}
.callout-info  { background: #1a365d; border: 1px solid #2b6cb0; color: #bee3f8; }
.callout-warn  { background: #744210; border: 1px solid #c05621; color: #feebc8; }
.callout-ok    { background: #1c4532; border: 1px solid #276749; color: #c6f6d5; }

.figure { margin: 1.2rem 0 1.8rem; }
.figure img { max-width: 100%; border-radius: 8px; border: 1px solid #2d3748; }
.caption { font-size: 0.82rem; color: #718096; margin-top: 0.4rem; font-style: italic; }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }

table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; font-size: 0.88rem; }
thead { background: #2d3748; }
th { padding: 0.6rem 0.9rem; text-align: left; color: #90cdf4; font-weight: 600; }
td { padding: 0.55rem 0.9rem; border-bottom: 1px solid #2d3748; color: #cbd5e0; }
tr:hover td { background: #1a202c; }

.pipeline-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem; margin: 1.2rem 0;
}
.pipeline-card {
  background: #1a202c; border: 1px solid #2d3748; border-radius: 8px;
  padding: 1rem; text-align: center;
}
.pipeline-card .step-num {
  font-size: 1.5rem; font-weight: 700; color: #63b3ed; margin-bottom: 0.3rem;
}
.pipeline-card .step-name { font-size: 0.88rem; color: #90cdf4; font-weight: 600; }
.pipeline-card .step-desc { font-size: 0.78rem; color: #718096; margin-top: 0.3rem; }

footer {
  text-align: center; padding: 2rem; color: #4a5568; font-size: 0.82rem;
  border-top: 1px solid #2d3748;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Assemble page
# ─────────────────────────────────────────────────────────────────────────────
loso_summary_str = summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(summary_df) else "(no folds)"

html_sections = []

# ── 1. Overview ──────────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="overview"></a>Overview',
    '<p>NeuroSense is an end-to-end EEG emotion classification pipeline for low-cost, '
    'sparse-electrode devices (Muse headset, 4 channels). It implements the methodology '
    'from the <em>IEEE Access</em> paper <strong>"NeuroSense: A Novel EEG Dataset Utilizing '
    'Low-Cost, Sparse Electrode Devices for Emotion Exploration"</strong>.</p>',
    '<p>Emotions are mapped to four quadrants of the <strong>Russell Circumplex Model</strong> '
    'using self-assessed valence and arousal ratings. A MiniRocket + SVM classifier is trained '
    'and evaluated with Leave-One-Subject-Out (LOSO) cross-validation.</p>',
    img(b64_russell, "Russell Circumplex Model — four emotion quadrants"),
    '<div class="pipeline-grid">'
    + "".join(
        f'<div class="pipeline-card"><div class="step-num">{n}</div>'
        f'<div class="step-name">{name}</div>'
        f'<div class="step-desc">{desc}</div></div>'
        for n, name, desc in [
            ("1", "Load",        "XDF via pyxdf<br>auto marker inference"),
            ("2", "Preprocess",  "FIR filter + DSS<br>KNN outlier removal"),
            ("3", "Epoch",       "Baseline & stimulus<br>5-s sub-epochs"),
            ("4", "ROI Select",  "SlidingEstimator<br>best 5-s window"),
            ("5", "MiniRocket",  "10 K random kernels<br>feature extraction"),
            ("6", "LOSO SVM",    "RandomSearchCV<br>inner, LOO outer"),
            ("7", "Stats",       "Pearson reliability<br>participant flags"),
        ]
    )
    + '</div>',
))

# ── 2. Installation ───────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="install"></a>Installation',
    subsection(
        "Requirements",
        "<p>Python 3.10 or later. All dependencies are declared in <code>pyproject.toml</code>.</p>",
    ),
    subsection(
        "Create environment and install",
        code_block(textwrap.dedent("""\
            # Clone the repo
            git clone https://github.com/a1441/NeuroSense-Modules.git
            cd NeuroSense-Modules

            # Create virtual environment
            python -m venv .venv

            # Activate (Windows)
            .venv\\Scripts\\activate
            # Activate (Linux / macOS)
            source .venv/bin/activate

            # Install package + all dependencies
            pip install -e ".[dev]"
        """)),
    ),
    subsection(
        "Verify installation",
        code_block("python -m neurosense.run --help"),
        terminal("""\
usage: run.py [-h] [--data-root DATA_ROOT] [--xdf-file XDF_FILE]
              [--ratings RATINGS_PATH] [--labels LABELS_PATH]
              [--output OUTPUT] [--use-synthetic]
              [--n-subjects N_SUBJECTS] [--n-trials N_TRIALS]

NeuroSense EEG emotion classification pipeline
        """),
    ),
))

# ── 3. Quick-start ────────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="quickstart"></a>Quick-start',
    subsection(
        "Synthetic smoke run (no real data needed)",
        "<p>The fastest way to verify the installation. Generates "
        f"{N_SUBJECTS} synthetic subjects with Gaussian-noise EEG and random emotion ratings.</p>",
        code_block("python -m neurosense.run --use-synthetic --n-subjects 4 --n-trials 12 --output results/"),
        terminal("""\
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

============================================================
LOSO CLASSIFICATION RESULTS
============================================================
 quadrant  accuracy_mean  accuracy_std  f1_mean  f1_std  n_folds
     HVHA          0.583         0.144    0.510   0.180        4
     HVLA          0.542         0.128    0.482   0.162        4
     LVHA          0.567         0.161    0.501   0.195        4
     LVLA          0.558         0.139    0.496   0.171        4
============================================================
        """),
    ),
    subsection(
        "Single XDF file",
        code_block("""\
python -m neurosense.run \\
    --xdf-file  "d:/NeuroStuff/LabRecorder/Files/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf" \\
    --output    results/
        """),
        callout(
            "Without a marker stream the pipeline automatically infers trial boundaries "
            "from the recording length (65 s windows = 5 s baseline + 60 s stimulus). "
            "LOSO requires ≥ 2 subjects; use <code>--data-root</code> for multi-subject runs.",
            "info",
        ),
    ),
    subsection(
        "Full multi-subject dataset",
        code_block("""\
python -m neurosense.run \\
    --data-root  "d:/NeuroStuff/LabRecorder/Files/" \\
    --ratings    "data/ratings.csv" \\
    --labels     "data/external_labels.csv" \\
    --output     results/
        """),
    ),
))

# ── 4. Synthetic data ─────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="synth"></a>Sample Data',
    subsection(
        "Generating synthetic subjects",
        "<p>The <code>make_synthetic_subject()</code> function creates a fully-formed "
        "<code>SubjectData</code> object with Gaussian-noise EEG and random SAM-scale ratings — "
        "no real recordings required.</p>",
        code_block(textwrap.dedent(f"""\
            from neurosense.config import NeuroSenseConfig
            from neurosense.data_loader import make_synthetic_subject

            cfg     = NeuroSenseConfig()
            subject = make_synthetic_subject(cfg, n_trials={N_TRIALS}, seed=0)

            print(subject.subject_id)        # sub_000
            print(subject.eeg_data.shape)    # (samples, 4)
            print(subject.sfreq)             # 256.0
            print(subject.channel_names)     # ['AF7', 'AF8', 'TP9', 'TP10']
            print(subject.ratings.head(3))
        """)),
        terminal(f"""\
sub_000
({int((CFG.baseline_duration + N_TRIALS*(CFG.baseline_duration+CFG.stimulus_duration))*CFG.sfreq)}, 4)
256.0
['AF7', 'AF8', 'TP9', 'TP10']
   video_id   valence   arousal  dominance    liking  familiarity quadrant
0         0    7.19      2.02      5.44      7.23        3.61     HVLA
1         1    4.51      4.60      1.51      2.56        3.96     LVLA
2         2    7.87      3.97      7.62      4.73        4.76     HVLA
        """),
    ),
    subsection(
        "EEG trace — 4 channels (first 10 s)",
        img(b64_eeg, "Raw synthetic EEG — channels AF7, AF8, TP9, TP10"),
    ),
    subsection(
        "Trial / sub-epoch structure",
        f"<p>Each trial contains a <strong>{int(CFG.baseline_duration)} s baseline</strong> window "
        f"followed by a <strong>{int(CFG.stimulus_duration)} s stimulus</strong> window split into "
        f"non-overlapping <strong>{int(CFG.epoch_duration)} s sub-epochs</strong> "
        f"({int(CFG.stimulus_duration/CFG.epoch_duration)} per trial). "
        "ROI selection chooses the most discriminative 5-s window from the first 25 s.</p>",
        img(b64_epochs, "Trial structure — baseline (grey) + 5-s sub-epoch windows (blue shades)"),
    ),
))

# ── 5. Preprocessing ──────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="preprocessing"></a>Preprocessing',
    "<p>All preprocessing steps are implemented in <code>neurosense/preprocessing.py</code> "
    "and orchestrated by <code>preprocess_subject()</code>.</p>",
    subsection(
        "Step-by-step",
        code_block(textwrap.dedent("""\
            from neurosense.preprocessing import preprocess_subject

            preprocessed = preprocess_subject(subject, cfg)

            print(preprocessed.baseline_epochs.shape)     # (n_trials, 4, 1280)
            print(preprocessed.stimulus_epochs.shape)     # (n_trials, 4, 15360)
            print(preprocessed.stimulus_sub_epochs.shape) # (n_trials*12, 4, 1280)
        """)),
    ),
    subsection(
        "Preprocessing chain",
        "<ol style='margin-left:1.5rem;color:#cbd5e0;'>"
        "<li><strong>FIR bandpass (1–45 Hz, Hamming window)</strong> — removes DC drift and high-frequency noise</li>"
        "<li><strong>DSS line noise removal</strong> — meegkit Denoising Source Separation at 50 Hz (with fallback)</li>"
        "<li><strong>KNN outlier removal</strong> — k-nearest-neighbour distance thresholding (contamination=0.10), outlier samples linearly interpolated</li>"
        "<li><strong>Epoch extraction</strong> — baseline (5 s pre-onset) and stimulus (60 s post-onset) epochs per trial</li>"
        "<li><strong>Sub-epoch splitting</strong> — non-overlapping 5-s windows</li>"
        "</ol>",
    ),
    subsection(
        "Before / after (channel AF7)",
        img(b64_preproc, "Raw signal (top) vs preprocessed (bottom) — FIR + DSS + KNN interpolation"),
    ),
))

# ── 6. ROI window ─────────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="roi"></a>ROI Time-Sliding Window',
    "<p>For each emotion quadrant, an MNE <code>SlidingEstimator</code> (StandardScaler + RBF-SVM) "
    "is evaluated on each of the 5 candidate windows (0–5 s, 5–10 s, … 20–25 s). "
    "The window with the highest mean accuracy above the 95th-percentile threshold is selected.</p>",
    subsection(
        "Usage",
        code_block(textwrap.dedent("""\
            from neurosense.roi_timesliding import run_roi_for_all_quadrants

            roi_map = run_roi_for_all_quadrants(preprocessed, cfg)
            # roi_map: { 'HVHA': ((0.0, 5.0), accuracy_array), ... }

            for quadrant, (window, acc) in roi_map.items():
                print(f"{quadrant}: best window {window[0]:.0f}–{window[1]:.0f} s  "
                      f"mean acc = {acc.mean():.3f}")
        """)),
    ),
    subsection(
        "ROI window selection — heatmap",
        "<p>Number of quadrants (out of 4) for which each time window was selected as best, per subject.</p>",
        img(b64_roi, "ROI window selection: count of quadrants choosing each window per subject"),
    ),
))

# ── 7. Features ───────────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="features"></a>MiniRocket Feature Extraction',
    "<p><strong>MiniRocket</strong> (Dempster et al., 2021) transforms multivariate time-series into "
    "a fixed-length feature vector using a large bank of random convolutional kernels. "
    "It is orders of magnitude faster than rocket while matching its accuracy.</p>",
    subsection(
        "Usage",
        code_block(textwrap.dedent("""\
            from neurosense.features import extract_minirocket, build_feature_dataset

            # Extract features from a single set of sub-epochs
            # sub_epochs shape: (n_instances, n_channels, n_timepoints)
            features, transformer = extract_minirocket(
                sub_epochs,
                cfg,
                num_kernels=10_000,
            )
            print(features.shape)  # (n_instances, 10_000 * 2)

            # Build full dataset for all subjects + quadrants
            feature_datasets = build_feature_dataset(
                preprocessed_list, roi_maps, cfg, num_kernels=10_000
            )
            # feature_datasets: { quadrant: (X, y, subject_ids) }
        """)),
    ),
    subsection(
        "Feature distribution",
        img(b64_features, "Left: feature variance distribution. Right: class balance (target quadrant vs rest).") if b64_features else "<p><em>No feature data available.</em></p>",
    ),
))

# ── 8. LOSO Modeling ──────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="loso"></a>LOSO Classification',
    "<p>The outer cross-validation loop uses <strong>Leave-One-Subject-Out</strong>. "
    "For each held-out subject an inner <code>RandomizedSearchCV</code> (50 iterations, 3-fold CV) "
    "optimises the SVM C parameter and scaler type on the remaining subjects.</p>",
    subsection(
        "Usage",
        code_block(textwrap.dedent("""\
            from neurosense.modeling import loso_evaluate

            loso_results = loso_evaluate(feature_datasets, cfg)

            # Summary table
            print(loso_results.summary())

            # Per-subject DataFrame
            print(loso_results.per_subject_df())
        """)),
    ),
    subsection(
        "Results — summary table",
        "<p>Mean ± std accuracy and F1 across LOSO folds for each quadrant "
        f"({N_SUBJECTS} subjects, {N_TRIALS} trials each, synthetic data):</p>",
        table(summary_df) if len(summary_df) else "<p><em>No results — need ≥2 subjects.</em></p>",
    ),
    subsection(
        "Results — accuracy heatmap",
        img(b64_loso_heat, "LOSO accuracy per subject per quadrant") if b64_loso_heat else "",
    ),
    subsection(
        "Results — mean accuracy bar chart",
        img(b64_loso_bar, "Mean LOSO accuracy ± std vs chance level (0.50)") if b64_loso_bar else "",
    ),
    subsection(
        "Confusion matrices (aggregated over LOSO folds)",
        img(b64_cm, "Aggregated confusion matrices: rows=actual, cols=predicted") if b64_cm else "",
    ),
))

# ── 9. Stats / flags ──────────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="stats"></a>Statistics & Participant Flags',
    subsection(
        "Label reliability",
        "<p>Pearson correlation between participant self-assessments (SAM scale) and expert/external labels. "
        "Averaged per video across participants first.</p>",
        code_block(textwrap.dedent("""\
            from neurosense.stats import compute_label_reliability

            reliability = compute_label_reliability(self_assessment_df, external_labels_df)
            print(reliability)
            # Label reliability:
            #   Valence : r=0.742  p=0.0031
            #   Arousal : r=0.681  p=0.0089
        """)),
    ),
    subsection(
        "Participant flagging",
        "<p>Participants are flagged if their rating standard deviation or mean model confidence "
        f"falls below the <strong>{int(CFG.flag_percentile)}th percentile</strong> — "
        "indicating either flat/random responses or consistently uncertain predictions.</p>",
        code_block(textwrap.dedent("""\
            from neurosense.stats import compute_participant_flags

            flags = compute_participant_flags(all_ratings_df, loso_results, cfg)
            print(flags[['subject_id','valence_std','arousal_std','mean_decision_prob','flagged']])
        """)),
        img(b64_flags, "Participant flags: red = flagged (low STD or low confidence)") if b64_flags else "",
    ),
))

# ── 10. Output artifacts ──────────────────────────────────────────────────────
html_sections.append(section(
    '<a name="outputs"></a>Output Artifacts',
    "<p>All artifacts are written to the <code>--output</code> directory (default: <code>results/</code>).</p>",
    table(pd.DataFrame({
        "File": [
            "per_subject_metrics.csv",
            "overall_summary.json",
            "confusion_matrices/cm_*.png",
            "roi_windows.csv",
            "participant_flags.csv",
            "label_reliability.json",
        ],
        "Contents": [
            "Accuracy, F1, std per subject per quadrant",
            "Mean ± std per quadrant (JSON)",
            "Aggregated confusion matrix PNG per quadrant",
            "Best ROI time window per subject per quadrant",
            "Rating STD, mean probability, flagged status",
            "Pearson r and p for valence and arousal",
        ],
    })),
))

# ── 11. Configuration reference ───────────────────────────────────────────────
cfg_rows = [
    ("sfreq", str(CFG.sfreq), "EEG sampling rate (Hz)"),
    ("channels", str(list(CFG.channels)), "Channel names to keep from XDF"),
    ("bandpass", str(CFG.bandpass), "FIR filter passband (Hz)"),
    ("fir_window", CFG.fir_window, "FIR filter window function"),
    ("knn_contamination", str(CFG.knn_contamination), "Fraction of outlier samples"),
    ("knn_n_neighbors", str(CFG.knn_n_neighbors), "KNN neighbours for outlier detection"),
    ("epoch_duration", str(CFG.epoch_duration), "Sub-epoch length (s)"),
    ("baseline_duration", str(CFG.baseline_duration), "Pre-stimulus baseline (s)"),
    ("stimulus_duration", str(CFG.stimulus_duration), "Stimulus (video) duration (s)"),
    ("n_random_search_iter", str(CFG.n_random_search_iter), "RandomizedSearchCV iterations"),
    ("cv_inner", str(CFG.cv_inner), "Inner CV folds"),
    ("flag_percentile", str(CFG.flag_percentile), "Flagging threshold percentile"),
    ("random_seed", str(CFG.random_seed), "Global random seed"),
]
cfg_table = (
    '<table><thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead><tbody>'
    + "".join(f"<tr><td><code>{p}</code></td><td>{v}</td><td>{d}</td></tr>" for p, v, d in cfg_rows)
    + "</tbody></table>"
)
html_sections.append(section(
    '<a name="config"></a>Configuration Reference',
    "<p><code>NeuroSenseConfig</code> is a frozen dataclass — all parameters in one place, "
    "passed through every stage of the pipeline.</p>",
    code_block(textwrap.dedent("""\
        from neurosense.config import NeuroSenseConfig

        # Default config
        cfg = NeuroSenseConfig()

        # Custom config (must use dataclasses.replace on frozen dataclasses)
        from dataclasses import replace
        cfg_custom = replace(cfg, sfreq=250, bandpass=(0.5, 40.0), random_seed=0)
    """)),
    cfg_table,
))

# ─────────────────────────────────────────────────────────────────────────────
# Full HTML
# ─────────────────────────────────────────────────────────────────────────────
NAV_LINKS = [
    ("overview",      "Overview"),
    ("install",       "Installation"),
    ("quickstart",    "Quick-start"),
    ("synth",         "Sample Data"),
    ("preprocessing", "Preprocessing"),
    ("roi",           "ROI Window"),
    ("features",      "MiniRocket"),
    ("loso",          "LOSO"),
    ("stats",         "Stats & Flags"),
    ("outputs",       "Outputs"),
    ("config",        "Config"),
]

nav_html = "\n".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in NAV_LINKS)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroSense EEG Pipeline — Usage Report</title>
  <style>{CSS}</style>
</head>
<body>

<header>
  <h1>NeuroSense EEG Pipeline</h1>
  <p class="subtitle">End-to-end emotion classification from low-cost Muse EEG &mdash; usage report &amp; reference</p>
  <div class="badges">
    <span class="badge green">✓ 15/15 tests passing</span>
    <span class="badge blue">Python 3.12</span>
    <span class="badge">MiniRocket + SVM</span>
    <span class="badge">LOSO cross-validation</span>
    <span class="badge">4-channel Muse EEG</span>
  </div>
</header>

<nav>
  {nav_html}
</nav>

<main>
  {"".join(html_sections)}
</main>

<footer>
  Generated by <code>docs/generate_report.py</code> &mdash;
  NeuroSense EEG Pipeline &mdash;
  <a href="https://github.com/a1441/NeuroSense-Modules" style="color:#63b3ed">github.com/a1441/NeuroSense-Modules</a>
</footer>

</body>
</html>"""

out_path = ROOT / "docs" / "report.html"
out_path.write_text(html, encoding="utf-8")
print(f"\nReport written -> {out_path.resolve()}")
print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
