"""
generate_tutorial.py
--------------------
Produces docs/tutorial.html — a fully self-contained, noob-friendly tutorial
covering both the NeuroSense paper methodology and this exact implementation.

Usage
-----
  python docs/generate_tutorial.py
"""
from __future__ import annotations

import base64, io, sys, textwrap, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ─── palette ──────────────────────────────────────────────────────────────────
P = dict(
    red="#e74c3c", orange="#e67e22", blue="#3498db", green="#2ecc71",
    purple="#9b59b6", teal="#1abc9c", yellow="#f1c40f", grey="#95a5a6",
    bg="#0f1117", card="#1a202c", border="#2d3748", text="#cbd5e0",
    accent="#63b3ed", soft="#a0aec0",
)
CH_COLORS = [P["red"], P["blue"], P["green"], P["purple"]]
plt.rcParams.update({
    "figure.facecolor": P["card"], "axes.facecolor": P["card"],
    "axes.edgecolor": P["border"], "axes.labelcolor": P["text"],
    "xtick.color": P["soft"], "ytick.color": P["soft"],
    "text.color": P["text"], "grid.color": P["border"],
    "grid.alpha": 0.5,
})

# ─── helpers ──────────────────────────────────────────────────────────────────

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def code(src, lang="python"):
    return f'<pre class="code"><code class="lang-{lang}">{esc(textwrap.dedent(src).strip())}</code></pre>'

def term(src):
    return f'<pre class="term">{esc(textwrap.dedent(src).strip())}</pre>'

def fig_html(b64, caption=""):
    cap = f'<p class="fig-cap">{caption}</p>' if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">{cap}</figure>'

def callout(kind, title, body):
    icons = {"info":"ℹ", "warn":"⚠", "ok":"✓", "gate":"⬡", "mistake":"✗", "math":"∑"}
    return (f'<div class="callout callout-{kind}">'
            f'<div class="callout-title"><span class="callout-icon">{icons[kind]}</span>{title}</div>'
            f'<div class="callout-body">{body}</div></div>')

def checklist(*items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<ul class="checklist">{lis}</ul>'

def schema_table(headers, rows):
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f'<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'

def df_table(df, fmt=".3f"):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{v:{fmt}}</td>" if isinstance(v, float) else f"<td>{v}</td>"
            for v in row
        )
        rows += f"<tr>{cells}</tr>"
    return f'<table><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table>'

def section_wrap(anchor, number, title, *content):
    body = "\n".join(content)
    num_badge = f'<span class="sec-num">{number}</span>' if number else ""
    return (f'<section id="{anchor}">'
            f'<h2>{num_badge}{title}</h2>'
            f'{body}</section>')

def subsec(title, *content):
    body = "\n".join(content)
    return f'<div class="subsec"><h3>{title}</h3>{body}</div>'

def have_now(*items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return (f'<div class="have-now">'
            f'<div class="have-now-title">What you should have now</div>'
            f'<ul>{lis}</ul></div>')

def shape_badge(s): return f'<code class="shape">{s}</code>'
def mono(s): return f'<code>{s}</code>'

# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("Loading modules...")
from neurosense.config import NeuroSenseConfig, RUSSELL_QUADRANTS, assign_quadrant
from neurosense.data_loader import make_synthetic_subject
from neurosense.preprocessing import (
    preprocess_subject, _to_mne_raw, apply_fir_filter,
    reduce_ringing_artifacts, remove_ocular_outliers, build_epochs,
)
from neurosense.roi_timesliding import run_roi_for_all_quadrants
from neurosense.features import build_feature_dataset, extract_minirocket
from neurosense.modeling import loso_evaluate
from neurosense.stats import compute_participant_flags

CFG  = NeuroSenseConfig()
NS   = 4    # subjects
NT   = 12   # trials

print(f"Generating {NS} synthetic subjects, {NT} trials each...")
subjects     = [make_synthetic_subject(CFG, n_trials=NT, seed=i*7) for i in range(NS)]
preprocessed = [preprocess_subject(s, CFG) for s in subjects]
print("ROI selection...")
roi_maps     = [run_roi_for_all_quadrants(p, CFG) for p in preprocessed]
print("MiniRocket features...")
feat_ds      = build_feature_dataset(preprocessed, roi_maps, CFG, num_kernels=500)
print("LOSO evaluation...")
loso         = loso_evaluate(feat_ds, CFG)
all_ratings  = pd.concat([s.ratings for s in subjects], ignore_index=True)
for s in subjects:
    s.ratings["user_id"] = s.subject_id
all_ratings = pd.concat([s.ratings for s in subjects], ignore_index=True)
flags        = compute_participant_flags(all_ratings, loso, CFG)
summary_df   = loso.summary()
per_subj_df  = loso.per_subject_df()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES — one per pipeline concept
# ─────────────────────────────────────────────────────────────────────────────
print("Building figures...")

# ── FIG 0: Russell Circumplex ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.add_patch(plt.Circle((0,0),1,fill=False,color=P["border"],lw=1.5,ls="--"))
ax.axhline(0,color=P["border"],lw=0.8); ax.axvline(0,color=P["border"],lw=0.8)
quad_data = {
    "HVHA": (0.68, 0.68, P["red"],    "HVHA\nHappy / Excited"),
    "LVHA": (-0.68,0.68, P["orange"], "LVHA\nAngry / Fearful"),
    "LVLA": (-0.68,-0.68,P["blue"],   "LVLA\nSad / Bored"),
    "HVLA": (0.68,-0.68, P["green"],  "HVLA\nCalm / Relaxed"),
}
for q,(x,y,c,lbl) in quad_data.items():
    ax.scatter(x,y,s=260,color=c,zorder=5,edgecolors="white",lw=0.5)
    ax.annotate(lbl,(x,y),xytext=(14 if x>0 else -14,6),
                textcoords="offset points",fontsize=9,ha="left" if x>0 else "right",
                color=c,fontweight="bold",va="center")
ax.text(0.02, 1.35,"High Arousal",ha="center",fontsize=9,color=P["soft"])
ax.text(0.02,-1.35,"Low Arousal", ha="center",fontsize=9,color=P["soft"])
ax.text( 1.35, 0,"High Valence",ha="center",fontsize=9,color=P["soft"],rotation=270)
ax.text(-1.35, 0,"Low Valence", ha="center",fontsize=9,color=P["soft"],rotation=90)
ax.set_title("Russell Circumplex Model of Affect",fontsize=11,fontweight="bold",pad=14)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
B_RUSSELL = fig_b64(fig)

# ── FIG 1: Pipeline diagram ───────────────────────────────────────────────────
stages = [
    ("1\nLoad",        P["blue"],   "XDF → SubjectData"),
    ("2\nPreprocess",  P["teal"],   "FIR·DSS·KNN"),
    ("3\nEpoch",       P["green"],  "baseline+stimulus\n→ sub-epochs"),
    ("4\nROI",         P["yellow"], "pick best\n5-s window"),
    ("5\nMiniRocket",  P["orange"], "10 K kernels\n→ features"),
    ("6\nLOSO SVM",    P["red"],    "RandomSearch\n+LeaveOneOut"),
    ("7\nStats",       P["purple"], "reliability\n+ flags"),
]
fig, ax = plt.subplots(figsize=(13, 2.4))
ax.set_xlim(0,13); ax.set_ylim(0,1); ax.axis("off")
for i,(label,color,sub) in enumerate(stages):
    x = 0.5 + i*1.85
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x-0.7,0.15),1.4,0.7,
        boxstyle="round,pad=0.05",facecolor=color,alpha=0.18,
        edgecolor=color,lw=1.5))
    ax.text(x,0.67,label.split("\n")[0],ha="center",va="center",
            fontsize=10,fontweight="bold",color=color)
    ax.text(x,0.38,sub,ha="center",va="center",fontsize=7.5,color=P["text"])
    if i<len(stages)-1:
        ax.annotate("",xy=(x+0.84,0.5),xytext=(x+0.74,0.5),
                    arrowprops=dict(arrowstyle="->",color=P["soft"],lw=1.3))
fig.tight_layout()
B_PIPELINE = fig_b64(fig)

# ── FIG 2: Raw EEG traces ─────────────────────────────────────────────────────
s0 = subjects[0]
show = int(10*CFG.sfreq); t10 = np.arange(show)/CFG.sfreq
fig, axes = plt.subplots(4,1,figsize=(11,5),sharex=True)
for i,(ax,ch) in enumerate(zip(axes,s0.channel_names)):
    ax.plot(t10,s0.eeg_data[:show,i],color=CH_COLORS[i],lw=0.75,alpha=0.9)
    ax.set_ylabel(ch,fontsize=9,rotation=0,labelpad=36,va="center")
    ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True,axis="x",alpha=0.3)
axes[-1].set_xlabel("Time (s)")
axes[0].set_title("Raw EEG — first 10 s of recording",fontweight="bold",pad=8)
fig.tight_layout()
B_RAWEEG = fig_b64(fig)

# ── FIG 3: Marker timeline ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,2))
ax.set_xlim(s0.timestamps[0], s0.timestamps[-1])
ax.set_ylim(0,1); ax.axis("off")
ax.fill_between([s0.timestamps[0],s0.timestamps[-1]],[0,0],[0.35],
                color=P["blue"],alpha=0.12)
ax.text(np.mean([s0.timestamps[0],s0.timestamps[-1]]),0.17,
        "Continuous EEG recording",ha="center",fontsize=9,color=P["blue"])
trial_len = CFG.baseline_duration + CFG.stimulus_duration
for _, row in s0.markers.iterrows():
    t = row["timestamp"]
    # baseline
    ax.barh(0.62, CFG.baseline_duration, left=t-CFG.baseline_duration,
            height=0.28, color=P["grey"], alpha=0.7)
    # stimulus
    ax.barh(0.62, CFG.stimulus_duration, left=t,
            height=0.28, color=P["teal"], alpha=0.7)
    ax.axvline(t, color=P["yellow"], lw=1.2, ls="--", alpha=0.8)
from matplotlib.patches import Patch as MPatch
ax.legend(handles=[
    MPatch(facecolor=P["grey"],label="Baseline (5 s)"),
    MPatch(facecolor=P["teal"],label="Stimulus (60 s)"),
    MPatch(facecolor=P["yellow"],label="Trial onset marker"),
],loc="upper right",fontsize=8,framealpha=0.2)
ax.set_title("Trial onset markers overlaid on recording timeline",fontweight="bold",pad=6)
fig.tight_layout()
B_MARKERS = fig_b64(fig)

# ── FIG 4: PSD before/after filter ───────────────────────────────────────────
raw0   = _to_mne_raw(s0, CFG)
raw_f  = apply_fir_filter(raw0, CFG)
ch_idx = 0
sig_raw  = raw0.get_data()[ch_idx]*1e6
sig_filt = raw_f.get_data()[ch_idx]*1e6
f_raw,  p_raw  = welch(sig_raw,  fs=CFG.sfreq, nperseg=512)
f_filt, p_filt = welch(sig_filt, fs=CFG.sfreq, nperseg=512)
fig, ax = plt.subplots(figsize=(9,3.5))
ax.semilogy(f_raw,  p_raw,  color=P["red"],  lw=1.2, alpha=0.8, label="Before FIR")
ax.semilogy(f_filt, p_filt, color=P["green"],lw=1.2, alpha=0.8, label="After FIR (1–45 Hz)")
ax.axvspan(0,1,   color=P["red"],alpha=0.08,label="DC / drift removed")
ax.axvspan(45,128,color=P["orange"],alpha=0.08,label="HF noise removed")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (µV²/Hz)")
ax.set_title(f"Power Spectral Density — channel {s0.channel_names[ch_idx]}",fontweight="bold")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_xlim(0,80)
fig.tight_layout()
B_PSD = fig_b64(fig)

# ── FIG 5: Before/after time domain ──────────────────────────────────────────
n5 = int(5*CFG.sfreq); t5 = np.arange(n5)/CFG.sfreq
raw_5  = sig_raw[:n5]
filt_5 = sig_filt[:n5]
# add synthetic 50 Hz line noise to raw for illustration
t_full = np.arange(len(sig_raw))/CFG.sfreq
line_noise = 8*np.sin(2*np.pi*50*t_full[:n5])
raw_noisy = raw_5 + line_noise
fig, axes = plt.subplots(3,1,figsize=(11,5),sharex=True)
axes[0].plot(t5,raw_noisy,color=P["red"],lw=0.75)
axes[0].set_title("Raw signal (+ simulated 50 Hz line noise)",fontsize=10)
axes[1].plot(t5,raw_5,color=P["orange"],lw=0.75)
axes[1].set_title("After FIR bandpass (1–45 Hz, Hamming)",fontsize=10)
axes[2].plot(t5,filt_5,color=P["green"],lw=0.75)
axes[2].set_title("After DSS line-noise removal",fontsize=10)
for ax in axes:
    ax.set_ylabel("µV"); ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Preprocessing stages — channel AF7 (first 5 s)",fontweight="bold",y=1.01)
fig.tight_layout()
B_PREPROC_TD = fig_b64(fig)

# ── FIG 6: KNN outlier mask ───────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors
p0   = preprocessed[0]
data_seg = p0.stimulus_epochs[0]   # (4, 15360)
ch_seg   = data_seg[0]             # first channel
n_seg    = len(ch_seg)
X_seg    = data_seg.T              # (n_times, 4)
nbrs     = NearestNeighbors(n_neighbors=CFG.knn_n_neighbors).fit(X_seg)
dists, _ = nbrs.kneighbors(X_seg)
k_dist   = dists[:,-1]
thresh   = np.percentile(k_dist, 100*(1-CFG.knn_contamination))
mask     = k_dist > thresh
t_seg    = np.arange(n_seg)/CFG.sfreq
fig, axes = plt.subplots(2,1,figsize=(11,4.5),sharex=True)
axes[0].plot(t_seg, ch_seg, color=P["blue"], lw=0.7, label="EEG", alpha=0.9)
axes[0].scatter(t_seg[mask], ch_seg[mask], color=P["red"], s=8, zorder=5,
                label=f"Outliers ({mask.sum()} samples, {mask.mean()*100:.1f}%)")
axes[0].set_ylabel("µV"); axes[0].legend(fontsize=8)
axes[0].set_title("KNN outlier detection — flagged samples (red)", fontweight="bold")
axes[0].spines[["top","right"]].set_visible(False)
axes[1].plot(t_seg, k_dist, color=P["orange"], lw=0.7)
axes[1].axhline(thresh, color=P["red"], lw=1.5, ls="--",
                label=f"Threshold (90th pct = {thresh:.2f})")
axes[1].set_ylabel("KNN distance"); axes[1].set_xlabel("Time (s)")
axes[1].legend(fontsize=8); axes[1].spines[["top","right"]].set_visible(False)
axes[1].set_title("k-distance to nearest neighbour — above threshold = outlier", fontsize=10)
fig.tight_layout()
B_KNN = fig_b64(fig)

# ── FIG 7: Epoch structure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,2.5))
trial_len = CFG.baseline_duration + CFG.stimulus_duration
n_show = 3
ax.set_xlim(0, n_show*trial_len+2); ax.set_ylim(0,1); ax.axis("off")
sub_colors = ["#1f4e79","#2471a3","#2e86c1","#3498db","#5dade2",
              "#85c1e9","#aed6f1","#d6eaf8","#ebf5fb","#f2f3f4","#c0c0c0","#a0a0a0"]
for t in range(n_show):
    x0 = t*trial_len
    # baseline bar
    ax.add_patch(plt.Rectangle((x0,0.35),CFG.baseline_duration,0.4,
                                facecolor=P["grey"],alpha=0.6,edgecolor=P["border"],lw=0.5))
    ax.text(x0+CFG.baseline_duration/2,0.55,"BL\n5 s",ha="center",va="center",
            fontsize=7.5,color="white",fontweight="bold")
    # sub-epoch bars
    n_sub = int(CFG.stimulus_duration/CFG.epoch_duration)
    for w in range(n_sub):
        wx = x0+CFG.baseline_duration+w*CFG.epoch_duration
        ax.add_patch(plt.Rectangle((wx+0.1,0.35),CFG.epoch_duration-0.2,0.4,
                                    facecolor=sub_colors[w%len(sub_colors)],
                                    alpha=0.85,edgecolor=P["border"],lw=0.3))
        if w<6:
            ax.text(wx+CFG.epoch_duration/2,0.55,f"W{w+1}",
                    ha="center",va="center",fontsize=6.5,color="white")
        elif w==6:
            ax.text(wx+CFG.epoch_duration/2,0.55,"…",
                    ha="center",va="center",fontsize=10,color="white")
    # trial label
    ax.text(x0+trial_len/2,0.22,f"Trial {t+1}",ha="center",va="center",
            fontsize=9,color=P["soft"])
    # marker line
    ax.axvline(x0+CFG.baseline_duration,color=P["yellow"],lw=1.3,ls="--",alpha=0.7)
ax.text(CFG.baseline_duration,0.92,"← Stimulus onset markers",
        fontsize=8,color=P["yellow"],va="top")
ax.set_title("Trial structure — baseline (grey) + 12 × 5-s sub-epoch windows",
             fontweight="bold",y=0.98)
fig.tight_layout()
B_EPOCHS = fig_b64(fig)

# ── FIG 8: ROI window selection ───────────────────────────────────────────────
win_labels = [f"{int(s)}-{int(e)}s" for s,e in CFG.roi_windows]
roi_matrix = np.zeros((NS, len(CFG.roi_windows)))
acc_matrix = np.zeros((NS, len(CFG.roi_windows)))
for si, roi_map in enumerate(roi_maps):
    for q in CFG.quadrant_names:
        win, acc = roi_map[q]
        wi = list(CFG.roi_windows).index(win)
        roi_matrix[si,wi] += 1
        acc_matrix[si,wi] = max(acc_matrix[si,wi], float(acc.mean()))
subj_labels = [s.subject_id for s in subjects]
fig, axes = plt.subplots(1,2,figsize=(11,3.5))
sns.heatmap(roi_matrix,annot=True,fmt=".0f",cmap="Blues",ax=axes[0],
            xticklabels=win_labels,yticklabels=subj_labels,
            linewidths=0.5,cbar_kws={"label":"# quadrants"})
axes[0].set_title("ROI windows chosen (count of quadrants)",fontweight="bold")
axes[0].set_xlabel("Time window"); axes[0].set_ylabel("Subject")
sns.heatmap(acc_matrix,annot=True,fmt=".2f",cmap="YlOrRd",ax=axes[1],
            xticklabels=win_labels,yticklabels=subj_labels,
            linewidths=0.5,cbar_kws={"label":"Max mean accuracy"},vmin=0,vmax=1)
axes[1].set_title("Best accuracy achieved per window",fontweight="bold")
axes[1].set_xlabel("Time window"); axes[1].set_ylabel("")
fig.tight_layout()
B_ROI = fig_b64(fig)

# ── FIG 9: MiniRocket features ───────────────────────────────────────────────
if feat_ds:
    q0 = list(feat_ds.keys())[0]
    X0, y0, _ = feat_ds[q0]
    var0 = X0.var(axis=0)
    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    axes[0].hist(var0,bins=50,color=P["blue"],alpha=0.85,edgecolor="none")
    axes[0].set_xlabel("Feature variance"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Feature variance dist.\n({X0.shape[1]} features, q={q0})")
    axes[0].spines[["top","right"]].set_visible(False)
    uq,cts = np.unique(y0,return_counts=True)
    bar_lbls = [q0 if u==1 else "Other" for u in uq]
    bar_cs   = [P["red"] if u==1 else P["grey"] for u in uq]
    axes[1].bar(bar_lbls,cts,color=bar_cs,edgecolor="none")
    for xi,ct in enumerate(cts):
        axes[1].text(xi,ct+0.5,str(ct),ha="center",fontsize=9)
    axes[1].set_ylabel("# instances"); axes[1].set_title(f"Class balance — {q0}")
    axes[1].spines[["top","right"]].set_visible(False)
    # cumulative variance
    sv = np.sort(var0)[::-1]; cv = np.cumsum(sv)/sv.sum()*100
    axes[2].plot(cv,color=P["teal"],lw=1.5)
    axes[2].axhline(90,color=P["yellow"],ls="--",lw=1,label="90% variance")
    idx90 = np.searchsorted(cv,90)
    axes[2].axvline(idx90,color=P["yellow"],ls="--",lw=1)
    axes[2].set_xlabel("# features (sorted)"); axes[2].set_ylabel("Cumulative variance (%)")
    axes[2].set_title("Cumulative variance explained"); axes[2].legend(fontsize=8)
    axes[2].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    B_FEAT = fig_b64(fig)
else:
    B_FEAT = ""

# ── FIG 10: LOSO results ──────────────────────────────────────────────────────
if loso.folds:
    pivot = per_subj_df.pivot_table(index="subject_id",columns="quadrant",values="accuracy")
    fig, axes = plt.subplots(1,2,figsize=(12,3.8))
    sns.heatmap(pivot,annot=True,fmt=".2f",cmap="RdYlGn",vmin=0,vmax=1,
                ax=axes[0],linewidths=0.5,cbar_kws={"label":"Accuracy"})
    axes[0].set_title("LOSO Accuracy per Subject per Quadrant",fontweight="bold")
    axes[0].set_xlabel("Quadrant"); axes[0].set_ylabel("Left-out subject")
    qs   = summary_df["quadrant"].tolist()
    means= summary_df["accuracy_mean"].tolist()
    stds = summary_df["accuracy_std"].tolist()
    bcs  = [P["red"],P["orange"],P["blue"],P["green"]][:len(qs)]
    axes[1].bar(qs,means,yerr=stds,capsize=6,color=bcs,alpha=0.85,edgecolor="none")
    axes[1].axhline(0.5,color=P["grey"],lw=1.5,ls="--",label="Chance (0.50)")
    axes[1].set_ylim(0,1); axes[1].set_ylabel("Accuracy"); axes[1].set_xlabel("Quadrant")
    axes[1].set_title("Mean LOSO Accuracy ± std",fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    B_LOSO = fig_b64(fig)

    # confusion matrices
    q_with_folds = sorted({f.quadrant for f in loso.folds})
    fig, axes = plt.subplots(1,len(q_with_folds),figsize=(4*len(q_with_folds),3.8))
    if len(q_with_folds)==1: axes=[axes]
    for ax,q in zip(axes,q_with_folds):
        agg = sum(f.confusion_matrix for f in loso.folds if f.quadrant==q)
        sns.heatmap(agg,annot=True,fmt="d",cmap="Blues",ax=ax,
                    xticklabels=["Other","Target"],yticklabels=["Other","Target"],
                    linewidths=0.5,cbar=False)
        ax.set_title(f"{q}",fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Aggregated Confusion Matrices (LOSO)",fontweight="bold")
    fig.tight_layout()
    B_CM = fig_b64(fig)
else:
    B_LOSO = B_CM = ""

# ── FIG 11: Participant flags ─────────────────────────────────────────────────
if flags is not None and "valence_std" in flags.columns and len(flags):
    flag_colors = [P["red"] if f else P["green"] for f in flags["flagged"]]
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    axes[0].bar(flags["subject_id"],flags["valence_std"],color=flag_colors,edgecolor="none")
    thr_v = np.nanpercentile(flags["valence_std"],CFG.flag_percentile)
    axes[0].axhline(thr_v,color=P["yellow"],ls="--",lw=1.5,label=f"25th pct ({thr_v:.2f})")
    axes[0].set_title("Valence STD per subject"); axes[0].set_ylabel("STD")
    axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)
    if "mean_decision_prob" in flags.columns and flags["mean_decision_prob"].notna().any():
        axes[1].bar(flags["subject_id"],flags["mean_decision_prob"].fillna(0),
                    color=flag_colors,edgecolor="none")
        thr_p = np.nanpercentile(flags["mean_decision_prob"].dropna(),CFG.flag_percentile)
        axes[1].axhline(thr_p,color=P["yellow"],ls="--",lw=1.5,label=f"25th pct ({thr_p:.2f})")
        axes[1].set_title("Mean decision probability"); axes[1].set_ylabel("Probability")
        axes[1].legend(fontsize=8); axes[1].spines[["top","right"]].set_visible(False)
    handles=[MPatch(facecolor=P["red"],label="Flagged"),MPatch(facecolor=P["green"],label="OK")]
    fig.legend(handles=handles,loc="upper right",fontsize=8,framealpha=0.2)
    fig.suptitle("Participant Quality Flags",fontweight="bold")
    fig.tight_layout()
    B_FLAGS = fig_b64(fig)
else:
    B_FLAGS = ""

print("Building HTML...")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root{
  --bg:#0f1117;--card:#1a202c;--card2:#141920;--border:#2d3748;
  --text:#cbd5e0;--soft:#a0aec0;--accent:#63b3ed;--green:#48bb78;
  --red:#fc8181;--yellow:#f6e05e;--orange:#f6ad55;--purple:#b794f4;
  --teal:#4fd1c5;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.7;font-size:15px;
  display:flex;flex-direction:column;min-height:100vh;
}
a{color:var(--accent);text-decoration:none;}
a:hover{text-decoration:underline;}

/* ── header ── */
header{
  background:linear-gradient(135deg,#1a1f35 0%,var(--bg) 100%);
  border-bottom:1px solid var(--border);padding:2.5rem 2rem 1.8rem;
  text-align:center;
}
header h1{font-size:2.2rem;color:var(--accent);letter-spacing:-0.5px;}
header .tagline{color:var(--soft);margin-top:.5rem;font-size:1rem;}
.badges{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:1rem;}
.badge{
  background:var(--card);border:1px solid var(--border);
  border-radius:999px;padding:.2rem .8rem;font-size:.75rem;color:var(--soft);
}
.badge.g{border-color:var(--green);color:var(--green);}
.badge.b{border-color:var(--accent);color:var(--accent);}

/* ── layout ── */
.layout{display:flex;flex:1;max-width:1300px;margin:0 auto;width:100%;padding:0 1rem;}

/* ── sidebar ── */
aside{
  width:230px;flex-shrink:0;
  position:sticky;top:0;height:100vh;overflow-y:auto;
  padding:1.5rem 0 2rem 0;
  border-right:1px solid var(--border);
}
aside h4{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
         color:var(--soft);padding:0 1rem .5rem;margin-top:1rem;}
aside a{
  display:block;padding:.35rem 1rem;font-size:.82rem;color:var(--soft);
  border-left:2px solid transparent;transition:all .15s;
}
aside a:hover,aside a.active{
  color:var(--accent);border-left-color:var(--accent);
  background:rgba(99,179,237,.06);text-decoration:none;
}
aside .toc-sub{padding-left:1.8rem;font-size:.77rem;}

/* ── main ── */
main{flex:1;padding:2rem 2.5rem 5rem;max-width:920px;}
section{margin-bottom:3.5rem;scroll-margin-top:1.5rem;}
section>h2{
  font-size:1.5rem;color:var(--accent);margin-bottom:1.4rem;
  padding-bottom:.5rem;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:.6rem;
}
.sec-num{
  background:var(--accent);color:#0f1117;border-radius:6px;
  padding:.1rem .5rem;font-size:.9rem;font-weight:700;flex-shrink:0;
}
.subsec{margin:1.6rem 0;}
.subsec h3{font-size:1.05rem;color:#90cdf4;margin-bottom:.75rem;}
p{margin-bottom:.85rem;color:var(--text);}
ul,ol{margin:.5rem 0 .9rem 1.4rem;color:var(--text);}
li{margin-bottom:.3rem;}
strong{color:#e2e8f0;}

/* ── code ── */
pre.code{
  background:#0d1117;border:1px solid var(--border);border-radius:8px;
  padding:1.1rem 1.3rem;overflow-x:auto;margin:.8rem 0 1.2rem;
  font-size:.82rem;line-height:1.65;
}
pre.code code{color:#e2e8f0;font-family:'Fira Code',Consolas,monospace;}
pre.term{
  background:#0b1120;border:1px solid #1a3050;border-radius:8px;
  padding:1rem 1.3rem;overflow-x:auto;margin:.8rem 0 1.2rem;
  font-size:.81rem;color:#68d391;font-family:Consolas,monospace;line-height:1.65;
}
code{
  background:#1a202c;border:1px solid var(--border);
  border-radius:4px;padding:.1rem .35rem;font-size:.83rem;
  font-family:'Fira Code',Consolas,monospace;color:#fbb6ce;
}
pre code{background:none;border:none;padding:0;color:inherit;}
code.shape{color:var(--teal);background:rgba(79,209,197,.12);border-color:rgba(79,209,197,.3);}

/* ── callouts ── */
.callout{
  border-radius:7px;padding:.85rem 1.1rem;margin:1rem 0 1.3rem;
  border-left:3px solid;
}
.callout-title{font-weight:700;margin-bottom:.4rem;display:flex;align-items:center;gap:.5rem;}
.callout-icon{font-size:1rem;}
.callout-body{font-size:.9rem;}
.callout-info  {background:#1a365d22;border-color:#3182ce;color:#bee3f8;}
.callout-info .callout-title{color:#90cdf4;}
.callout-warn  {background:#7b341e22;border-color:#dd6b20;color:#fbd38d;}
.callout-warn .callout-title{color:#f6ad55;}
.callout-ok    {background:#1c453222;border-color:#38a169;color:#c6f6d5;}
.callout-ok .callout-title{color:#68d391;}
.callout-gate  {background:#44337a22;border-color:#805ad5;color:#d6bcfa;}
.callout-gate .callout-title{color:#b794f4;}
.callout-mistake{background:#74212122;border-color:#e53e3e;color:#fed7d7;}
.callout-mistake .callout-title{color:#fc8181;}
.callout-math  {background:#1a365d22;border-color:#4299e1;color:#bee3f8;}
.callout-math .callout-title{color:#63b3ed;}

/* ── have-now box ── */
.have-now{
  background:linear-gradient(135deg,#1c4532,#153227);
  border:1px solid #276749;border-radius:8px;
  padding:1rem 1.3rem;margin:1.5rem 0;
}
.have-now-title{color:#68d391;font-weight:700;margin-bottom:.5rem;font-size:.95rem;}
.have-now ul{margin-left:1.2rem;color:#c6f6d5;}

/* ── figures ── */
figure{margin:1.2rem 0 1.8rem;}
figure img{max-width:100%;border-radius:8px;border:1px solid var(--border);}
.fig-cap{font-size:.8rem;color:var(--soft);margin-top:.4rem;font-style:italic;}

/* ── tables ── */
table{width:100%;border-collapse:collapse;margin:1rem 0 1.5rem;font-size:.86rem;}
thead{background:var(--card2);}
th{padding:.6rem .85rem;text-align:left;color:#90cdf4;font-weight:600;}
td{padding:.5rem .85rem;border-bottom:1px solid var(--border);color:var(--text);}
tr:hover td{background:rgba(255,255,255,.02);}

/* ── checklists ── */
ul.checklist{list-style:none;margin-left:0;}
ul.checklist li::before{content:"☐ ";color:var(--green);}

/* ── two-col ── */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1rem 0;}
@media(max-width:700px){.two-col{grid-template-columns:1fr;}}

/* ── step cards ── */
.step-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:.85rem;margin:1.2rem 0;
}
.step-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:.9rem;text-align:center;
}
.step-card-num{font-size:1.4rem;font-weight:700;margin-bottom:.2rem;}
.step-card-name{font-size:.85rem;color:#90cdf4;font-weight:600;}
.step-card-sub{font-size:.75rem;color:var(--soft);margin-top:.25rem;}

/* ── rookie mistakes ── */
.mistake-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin:1rem 0;
}
.mistake-card{
  background:rgba(229,62,62,.08);border:1px solid rgba(229,62,62,.3);
  border-radius:7px;padding:.85rem 1rem;
}
.mistake-card .m-title{color:var(--red);font-weight:700;margin-bottom:.35rem;font-size:.88rem;}
.mistake-card p{font-size:.83rem;margin:0;}

/* ── param table ── */
.param-tag{color:var(--teal);font-weight:600;}

footer{
  text-align:center;padding:1.5rem;color:#4a5568;font-size:.8rem;
  border-top:1px solid var(--border);
}
"""

# Run the assembly module (shares this module's namespace)
exec(open(ROOT / "docs" / "_tut_assemble.py", encoding="utf-8").read())
