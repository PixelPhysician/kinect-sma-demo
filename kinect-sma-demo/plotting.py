"""
plotting.py — all matplotlib rendering, kept out of app.py so it can be tested
without a Streamlit runtime.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import kinematics as K

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "legend.frameon": False,
})

INK = "#1f2733"
MUTED = "#8a93a0"

# the whole-recording charts are stacked, so their plotting areas are pinned to
# the same figure fractions and they line up frame for frame
FIG_W = 13.2
AX_LEFT, AX_RIGHT = 0.105, 0.988   # room for the "Completed" label


def _proj(view, xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if view == "front":
        return x, y
    if view == "side":
        return z, y
    return x, z


# =========================================================================
# skeleton
# =========================================================================
def draw_skeleton(pos, conf, cube, joints, title="", show_labels=False):
    """pos (32,3) metres with +y up, conf (32,) confidence codes."""
    jidx = {n: i for i, n in enumerate(joints)}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    views = [("front", "Front view  (x vs -y)", "Left  <-  x  ->  Right",
              "Down  <-  -y  ->  Up", "x", "y"),
             ("side", "Side view  (z vs -y)", "Front  <-  z  ->  Back",
              "Down  <-  -y  ->  Up", "z", "y"),
             ("top", "Top view  (x vs z)", "Left  <-  x  ->  Right",
              "Back  <-  z  ->  Front", "x", "z")]

    for ax, (view, ttl, xl, yl, ax_x, ax_y) in zip(axes, views):
        # bones first -> background
        for a, b in K.SKELETON_EDGES:
            if a in jidx and b in jidx:
                xa, ya = _proj(view, pos[jidx[a]])
                xb, yb = _proj(view, pos[jidx[b]])
                ax.plot([xa, xb], [ya, yb], color=K.JOINT_COLORS.get(a, "black"),
                        alpha=0.55, linewidth=2.1, zorder=1, solid_capstyle="round")
        # joints on top -> foreground, opacity = tracking confidence
        for nm, k in jidx.items():
            px, py = _proj(view, pos[k])
            ax.scatter(px, py, s=32, zorder=3, edgecolors="none",
                       color=K.JOINT_COLORS.get(nm, "black"),
                       alpha=K.CONF_ALPHA.get(int(conf[k]), 0.25))
            if show_labels and view == "front" and nm in (
                    "Head", "ShoulderRight", "ElbowRight", "HandRight", "Pelvis"):
                ax.annotate(nm, (px, py), fontsize=6.5, color=MUTED,
                            xytext=(4, 3), textcoords="offset points")

        ax.set_xlim(cube[ax_x])
        ax.set_ylim(cube[ax_y])
        ax.set_aspect("equal")
        ax.set_title(ttl, fontweight="bold", color=INK)
        ax.set_xlabel(xl, color=MUTED)
        ax.set_ylabel(yl, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)

    if title:
        fig.suptitle(title, fontsize=10.5, color=INK)
    fig.tight_layout()
    return fig


# =========================================================================
# timeline: game phase ribbon + movement ribbon + completed-movement raster
# =========================================================================
def draw_timeline(stage, mov, movements, flags, flag_names, frame=None,
                  nbody=None, height=3.05, window=None):
    """Game phase along the top, completed repetitions below.

    `mov` is still accepted so callers do not change, but the movement-block
    ribbon is no longer drawn: the completed-repetition raster carries the same
    information and reads more cleanly over a twenty-minute recording.
    """
    n = len(stage)
    rows = 2 if flags is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(FIG_W, height),
                             gridspec_kw={"height_ratios": [1, 2.6][:rows]},
                             sharex=True)
    axes = np.atleast_1d(axes)

    # --- game phase ---
    ax = axes[0]
    for a, b, s in K.stage_segments(stage):
        ax.axvspan(a, b, color=K.STAGE_COLORS.get(s, "#cccccc"), lw=0)
    if nbody is not None:
        bad = nbody != 1
        if bad.any():
            for a, b, v in K.stage_segments(bad.astype(np.int8)):
                if v:
                    ax.axvspan(a, b, color="#d62728", alpha=0.95, lw=0)
    ax.set_yticks([])
    ax.set_ylabel("Phase", rotation=0, ha="right", va="center", color=MUTED)
    ax.grid(False)

    # --- completed-repetition raster ---
    if flags is not None:
        ax = axes[1]
        for i, nm in enumerate(flag_names):
            v = flags[:, i].astype(bool)
            if not v.any():
                continue
            col = K.MOVEMENT_COLORS.get(nm.split("-")[0], "#666666")
            runs = [(a, b) for a, b, on in K.stage_segments(v.astype(np.int8)) if on]
            if not runs:
                continue
            # a scored repetition lasts a fraction of a second out of tens of
            # thousands of frames, so a bar alone is sub-pixel wide: the bar keeps
            # the duration and a diamond guarantees the event is visible
            for a, b in runs:
                ax.plot([a, b], [i, i], lw=3.0, solid_capstyle="butt",
                        color=col, alpha=0.9, zorder=2)
            mids = [(a + b) / 2 for a, b in runs]
            ax.scatter(mids, [i] * len(mids), marker="D", s=26, color=col,
                       alpha=0.95, edgecolors=col, linewidths=0.5, zorder=3)
        ax.set_yticks(range(len(flag_names)))
        ax.set_yticklabels(flag_names, fontsize=7)
        ax.set_ylim(-0.8, len(flag_names) - 0.2)
        ax.set_ylabel("Completed", rotation=0, ha="right", va="center", color=MUTED)
        ax.grid(axis="x", alpha=0.2)

    for ax in axes:
        if window is not None:
            ax.axvspan(window[0], window[1], facecolor="none", edgecolor=INK,
                       lw=1.4, zorder=7)
            ax.axvspan(window[0], window[1], color="#1f2733", alpha=0.10, zorder=6)
        if frame is not None:
            ax.axvline(frame, color=INK, lw=1.3, zorder=8)
        ax.set_xlim(0, n - 1)
        ax.tick_params(labelsize=7, colors=MUTED)
    axes[-1].set_xlabel("Frame", color=MUTED)

    phase_h = [mpatches.Patch(color=K.STAGE_COLORS[s], label=K.STAGE_NAMES[s])
               for s in sorted(set(int(v) for v in np.unique(stage)))]
    if nbody is not None and (nbody != 1).any():
        phase_h.append(mpatches.Patch(color="#d62728", label="second body in frame"))
    scored = {nm.split("-")[0] for nm in (flag_names or [])}
    mov_h = [mpatches.Patch(color=K.MOVEMENT_COLORS[m],
                            label=K.MOVEMENT_LABELS.get(m, m))
             for m in movements if m in K.MOVEMENT_COLORS and m in scored]

    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, top=0.775, bottom=0.155,
                        hspace=0.18)
    for hs, y in ((phase_h, 0.925), (mov_h, 0.845)):
        if hs:
            fig.legend(handles=hs, ncol=len(hs), fontsize=6.5, frameon=False,
                       loc="lower left", bbox_to_anchor=(AX_LEFT, y),
                       bbox_transform=fig.transFigure, columnspacing=1.4,
                       handlelength=1.5, handletextpad=0.45)
    return fig


# =========================================================================
# feature overlay
# =========================================================================
def draw_features(traces, selected, frame=None, normalise=True, max_points=2500,
                  window=None, span=None):
    rows_leg = int(np.ceil(len(selected) / 4)) if selected else 1
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6 + 0.16 * rows_leg))
    if not selected:
        ax.text(0.5, 0.5, "Select one or more features above",
                ha="center", va="center", color=MUTED, fontsize=11)
        ax.set_axis_off()
        fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT)
        return fig

    n = len(next(iter(traces.values())))
    lo, hi = (0, n) if window is None else window
    lo, hi = max(0, int(lo)), min(n, int(hi))
    step = max(1, (hi - lo) // max_points)
    xs = np.arange(lo, hi, step)

    for key in selected:
        v = np.asarray(traces[key], float)
        seg = v[lo:hi:step]
        if normalise:
            a, b = np.nanmin(v), np.nanmax(v)
            seg = (seg - a) / (b - a) if np.isfinite(b - a) and b - a > 1e-12 \
                else np.zeros_like(seg)
        if frame is None:
            lbl = K.FEATURE_DISPLAY_NAMES[key]
        else:
            cur = v[min(frame, n - 1)]
            lbl = (f"{K.FEATURE_DISPLAY_NAMES[key]} = {cur:.3f}"
                   if np.isfinite(cur) else f"{K.FEATURE_DISPLAY_NAMES[key]} = n/a")
        c = K.FEATURE_COLORS[key]
        ax.plot(xs, seg, color=c, lw=1.15, alpha=0.9, label=lbl, zorder=2)
        if frame is not None and lo <= frame < hi:
            yv = v[frame]
            if normalise and np.isfinite(yv):
                a, b = np.nanmin(v), np.nanmax(v)
                yv = (yv - a) / (b - a) if b - a > 1e-12 else 0.0
            if np.isfinite(yv):
                ax.scatter([frame], [yv], s=44, color=c, zorder=5,
                           edgecolors="white", linewidths=0.9)

    if frame is not None:
        ax.axvline(frame, color=INK, lw=1.1, ls="--", alpha=0.85, zorder=4)
    if span is not None:
        ax.axvspan(span[0], span[1], color="#1f2733", alpha=0.10, zorder=1)
    ax.set_xlim(lo, hi - 1 if hi > lo else lo + 1)
    ax.set_xlabel("Frame", color=MUTED)
    ax.set_ylabel("Normalised 0-1" if normalise else "Raw value", color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=4, frameon=False, borderaxespad=0, columnspacing=1.6,
              handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, bottom=0.15,
                        top=0.90 - 0.045 * rows_leg)
    return fig


# =========================================================================
# phase composition
# =========================================================================
def draw_stage_breakdown(counts, total):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    vals = [100.0 * v / total for _, v in items]
    cols = [K.STAGE_COLORS.get(
        next((s for s, nm in K.STAGE_NAMES.items() if nm == n), 7), "#999") for n in names]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.62)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.8, i, f"{v:.1f} %", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("Share of recorded frames (%)", color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def draw_comparison(summaries, feature, order):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    vals = [summaries[s].get(feature, np.nan) for s in order]
    cols = ["#4a78b8" if s.startswith("P02") else "#c0504d" for s in order]
    ax.bar(order, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5,
                    color=MUTED)
    ax.set_title(K.FEATURE_DISPLAY_NAMES.get(feature, feature), fontweight="bold",
                 color=INK)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig
