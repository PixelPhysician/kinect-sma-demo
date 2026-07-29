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
def draw_timeline(stage, mov, movements, flags, flag_names, frame,
                  nbody=None, height=3.0):
    n = len(stage)
    rows = 3 if flags is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(13.2, height),
                             gridspec_kw={"height_ratios": [1, 1, 2.4][:rows]},
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

    # --- movement block ---
    ax = axes[1]
    for a, b, m in K.stage_segments(mov.astype(np.int16)):
        if m >= 0:
            name = movements[m]
            ax.axvspan(a, b, color=K.MOVEMENT_COLORS.get(name, "#999999"),
                       alpha=0.85, lw=0)
    ax.set_yticks([])
    ax.set_ylabel("Movement", rotation=0, ha="right", va="center", color=MUTED)
    ax.grid(False)

    # --- completed-movement raster ---
    if flags is not None:
        ax = axes[2]
        for i, nm in enumerate(flag_names):
            v = flags[:, i].astype(bool)
            if not v.any():
                continue
            base = nm.split("-")[0]
            for a, b, on in K.stage_segments(v.astype(np.int8)):
                if on:
                    ax.plot([a, b], [i, i], lw=4.2, solid_capstyle="butt",
                            color=K.MOVEMENT_COLORS.get(base, "#666666"), alpha=0.9)
        ax.set_yticks(range(len(flag_names)))
        ax.set_yticklabels(flag_names, fontsize=6.5)
        ax.set_ylim(-0.8, len(flag_names) - 0.2)
        ax.set_ylabel("Completed", rotation=0, ha="right", va="center", color=MUTED)
        ax.grid(axis="x", alpha=0.2)

    for ax in axes:
        ax.axvline(frame, color=INK, lw=1.3, zorder=6)
        ax.set_xlim(0, n - 1)
        ax.tick_params(labelsize=7, colors=MUTED)
    axes[-1].set_xlabel("Frame", color=MUTED)

    handles = [mpatches.Patch(color=K.STAGE_COLORS[s], label=K.STAGE_NAMES[s])
               for s in sorted(set(int(v) for v in np.unique(stage)))]
    handles += [mpatches.Patch(color=K.MOVEMENT_COLORS[m], label=m)
                for m in movements]
    if nbody is not None and (nbody != 1).any():
        handles.append(mpatches.Patch(color="#d62728", label="second body in frame"))
    axes[0].legend(handles=handles, ncol=8, fontsize=6.5,
                   loc="lower left", bbox_to_anchor=(0, 1.05))
    fig.tight_layout()
    return fig


# =========================================================================
# feature overlay
# =========================================================================
def draw_features(traces, selected, frame, normalise=True, max_points=2500,
                  window=None):
    fig, ax = plt.subplots(figsize=(13.2, 4.0))
    if not selected:
        ax.text(0.5, 0.5, "Select one or more features in the sidebar",
                ha="center", va="center", color=MUTED, fontsize=11)
        ax.set_axis_off()
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
        cur = v[min(frame, n - 1)]
        lbl = (f"{K.FEATURE_DISPLAY_NAMES[key]} = {cur:.3f}"
               if np.isfinite(cur) else f"{K.FEATURE_DISPLAY_NAMES[key]} = n/a")
        c = K.FEATURE_COLORS[key]
        ax.plot(xs, seg, color=c, lw=1.15, alpha=0.9, label=lbl, zorder=2)
        if lo <= frame < hi:
            yv = v[frame]
            if normalise and np.isfinite(yv):
                a, b = np.nanmin(v), np.nanmax(v)
                yv = (yv - a) / (b - a) if b - a > 1e-12 else 0.0
            if np.isfinite(yv):
                ax.scatter([frame], [yv], s=44, color=c, zorder=5,
                           edgecolors="white", linewidths=0.9)

    ax.axvline(frame, color=INK, lw=1.1, ls="--", alpha=0.85, zorder=4)
    ax.set_xlim(lo, hi - 1 if hi > lo else lo + 1)
    ax.set_xlabel("Frame (gameplay frames, 30 Hz)", color=MUTED)
    ax.set_ylabel("Normalised 0-1" if normalise else "Raw value", color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
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
