"""
kinematics.py
=============
Loading, quality control, signal preprocessing and the 12 canonical upper-limb
features, computed as per-frame traces so that a full 20-minute session can be
scrubbed and animated.

Everything is vectorised: a 40,000-frame session processes in roughly a second.

Note on traces vs. scalars: in the analysis these 12 features are *session-level
scalars* (an ROM is a range over the whole session). For the viewer they are
re-expressed as instantaneous, rolling or cumulative signals; `session_summary`
collapses each trace back to the scalar, using maximum minus minimum for the
angular and excursion features.
"""

import json
import os

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew

# =========================================================================
# configuration, mirroring the analysis pipeline
# =========================================================================
FPS = 30.0
FILTER_CUTOFF = 5.0        # Hz, low-pass
FILTER_ORDER = 4           # 4th order, zero-phase
MAX_VELOCITY_MS = 3.0      # m/s glitch threshold
MAX_GAP_FRAMES = 5         # gap interpolation threshold
ALCHEMY_STAGE = 7
CONF_KEEP_CODE = 1         # 0 = Low, 1 = Medium, 2 = High

# session-retention filters
MIN_FRAMES = 9000          # level 1a
MIN_ALCHEMY_PCT = 50.0     # level 1b
MAX_UNEXPECTED_PCT = 0.0   # level 1c
UNEXPECTED_PHASES = {"Calibration", "ChangePlayer", "Settings", "Loading"}

STAGE_NAMES = {0: "Loading", 1: "GameIntro", 2: "MainMenu", 3: "LarkRoom",
               4: "Settings", 5: "ChangePlayer", 6: "Calibration", 7: "Alchemy"}
STAGE_COLORS = {0: "#b0b0b0", 1: "#9fb8d4", 2: "#c8d6e5", 3: "#8fd0c4",
                4: "#d9b38c", 5: "#d9a0a0", 6: "#f0c674", 7: "#4a78b8"}

MOVEMENT_NAMES = {
    "EF": "Elbow flexion (continuous, 30-100 deg)",
    "HA": "Horizontal abduction (continuous, 0-70 deg)",
    "HR": "Head rotation (sequential, ~45 deg then neutral)",
    "TE": "Thoracic extension (sequential, ~190 deg then neutral)",
    "OC": "Hand open / close (sequential, 30-90 then 100-180 deg)",
    "REACH": "In-game nonspecific reaching (not a scored gesture)",
}

# EF, HA, HR, TE and OC are the gesture identifiers the game logs and scores.
# Reaching is not one of them: it is ordinary in-game arm movement between the
# scored gestures, so it is spelled out rather than abbreviated.
MOVEMENT_LABELS = {"EF": "EF", "HA": "HA", "HR": "HR", "TE": "TE", "OC": "OC",
                   "REACH": "in-game reaching"}
# muted so a long ribbon of movement blocks stays readable
MOVEMENT_COLORS = {"EF": "#d4969a", "HA": "#839cb7", "HR": "#a2bec7", "TE": "#97ae99", "OC": "#e4c199", "REACH": "#b0a4b9"}

SKELETON_EDGES = [
    ("Pelvis", "SpineNavel"), ("SpineNavel", "SpineChest"), ("SpineChest", "Neck"),
    ("Neck", "Head"),
    ("SpineChest", "ClavicleLeft"), ("ClavicleLeft", "ShoulderLeft"),
    ("ShoulderLeft", "ElbowLeft"), ("ElbowLeft", "WristLeft"),
    ("WristLeft", "HandLeft"), ("HandLeft", "HandTipLeft"), ("HandLeft", "ThumbLeft"),
    ("SpineChest", "ClavicleRight"), ("ClavicleRight", "ShoulderRight"),
    ("ShoulderRight", "ElbowRight"), ("ElbowRight", "WristRight"),
    ("WristRight", "HandRight"), ("HandRight", "HandTipRight"), ("HandRight", "ThumbRight"),
    ("Pelvis", "HipLeft"), ("HipLeft", "KneeLeft"), ("KneeLeft", "AnkleLeft"),
    ("AnkleLeft", "FootLeft"),
    ("Pelvis", "HipRight"), ("HipRight", "KneeRight"), ("KneeRight", "AnkleRight"),
    ("AnkleRight", "FootRight"),
]

JOINT_COLORS = {
    "Pelvis": "navy", "SpineNavel": "royalblue", "SpineChest": "cornflowerblue",
    "Neck": "dodgerblue", "Head": "lightsteelblue", "Nose": "skyblue",
    "EyeLeft": "deepskyblue", "EyeRight": "cyan", "EarLeft": "teal",
    "EarRight": "turquoise",
    "ClavicleLeft": "darkred", "ShoulderLeft": "firebrick", "ElbowLeft": "deeppink",
    "WristLeft": "indigo", "HandLeft": "darkorange", "HandTipLeft": "salmon",
    "ThumbLeft": "goldenrod",
    "ClavicleRight": "indianred", "ShoulderRight": "tomato", "ElbowRight": "pink",
    "WristRight": "orchid", "HandRight": "orange", "HandTipRight": "lightsalmon",
    "ThumbRight": "gold",
    "HipLeft": "darkgreen", "KneeLeft": "seagreen", "AnkleLeft": "mediumseagreen",
    "FootLeft": "lightgreen",
    "HipRight": "forestgreen", "KneeRight": "mediumaquamarine",
    "AnkleRight": "mediumspringgreen", "FootRight": "palegreen",
}
CONF_ALPHA = {0: 0.25, 1: 0.60, 2: 1.00}

ANGLE_JOINTS = ["ShoulderRight", "ShoulderLeft", "ElbowRight", "ElbowLeft",
                "WristRight", "WristLeft", "HandRight", "HandLeft",
                "SpineChest", "SpineNavel", "Neck", "Head", "EarLeft", "EarRight"]

# =========================================================================
# the 12 canonical features
# =========================================================================
TARGET_12_FEATURES = [
    "norm_jerk_R", "elbow_rom_R", "shoulder_flexext_rom_R", "shoulder_abdadd_rom_R",
    "shoulder_girdle_rom_R", "neck_rotation_rom", "neck_flexext_rom",
    "thoracic_ext_rom", "workspace_R_m3", "vel_hand_R_mean",
    "trunk_comp_R", "pct_above_shoulder_R",
]

FEATURE_DISPLAY_NAMES = {
    "norm_jerk_R": "Log Dim. Jerk (LDLJ)",
    "elbow_rom_R": "Elbow ROM (deg)",
    "shoulder_flexext_rom_R": "Shoulder Flex/Ext ROM (deg)",
    "shoulder_abdadd_rom_R": "Shoulder Abd/Add ROM (deg)",
    "shoulder_girdle_rom_R": "Shoulder Girdle ROM (normalised)",
    "neck_rotation_rom": "Neck Rotation ROM (deg)",
    "neck_flexext_rom": "Neck Flex/Ext ROM (deg)",
    "thoracic_ext_rom": "Thoracic Extension ROM (deg)",
    "workspace_R_m3": "Workspace Volume (m3)",
    "vel_hand_R_mean": "Mean Hand Speed (m/s)",
    "trunk_comp_R": "Trunk Compensation Ratio",
    "pct_above_shoulder_R": "% Time Above Shoulder",
}

FEATURE_DOMAIN = {
    "elbow_rom_R": "Joint kinematics", "shoulder_flexext_rom_R": "Joint kinematics",
    "shoulder_abdadd_rom_R": "Joint kinematics", "shoulder_girdle_rom_R": "Joint kinematics",
    "neck_rotation_rom": "Joint kinematics", "neck_flexext_rom": "Joint kinematics",
    "thoracic_ext_rom": "Joint kinematics",
    "workspace_R_m3": "Spatial kinematics",
    "vel_hand_R_mean": "Temporal kinematics", "norm_jerk_R": "Temporal kinematics",
    "trunk_comp_R": "Functional patterns", "pct_above_shoulder_R": "Functional patterns",
}

# Only the cumulative traces are annotated. The instantaneous and rolling ones
# are left unannotated on purpose, to be described in the accompanying text.
TRACE_MEANING = {
    "norm_jerk_R": "",
    "elbow_rom_R": "",
    "shoulder_flexext_rom_R": "",
    "shoulder_abdadd_rom_R": "",
    "shoulder_girdle_rom_R": "",
    "neck_rotation_rom": "",
    "neck_flexext_rom": "",
    "thoracic_ext_rom": "",
    "workspace_R_m3": "cumulative convex-hull volume of the hand path",
    "vel_hand_R_mean": "",
    "trunk_comp_R": "",
    "pct_above_shoulder_R": "cumulative % of frames with hand above shoulder",
}

ROM_LIKE = ["elbow_rom_R", "shoulder_flexext_rom_R", "shoulder_abdadd_rom_R",
            "shoulder_girdle_rom_R", "neck_rotation_rom", "neck_flexext_rom",
            "thoracic_ext_rom"]

# distinguishable but not fluorescent
# Thematic, by feature domain:
#   joint kinematics   a cool ramp, dark blue -> blue -> blue-grey -> green
#   spatial kinematics orange-red
#   temporal kinematics neutral greys
#   functional patterns ambers
# "Ocean Sunset": #001219 #005F73 #0A9396 #94D2BD #E9D8A6 #EE9B00 #CA6702
#                #BB3E03 #AE2012 #9B2226
#   joint kinematics    the four cool anchors, interpolated to seven steps
#                       running axial -> proximal -> distal
#   spatial kinematics  the sand, darkened for legibility on white
#   temporal kinematics the reds
#   functional patterns the oranges
FEATURE_COLORS = {
    "norm_jerk_R": "#9B2226",
    "elbow_rom_R": "#5bb8ad",
    "shoulder_flexext_rom_R": "#249f9d",
    "shoulder_abdadd_rom_R": "#08878e",
    "shoulder_girdle_rom_R": "#047380",
    "neck_rotation_rom": "#005d71",
    "neck_flexext_rom": "#003e4d",
    "thoracic_ext_rom": "#002029",
    "workspace_R_m3": "#C4A75F",
    "vel_hand_R_mean": "#AE2012",
    "trunk_comp_R": "#CA6702",
    "pct_above_shoulder_R": "#EE9B00",
}

# interpretation bands
ICC_BANDS = [(0.50, "poor"), (0.75, "moderate"), (0.90, "good"), (9e9, "excellent")]
CV_BANDS = [(15.0, "highly stable"), (30.0, "moderately stable"), (9e9, "unstable")]
RHO_BANDS = [(0.30, "negligible"), (0.50, "low"), (0.70, "moderate"),
             (0.90, "strong"), (9e9, "very strong")]


def band(value, bands):
    if value is None or not np.isfinite(value):
        return "n/a"
    for thr, name in bands:
        if abs(value) < thr:
            return name
    return bands[-1][1]


# =========================================================================
# loading
# =========================================================================
def load_npz(path):
    z = np.load(path, allow_pickle=False)
    rec = {k: z[k] for k in ("P", "stage", "nbody", "bodyid", "conf",
                             "flags", "ang", "mov", "intruder")}
    rec["meta"] = json.loads(str(z["meta"]))
    return rec


def load_json_gz(path):
    """Reads the full schema back. Slow and memory-hungry; the app uses npz."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


# =========================================================================
# session-level quality control
# =========================================================================
def qc_metrics(rec):
    stage = rec["stage"]
    n = len(stage)
    counts = {STAGE_NAMES.get(int(s), str(s)): int((stage == s).sum())
              for s in np.unique(stage)}
    alchemy_pct = 100.0 * counts.get("Alchemy", 0) / n
    unexpected_pct = 100.0 * sum(counts.get(p, 0) for p in UNEXPECTED_PHASES) / n
    multi_body_pct = 100.0 * float((rec["nbody"] != 1).mean())
    checks = [
        ("Duration", f"{n} frames ({n / FPS / 60:.1f} min)",
         f"at least {MIN_FRAMES} frames", n >= MIN_FRAMES),
        ("Gameplay share", f"{alchemy_pct:.1f} % Alchemy",
         f"at least {MIN_ALCHEMY_PCT:.0f} %", alchemy_pct >= MIN_ALCHEMY_PCT),
        ("Unexpected phases", f"{unexpected_pct:.1f} %",
         f"at most {MAX_UNEXPECTED_PCT:.0f} %", unexpected_pct <= MAX_UNEXPECTED_PCT),
    ]
    return dict(n_frames=n, minutes=n / FPS / 60, counts=counts,
                alchemy_pct=alchemy_pct, unexpected_pct=unexpected_pct,
                multi_body_pct=multi_body_pct, checks=checks,
                retained=all(c[3] for c in checks))


def stage_segments(stage):
    """Contiguous runs of the stage track: list of (start, end, stage_id)."""
    idx = np.flatnonzero(np.diff(stage)) + 1
    starts = np.concatenate([[0], idx])
    ends = np.concatenate([idx, [len(stage)]])
    return [(int(a), int(b), int(stage[a])) for a, b in zip(starts, ends)]


def gesture_events(rec):
    """Contiguous runs of each completed-movement flag."""
    flags = rec["flags"]
    names = rec["meta"]["gestureFlags"]
    out = {}
    for k, nm in enumerate(names):
        v = flags[:, k].astype(bool)
        if not v.any():
            continue
        d = np.diff(v.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1) + 1)
        if v[0]:
            starts = [0] + starts
        if v[-1]:
            ends = ends + [len(v)]
        out[nm] = list(zip(starts, ends))
    return out


# =========================================================================
# preprocessing
# =========================================================================
def _interp_nan_1d(col, max_gap=MAX_GAP_FRAMES):
    """Linear fill of NaN runs no longer than max_gap."""
    col = col.copy()
    nan = np.isnan(col)
    if not nan.any() or nan.all():
        return col
    idx = np.arange(len(col))
    d = np.diff(nan.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if nan[0]:
        starts = [0] + starts
    if nan[-1]:
        ends = ends + [len(col)]
    for a, b in zip(starts, ends):
        if b - a <= max_gap and a > 0 and b < len(col):
            col[a:b] = np.interp(idx[a:b], [a - 1, b], [col[a - 1], col[b]])
    return col


def clean_positions(pos):
    """pos: (N, 3) with NaN. Gap fill -> zero-phase low-pass -> glitch reject."""
    n = len(pos)
    out = pos.astype(np.float64).copy()
    if n < 3 * FILTER_ORDER:
        return out
    b, a = butter(FILTER_ORDER, FILTER_CUTOFF / (FPS / 2.0), btype="low")
    idx = np.arange(n)
    for k in range(3):
        col = out[:, k]
        good = ~np.isnan(col)
        if good.sum() < 2:
            continue
        col = _interp_nan_1d(col)
        still = np.isnan(col)
        tmp = col.copy()
        if still.any():
            tmp = np.interp(idx, idx[~still], col[~still])
        tmp = filtfilt(b, a, tmp)
        tmp[still] = np.nan
        out[:, k] = tmp
    vel = np.linalg.norm(np.diff(out, axis=0), axis=1) * FPS
    bad = np.concatenate([[False], vel > MAX_VELOCITY_MS])
    out[bad] = np.nan
    return out


def prepare(rec, alchemy_only=True):
    """
    Returns a dict with:
      keep   indices of the original frames that survive the frame-level filter
      raw    (M, 32, 3) metres, +y UP, every joint regardless of confidence
      conf   (M, 32) confidence codes
      J      cleaned joint arrays for the feature set, NaN where dropped
    """
    stage, nbody = rec["stage"], rec["nbody"]
    keep = nbody == 1
    if alchemy_only:
        keep &= stage == ALCHEMY_STAGE
    keep = np.flatnonzero(keep)

    P = rec["P"][keep].astype(np.float64)
    raw = np.empty_like(P)
    raw[:, :, 0] = P[:, :, 0] / 1000.0
    raw[:, :, 1] = -P[:, :, 1] / 1000.0        # +y becomes UP
    raw[:, :, 2] = P[:, :, 2] / 1000.0
    conf = rec["conf"][keep]

    names = rec["meta"]["joints"]
    jidx = {n: i for i, n in enumerate(names)}
    J = {}
    for nm in ANGLE_JOINTS:
        k = jidx[nm]
        arr = raw[:, k, :].copy()
        arr[conf[:, k] < CONF_KEEP_CODE] = np.nan   # confidence filter
        J[nm] = clean_positions(arr)
    return dict(keep=keep, raw=raw, conf=conf, J=J, jidx=jidx)


# =========================================================================
# geometry
# =========================================================================
def angle_between(u, v):
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = nu * nv
    cos = np.einsum("...i,...i->...", u, v) / np.where(denom == 0, np.nan, denom)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def signed_angle_in_plane(u, v, normal):
    cross = np.cross(u, v)
    sin = np.einsum("...i,...i->...", cross, normal) / (
        np.linalg.norm(normal, axis=-1) + 1e-12)
    cos = np.einsum("...i,...i->...", u, v)
    return np.degrees(np.arctan2(sin, cos))


def project_onto_plane(vec, normal):
    nn = np.einsum("...i,...i->...", normal, normal)
    scale = np.einsum("...i,...i->...", vec, normal) / np.where(nn == 0, np.nan, nn)
    return vec - scale[..., None] * normal


def _nan_rolling_sum(x, win):
    """Sum over a centred window, ignoring NaN. Vectorised via cumulative sums."""
    v = np.nan_to_num(x, nan=0.0)
    c = np.concatenate([[0.0], np.cumsum(v)])
    n = len(x)
    h = win // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return c[hi] - c[lo]


def _nan_rolling_count(x, win):
    m = (~np.isnan(x)).astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(m)])
    n = len(x)
    h = win // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return c[hi] - c[lo]


# =========================================================================
# per-frame feature traces
# =========================================================================
def compute_traces(prep, win_sec=4.0):
    J = prep["J"]
    sc, sn = J["SpineChest"], J["SpineNavel"]
    neck, head = J["Neck"], J["Head"]
    shR, shL = J["ShoulderRight"], J["ShoulderLeft"]
    elR, wrR, hR = J["ElbowRight"], J["WristRight"], J["HandRight"]
    earL, earR = J["EarLeft"], J["EarRight"]
    n = len(sc)
    win = max(3, int(win_sec * FPS) | 1)

    trunk_axis = sc - sn
    sh_axis = shR - shL
    sw = np.linalg.norm(sh_axis, axis=1)
    frontal_normal = np.cross(trunk_axis, sh_axis)
    upper_arm = elR - shR

    T = {}
    T["elbow_rom_R"] = angle_between(shR - elR, wrR - elR)
    T["shoulder_flexext_rom_R"] = angle_between(
        project_onto_plane(upper_arm, sh_axis), project_onto_plane(trunk_axis, sh_axis))
    T["shoulder_abdadd_rom_R"] = angle_between(
        project_onto_plane(upper_arm, frontal_normal),
        project_onto_plane(-trunk_axis, frontal_normal))
    T["shoulder_girdle_rom_R"] = (shR[:, 1] - sc[:, 1]) / np.where(sw == 0, np.nan, sw)

    ear_t = (earR - earL).copy()
    ear_t[:, 1] = 0.0
    sh_t = sh_axis.copy()
    sh_t[:, 1] = 0.0
    up = np.tile([0.0, 1.0, 0.0], (n, 1))
    T["neck_rotation_rom"] = np.abs(signed_angle_in_plane(sh_t, ear_t, up))
    T["neck_flexext_rom"] = angle_between(head - neck, trunk_axis)
    T["thoracic_ext_rom"] = angle_between(neck - sc, sc - sn)

    # workspace volume: incremental convex hull of the hand path
    T["workspace_R_m3"] = _cumulative_hull_volume(hR, chunk=int(FPS))

    # hand speed
    sp = np.full(n, np.nan)
    sp[1:] = np.linalg.norm(np.diff(hR, axis=0), axis=1) * FPS
    T["vel_hand_R_mean"] = sp

    # rolling log dimensionless jerk
    T["norm_jerk_R"] = _rolling_ldlj(hR, win)

    # rolling trunk compensation ratio
    d_tr = np.concatenate([[np.nan], np.linalg.norm(np.diff(sc, axis=0), axis=1)])
    d_hd = np.concatenate([[np.nan], np.linalg.norm(np.diff(hR, axis=0), axis=1)])
    tm = _nan_rolling_sum(d_tr, win)
    hm = _nan_rolling_sum(d_hd, win)
    T["trunk_comp_R"] = np.where(hm > 1e-9, tm / np.where(hm == 0, np.nan, hm), np.nan)

    # cumulative % of frames with the hand above the shoulder
    valid = ~np.isnan(hR[:, 1]) & ~np.isnan(shR[:, 1])
    above = np.zeros(n, bool)
    above[valid] = (hR[valid, 1] - shR[valid, 1]) > 0
    T["pct_above_shoulder_R"] = 100.0 * np.cumsum(above) / np.maximum(
        np.cumsum(valid), 1)
    return T


def _cumulative_hull_volume(pos, chunk=30):
    """Expanding convex-hull volume of the hand trajectory, updated every chunk."""
    n = len(pos)
    out = np.full(n, np.nan)
    good = ~np.isnan(pos).any(axis=1)
    hull, last, seen = None, 0.0, []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        pts = pos[start:stop][good[start:stop]]
        if len(pts):
            seen.append(pts)
            try:
                if hull is None:
                    allp = np.vstack(seen)
                    if len(allp) >= 8:
                        hull = ConvexHull(allp, incremental=True)
                        last = float(hull.volume)
                else:
                    hull.add_points(pts)
                    last = float(hull.volume)
            except Exception:
                pass
        out[start:stop] = last
    if hull is not None:
        hull.close()
    return out


def _rolling_ldlj(pos, win):
    """Log dimensionless jerk over a centred sliding window."""
    n = len(pos)
    dt = 1.0 / FPS
    filled = pos.copy()
    for k in range(3):
        col = filled[:, k]
        m = np.isnan(col)
        if m.all():
            return np.full(n, np.nan)
        if m.any():
            col[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), col[~m])
    vel = np.gradient(filled, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    j2 = np.sum(jerk ** 2, axis=1)
    speed = np.linalg.norm(vel, axis=1)

    integ = _nan_rolling_sum(j2, win) * dt
    cnt = _nan_rolling_count(j2, win)
    dur = cnt * dt
    peak = maximum_filter1d(speed, size=win, mode="nearest")
    val = np.where((peak > 1e-6) & (cnt > 10),
                   (dur ** 3) / np.where(peak <= 0, np.nan, peak ** 2) * integ, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = -np.log(np.where(val > 0, val, np.nan))
    return out


# =========================================================================
# session-level scalars and distributions
# =========================================================================
def session_summary(prep, traces):
    """
    One scalar per feature, computed the way the analysis pipeline does it:
      angular / excursion features   maximum minus minimum
      workspace                      convex-hull volume of the whole hand path
      hand speed                     mean instantaneous speed
      trunk compensation             total trunk path / total hand path
      % above shoulder               share of valid frames
      LDLJ                           log dimensionless jerk over the full trajectory
    """
    out = {}
    for k in ROM_LIKE:
        v = traces[k]
        v = v[np.isfinite(v)]
        out[k] = float(v.max() - v.min()) if v.size else np.nan

    hR, sc, shR = prep["J"]["HandRight"], prep["J"]["SpineChest"], prep["J"]["ShoulderRight"]

    w = traces["workspace_R_m3"]
    w = w[np.isfinite(w)]
    out["workspace_R_m3"] = float(w[-1]) if w.size else np.nan

    sp = np.linalg.norm(np.diff(hR, axis=0), axis=1) * FPS
    out["vel_hand_R_mean"] = float(np.nanmean(sp)) if np.isfinite(sp).any() else np.nan

    trunk_move = float(np.nansum(np.linalg.norm(np.diff(sc, axis=0), axis=1)))
    hand_move = float(np.nansum(np.linalg.norm(np.diff(hR, axis=0), axis=1)))
    out["trunk_comp_R"] = trunk_move / hand_move if hand_move > 1e-9 else np.nan

    valid = np.isfinite(hR[:, 1]) & np.isfinite(shR[:, 1])
    out["pct_above_shoulder_R"] = (
        float(100.0 * np.mean((hR[valid, 1] - shR[valid, 1]) > 0)) if valid.any() else np.nan)

    out["norm_jerk_R"] = _full_ldlj(hR)
    return out


def _full_ldlj(pos):
    """Log dimensionless jerk over the entire trajectory."""
    m = ~np.isnan(pos).any(axis=1)
    if m.sum() < 30:
        return np.nan
    p = pos[m]
    dt = 1.0 / FPS
    vel = np.gradient(p, dt, axis=0)
    jerk = np.gradient(np.gradient(vel, dt, axis=0), dt, axis=0)
    dur = len(p) * dt
    peak = float(np.max(np.linalg.norm(vel, axis=1)))
    integ = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt)
    val = (dur ** 3) / (peak ** 2) * integ if peak > 1e-9 else 0.0
    return float(-np.log(val)) if val > 0 else np.nan


def distribution_table(traces):
    """Per-session distribution of every feature, as specified in the methods."""
    rows = []
    for k in TARGET_12_FEATURES:
        v = traces[k]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        p = np.percentile(v, [5, 10, 25, 75, 90, 95])
        rows.append({
            "Feature": FEATURE_DISPLAY_NAMES[k], "Domain": FEATURE_DOMAIN[k],
            "Median": np.median(v), "Mean": v.mean(), "SD": v.std(ddof=1),
            "5th": p[0], "10th": p[1], "25th": p[2],
            "75th": p[3], "90th": p[4], "95th": p[5],
            # skewness and kurtosis are undefined for a constant trace
            "Skewness": float(skew(v)) if v.std() > 1e-12 else np.nan,
            "Kurtosis": float(kurtosis(v)) if v.std() > 1e-12 else np.nan,
            "Range (max-min)": float(v.max() - v.min()),
        })
    return rows


def cv_percent(a, b):
    """Coefficient of variation across two session values of the same feature."""
    vals = np.array([a, b], float)
    if not np.all(np.isfinite(vals)) or abs(vals.mean()) < 1e-12:
        return np.nan
    return float(100.0 * vals.std(ddof=1) / abs(vals.mean()))

# =========================================================================
# reading real recordings from a local folder
# =========================================================================
LOCAL_SUFFIXES = (".zip", ".json", ".json.gz")


def list_local_sessions(folder):
    """Every recording file in a folder, sorted. Accepts .zip, .json, .json.gz."""
    import glob as _glob
    out = []
    for pat in ("*.zip", "*.json", "*.json.gz"):
        out += _glob.glob(os.path.join(folder, "**", pat), recursive=True)
    return sorted(p for p in out if not p.endswith("_excerpt.json"))


def session_key(path):
    """A short, stable label for a recording file."""
    base = os.path.basename(path)
    for suf in (".json.gz", ".json", ".zip", ".npz"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def read_recording(path):
    """Loads one recording, whether it is a .zip containing a .json, a plain
    .json, or a .json.gz. Returns the parsed dictionary."""
    import gzip
    import zipfile
    low = path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not names:
                return None
            with zf.open(names[0]) as f:
                return json.load(f)
    if low.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _label_index(rec):
    """Maps every field name in labelsNames to its position in the labels string."""
    names = [n for n in str(rec.get("labelsNames", "")).split(";")]
    return {n: i for i, n in enumerate(names) if n}


def recording_to_arrays(rec, name="session", patient_map=None, fps=FPS):
    """
    Converts a parsed recording into exactly the array bundle the viewer uses,
    so real and synthetic sessions travel through identical code.

    Returns P (N,32,3) mm with +y down, stage, nbody, bodyid, conf, flags, ang,
    mov, intruder, meta.
    """
    frames = rec.get("frameData", []) or []
    N = len(frames)
    if N == 0:
        raise ValueError("recording contains no frameData")

    joint_names = [j.get("jointType") for j in frames[0].get("joints", [])]
    if not joint_names:
        raise ValueError("first frame contains no joints")
    jidx = {nm: i for i, nm in enumerate(joint_names)}
    nj = len(joint_names)

    lab_idx = _label_index(rec)
    flag_names = [n for n in lab_idx if "-" in n and len(n) <= 5]
    known = [n for n in ["HA-L", "HA-R", "OC-L", "OC-R", "TE",
                         "HR-L", "HR-R", "EF-L", "EF-R"] if n in lab_idx]
    flag_names = known or flag_names
    ang_names = [n for n in lab_idx if n.count("-") == 1 and "_" in n]

    P = np.full((N, nj, 3), np.nan, np.float32)
    conf = np.zeros((N, nj), np.uint8)
    stage = np.full(N, -1, np.int16)
    nbody = np.zeros(N, np.int8)
    bodyid = np.zeros(N, np.int8)
    flags = np.zeros((N, len(flag_names)), np.int8)
    ang = np.zeros((N, len(ang_names)), np.float32)

    conf_code = {"Low": 0, "Medium": 1, "High": 2}
    for i, fr in enumerate(frames):
        parts = str(fr.get("labels", "")).split(";")

        def fld(key, cast, default):
            k = lab_idx.get(key)
            if k is None or k >= len(parts):
                return default
            try:
                return cast(float(parts[k]))
            except (ValueError, TypeError):
                return default

        stage[i] = fld("stageId", int, -1)
        nbody[i] = fld("numberOfBodiesFound", int, 0)
        bodyid[i] = fld("bodyId", int, 0)
        for k, nm in enumerate(flag_names):
            flags[i, k] = fld(nm, int, 0)
        for k, nm in enumerate(ang_names):
            ang[i, k] = fld(nm, float, np.nan)

        for j in fr.get("joints", []):
            k = jidx.get(j.get("jointType"))
            if k is None:
                continue
            pos = j.get("position", {})
            P[i, k, 0] = pos.get("x", np.nan)
            P[i, k, 1] = pos.get("y", np.nan)
            P[i, k, 2] = pos.get("z", np.nan)
            conf[i, k] = conf_code.get(j.get("confidenceLevel"), 0)

    pid = rec.get("patientId", "")
    label = (patient_map or {}).get(pid, name)

    meta = {
        "name": name, "label": label, "fps": fps, "syntheticData": False,
        "profileLabel": f"local recording ({label})",
        "joints": joint_names,
        "movements": list(MOVEMENT_NAMES.keys()),
        "gestureFlags": flag_names,
        "angleChannels": ang_names,
        "stageNames": {str(k): v for k, v in STAGE_NAMES.items()},
        "demoClinical": {},
    }
    # real recordings carry no movement-block track; the completed-movement
    # flags stand in for it
    mov = np.full(N, -1, np.int8)
    return dict(P=P, stage=stage, nbody=nbody, bodyid=bodyid, conf=conf,
                flags=flags, ang=ang, mov=mov,
                intruder=np.zeros(N, bool), meta=meta)


def cache_recording(src_path, cache_dir, patient_map=None, force=False):
    """Converts one local recording to a .npz next to the others. Returns its path."""
    os.makedirs(cache_dir, exist_ok=True)
    key = session_key(src_path)
    dst = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(dst) and not force:
        return dst
    rec = read_recording(src_path)
    if rec is None:
        raise ValueError(f"no JSON payload inside {src_path}")
    arr = recording_to_arrays(rec, name=key, patient_map=patient_map)
    np.savez_compressed(
        dst, P=arr["P"], stage=arr["stage"], nbody=arr["nbody"],
        bodyid=arr["bodyid"], conf=arr["conf"], flags=arr["flags"],
        ang=arr["ang"], mov=arr["mov"], intruder=arr["intruder"],
        meta=np.array(json.dumps(arr["meta"])))
    return dst


def blocks_from_flags(flags, flag_names):
    """Movement blocks inferred from the completed-movement flags."""
    out = []
    for k, nm in enumerate(flag_names):
        base = nm.split("-")[0]
        for a, b, on in stage_segments(flags[:, k].astype(np.int8)):
            if on:
                out.append((a, b, base))
    return sorted(out)
