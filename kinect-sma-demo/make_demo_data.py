"""
make_demo_data.py
=================
Generates five fully SYNTHETIC Azure Kinect DK session recordings in the KEEP /
"Tales from the Magic Keep" JSON schema, following the physiotherapy movement
protocol (EF, HA, HR, TE, OC) performed from the neutral resting position
(sitting upright, arms relaxed on the thighs).

    data/P01_S1   ~19 min   weaker motor profile   -> retained
    data/P01_S2   ~22 min   weaker motor profile   -> retained
    data/P02_S1   ~19 min   stronger motor profile -> retained
    data/P02_S2   ~24 min   stronger motor profile -> retained
    data/BAD_S1   ~2.5 min  deliberately faulty    -> excluded by all 3 filters

The faulty session reproduces three realistic failure modes at once:
  * far too short (below the minimum-frame threshold),
  * Loading / Calibration frames present (unexpected phases),
  * a stretch where a second person walks into the frame and body tracking
    latches onto them instead of the participant (nbody = 2, bodyId flips).

Two artefacts are written per session:
  <name>.json.gz  full schema, gzip-compressed real JSON
  <name>.npz      compact float32 arrays that the Streamlit app reads

An uncompressed 20-minute session would be roughly 200 MB, which is why the
JSON is gzipped. `json.load(gzip.open(path, "rt"))` reads it back unchanged.

NO REAL PARTICIPANT DATA, IDENTIFIERS OR CLINICAL SCORES ARE USED ANYWHERE.

Usage
-----
    python make_demo_data.py              # npz + json.gz  (full artefacts)
    python make_demo_data.py --npz-only   # npz only       (fast, what the app needs)
    python make_demo_data.py --minutes 4  # short sessions, for quick testing
"""

import argparse
import gzip
import json
import os

import numpy as np

# =========================================================================
# constants
# =========================================================================
FPS = 30.0

JOINTS_32 = [
    "Pelvis", "SpineNavel", "SpineChest", "Neck", "Head", "Nose",
    "EyeLeft", "EarLeft", "EyeRight", "EarRight",
    "ClavicleLeft", "ShoulderLeft", "ElbowLeft", "WristLeft",
    "HandLeft", "HandTipLeft", "ThumbLeft",
    "ClavicleRight", "ShoulderRight", "ElbowRight", "WristRight",
    "HandRight", "HandTipRight", "ThumbRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
]
JIDX = {n: i for i, n in enumerate(JOINTS_32)}

STAGE_NAMES = {0: "Loading", 1: "GameIntro", 2: "MainMenu", 3: "LarkRoom",
               4: "Settings", 5: "ChangePlayer", 6: "Calibration", 7: "Alchemy"}
STAGES_NAMES_STR = ("0-Loading;1-GameIntro;2-MainMenu;3-LarkRoom;4-Settings;"
                    "5-ChangePlayer;6-Calibration;7-Alchemy;"
                    "1000-EditorDebugCutscenes;-1-None;")

# the 15 continuous comparison channels, in label order
ANGLE_CHANNELS = [
    "OC_Right-OC_HandRightToHandTipRightRotation",
    "OC_Right-OC_HandTipsToWristRightLength",
    "OC_Left-OC_HandLeftToHandTipLeftRotation",
    "OC_Left-OC_HandTipsToWristLeftLength",
    "HA_Right-HA_RightArmForwardToSpineForwardRotation",
    "HA_Right-HA_ArmRightToDownRotation",
    "HA_Right-HA_ForearmRightToSpineUpRotation",
    "HA_Left-HA_LeftArmForwardToSpineForwardRotation",
    "HA_Left-HA_ArmLeftToDownRotation",
    "HA_Left-HA_ForearmLeftToSpineUpRotation",
    "EF_Right-EF_ArmRightToForeamrRightRotation",
    "EF_Left-EF_ArmLeftToForeamrLeftRotation",
    "HR_Right-HR_HeadForwardToSpineForwardRotation",
    "HR_Left-HR_HeadForwardToSpineForwardRotation",
    "TE_Undefined-TE_LeftShoulderToRightShoulderRotation",
]
# the 9 per-frame movement-completion counters, in label order
GESTURE_FLAGS = ["HA-L", "HA-R", "OC-L", "OC-R", "TE", "HR-L", "HR-R", "EF-L", "EF-R"]

LABELS_NAMES = ("stageId;numberOfBodiesFound;bodyId;timeZoneInformation;kinectTimestamp;"
                "systemTimestamp;unityTimestamp;" + ";".join(ANGLE_CHANNELS) + ";"
                + ";".join(GESTURE_FLAGS) + ";")

LOW_CONF_JOINTS = {"ThumbLeft", "ThumbRight", "FootLeft", "FootRight",
                   "AnkleLeft", "AnkleRight", "KneeLeft", "KneeRight"}
CONF_CODE = {"Low": 0, "Medium": 1, "High": 2}
CONF_NAME = {v: k for k, v in CONF_CODE.items()}

# --------------------------------------------------------------------------
# movement protocol targets, from the movement definitions
#   EF  elbow flexion         continuous, target 30-100 deg
#   HA  horizontal abduction  continuous, arm-forward vs spine-forward 0-70 deg,
#                             upper arm vs spinal down 30-80 deg,
#                             forearm vs spinal up 77-113 deg (near horizontal)
#   HR  head rotation         sequential, phase 1 approx 45 deg, phase 2 back to 0
#   TE  thoracic extension    sequential, reach approx 190 deg then return
#   OC  hand open / close     sequential, fist 30-90 deg then opening 100-180 deg
#   REACH  lateral reaching
# --------------------------------------------------------------------------
MOVEMENTS = ["EF", "HA", "HR", "TE", "OC", "REACH"]

# --------------------------------------------------------------------------
# two synthetic motor profiles (NOT real clinical values)
# --------------------------------------------------------------------------
PROFILES = {
    "P01": dict(
        label="Weaker motor profile (synthetic)",
        sma_type="II", functional="sitting", hfmse=15.0, rulm=11.0,
        capability=0.42,          # 0 = minimal, 1 = full protocol amplitude
        rep_hz=0.30,              # repetitions per second
        trunk_gain=3.4,           # trunk compensation multiplier
        tremor_mm=2.6, noise_mm=3.2, drop_rate=0.030, stature=1.00,
        success_rate=0.45,        # fraction of repetitions reaching the target range
    ),
    "P02": dict(
        label="Stronger motor profile (synthetic)",
        sma_type="III", functional="walking_assisted", hfmse=34.0, rulm=25.0,
        capability=0.92, rep_hz=0.45, trunk_gain=0.40,
        tremor_mm=2.2, noise_mm=3.0, drop_rate=0.012, stature=1.12,
        success_rate=0.88,
    ),
}

SESSION_PLAN = [
    # (name, profile, minutes, seed, faulty, mid-session returns to the room)
    ("P01_S1", "P01", 18.6, 1101, False, 0),
    ("P01_S2", "P01", 21.0, 1102, False, 1),   # this one child took a break
    ("P02_S1", "P02", 17.8, 2201, False, 0),
    ("P02_S2", "P02", 22.8, 2202, False, 0),
    ("BAD_S1", "P01", 2.4, 9099, True, 0),
]


# =========================================================================
# vector helpers, all operating on (N, 3) arrays
# =========================================================================
def _u(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def _ang(u, v):
    c = np.einsum("ij,ij->i", _u(u), _u(v))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _smooth(rng, n, scale, hz):
    """Band-limited wobble built from a handful of random sinusoids."""
    t = np.arange(n) / FPS
    out = np.zeros(n)
    for f in (hz * 0.4, hz, hz * 2.3, hz * 4.7):
        out += rng.normal(0, 1) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out * scale / 2.0


def _ramp(n, lo, hi, hz, rng, phase=0.0):
    """Smooth repeated excursion between lo and hi (used for calibration sweeps)."""
    t = np.arange(n) / FPS
    jitter = _smooth(rng, n, 0.45, max(hz * 0.5, 0.05))
    c = 0.5 - 0.5 * np.cos(2 * np.pi * hz * t + phase + jitter)
    return lo + (hi - lo) * c


def _minjerk(t):
    """Minimum-jerk position profile, the standard model for a voluntary reach."""
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _rep_envelope(n, rng, n_reps, hold=(0.08, 0.24), pause=(0.10, 0.34),
                  success_rate=0.8, fatigue=0.22, vigour=1.0):
    """
    A block of discrete repetitions rather than a sine wave.

    Every repetition gets its own duration, rise/fall asymmetry, hold at the
    extreme, and inter-repetition pause. Amplitude varies rep to rep: some reach
    the target, some fall short, and all of them decay slightly as the block
    goes on. `vigour` above 1 shortens the movement itself and lengthens the
    pause after it, so a stronger participant moves faster between longer rests.

    Returns (envelope in 0..1, list of (start, end), list of hit flags,
             list of (hold_start, hold_end)).
    """
    env = np.zeros(n)
    spans, hits, holds = [], [], []
    if n_reps < 1 or n < 12:
        return env, spans, hits, holds

    edges = np.linspace(0, n, n_reps + 1)
    if n_reps > 1:                                   # uneven repetition lengths
        edges[1:-1] += rng.normal(0, n / (n_reps * 7.0), n_reps - 1)
    edges = np.clip(np.sort(edges), 0, n).astype(int)

    for r in range(n_reps):
        a, b = int(edges[r]), int(edges[r + 1])
        L = b - a
        if L < 8:
            continue
        hit = bool(rng.random() < success_rate)
        amp = 1.0 if hit else float(rng.uniform(0.40, 0.85))
        amp *= 1.0 - fatigue * (r / max(1, n_reps - 1))
        amp *= float(rng.uniform(0.90, 1.10))

        hf, pf = float(rng.uniform(*hold)), float(rng.uniform(*pause))
        active = max(4, int(L * (1.0 - hf - pf) / max(0.4, vigour)))
        rise = max(2, int(active * rng.uniform(0.40, 0.60)))   # rise/fall asymmetry
        fall = max(2, active - rise)
        hold_n = max(1, int(L * hf))
        pause_n = max(0, L - rise - hold_n - fall)

        t = a
        env[t:t + rise] = amp * _minjerk(np.linspace(0, 1, rise, endpoint=False))
        t += rise
        hs = t
        env[t:t + hold_n] = amp
        t += hold_n
        env[t:t + fall] = amp * _minjerk(np.linspace(1, 0, fall, endpoint=False))
        t += fall
        # the pause stays at zero, back in the neutral resting position
        spans.append((a, b))
        hits.append(hit)
        holds.append((hs, hs + hold_n))
    return env, spans, hits, holds


# how many repetitions a block contains, and how well they go
BLOCK_STYLES = [
    ("standard", 1.00, 1.00, 0.42),
    ("long",     1.90, 0.95, 0.32),      # a persistent stretch, more reps
    ("short",    0.50, 1.05, 0.14),      # a quick couple of reps
    ("tired",    1.10, 0.55, 0.12),      # fatigued: most reps fall short
]


# =========================================================================
# session schedule
# =========================================================================
def build_schedule(rng, minutes, faulty, n_breaks=0):
    """Returns list of [stage_id, movement_or_None, side, n_frames].

    n_breaks is how many times the child leaves the alchemy workshop and goes
    back to the room mid-session. Most recordings have none.
    """
    blocks = []

    def add(stage, n_sec, mov=None, side="Undefined"):
        blocks.append([stage, mov, side, max(1, int(round(n_sec * FPS)))])

    if faulty:
        add(0, 6)            # Loading      -> unexpected phase
        add(1, 12)           # GameIntro
        add(2, 8)            # MainMenu
        add(6, 42)           # Calibration  -> unexpected phase
        add(3, 10)           # LarkRoom
        remaining = minutes * 60 - 78
        while remaining > 0:
            d = min(remaining, float(rng.uniform(18, 30)))
            add(7, d, str(rng.choice(MOVEMENTS)), str(rng.choice(["Right", "Left"])))
            remaining -= d
        return blocks

    # a well-formed session contains no Loading / Calibration / Settings /
    # ChangePlayer frames at all, so it survives the retention filters.
    # The child works through the room once and then stays in the alchemy
    # workshop: GameIntro -> MainMenu -> LarkRoom -> Alchemy -> LarkRoom,
    # with at most one mid-session return to the room.
    add(1, rng.uniform(14, 22))                  # GameIntro
    add(2, rng.uniform(8, 16))                   # MainMenu
    add(3, rng.uniform(22, 34))                  # LarkRoom, choosing a recipe

    tail = rng.uniform(16, 28)                   # LarkRoom again at the very end
    target = minutes * 60 - tail
    used = sum(b[3] for b in blocks) / FPS

    # where the optional mid-session breaks fall, as a fraction of the session
    break_at = sorted(rng.uniform(0.35, 0.72, n_breaks)) if n_breaks else []
    bi = 0

    # the game cycles through the whole movement set, so the composition of a
    # session is stable from one recording to the next
    cycle, ci = list(MOVEMENTS), 0
    rng.shuffle(cycle)
    while used < target - 20:
        if bi < len(break_at) and used > (target * break_at[bi]):
            d = float(rng.uniform(20, 32))
            add(3, d)                            # one trip back to the room
            used += d
            bi += 1
            continue
        mov = cycle[ci % len(cycle)]
        ci += 1
        if ci % len(cycle) == 0:
            rng.shuffle(cycle)
        side = "Undefined" if mov == "TE" else ("Right" if ci % 3 else "Left")
        d = float(rng.uniform(20, 28))
        add(7, d, mov, side)
        add(7, float(rng.uniform(3, 6)))         # rest at neutral, still Alchemy
        used += d + 4.5
    add(7, max(5.0, target - used))
    add(3, tail)                                 # back to the room, session ends
    return blocks


# =========================================================================
# joint-angle programme -> arrays of length N
# =========================================================================
def angle_program(p, blocks, rng):
    N = sum(b[3] for b in blocks)
    cap, hz0 = p["capability"], p["rep_hz"]

    A = dict(
        elev_R=np.full(N, 8.0), horiz_R=np.full(N, 10.0), elb_R=np.full(N, 78.0),
        open_R=np.full(N, 0.15),
        elev_L=np.full(N, 8.0), horiz_L=np.full(N, 10.0), elb_L=np.full(N, 78.0),
        open_L=np.full(N, 0.15),
        trunk_pitch=np.zeros(N), trunk_roll=np.zeros(N), retract=np.zeros(N),
        neck_rot=np.zeros(N), neck_pitch=np.full(N, 4.0),
    )
    stage = np.full(N, 7, np.int16)
    mov_id = np.full(N, -1, np.int8)
    flags = np.zeros((N, len(GESTURE_FLAGS)), np.int8)

    i = 0
    for st, mov, side, n in blocks:
        j = i + n
        stage[i:j] = st
        if mov is not None:
            mov_id[i:j] = MOVEMENTS.index(mov)
        R = side != "Left"
        sfx = "R" if R else "L"

        if st == 6:
            # calibration sweeps the full available range once per degree of freedom
            A["elev_R"][i:j] = _ramp(n, 5, 135, 0.12, rng)
            A["horiz_R"][i:j] = _ramp(n, 0, 80, 0.12, rng, phase=1.1)
            A["elb_R"][i:j] = _ramp(n, 20, 140, 0.16, rng)
            A["neck_rot"][i:j] = _ramp(n, -45, 45, 0.10, rng)
            i = j
            continue

        if st != 7 or mov is None:
            # menus, the room between potions, and the rests inside a potion:
            # near-neutral idling with a little postural drift
            A["elev_" + sfx][i:j] = 8 + _smooth(rng, n, 3.0, 0.25)
            A["neck_rot"][i:j] = _smooth(rng, n, 8.0, 0.2)
            i = j
            continue

        # ---- a movement block -------------------------------------------
        # blocks differ in how many repetitions they contain and how well they
        # go, so the same movement never looks quite the same twice
        style_i = int(rng.choice(len(BLOCK_STYLES), p=[b[3] for b in BLOCK_STYLES]))
        style, rep_mul, ok_mul, _w = BLOCK_STYLES[style_i]
        dur = n / FPS
        seq = mov in ("HR", "TE", "OC")           # sequential moves hold the extreme
        n_reps = int(max(1, round(dur * hz0 * rep_mul * rng.uniform(0.8, 1.25))))
        # a repetition cannot be arbitrarily quick: a continuous movement needs
        # about 1.6 s, a sequential one about 2.8 s including the hold and return
        n_reps = int(min(n_reps, dur / (2.8 if seq else 1.6)))
        n_reps = max(1, n_reps)
        ok = float(np.clip(p["success_rate"] * ok_mul, 0.05, 0.98))
        env, spans, hits, holds = _rep_envelope(
            n, rng, n_reps,
            hold=(0.14, 0.32) if seq else (0.06, 0.18),
            pause=(0.14, 0.38) if seq else (0.08, 0.26),
            success_rate=ok, fatigue=0.28 if style == "tired" else 0.16,
            vigour=0.45 + 1.15 * cap)

        done = np.zeros(n, bool)                  # frames where the game scores a rep
        for (a0, b0), hit in zip(holds, hits):
            if hit:
                done[a0:b0] = True

        def mark(code):
            flags[i:j, GESTURE_FLAGS.index(code)] = done.astype(np.int8)

        if mov == "EF":                           # elbow flexion, 30-100 deg
            A["elb_" + sfx][i:j] = 30 + 70 * env
            A["elev_" + sfx][i:j] = 12 + 6 * env
            mark("EF-" + ("R" if R else "L"))
        elif mov == "HA":                         # horizontal abduction
            A["elev_" + sfx][i:j] = 30 + 50 * env
            A["horiz_" + sfx][i:j] = 5 + 65 * env
            A["elb_" + sfx][i:j] = 88 + 10 * env
            mark("HA-" + ("R" if R else "L"))
        elif mov == "HR":                         # head rotation to ~45 deg and back
            A["neck_rot"][i:j] = 45 * env * (1 if R else -1)
            mark("HR-" + ("R" if R else "L"))
        elif mov == "TE":                         # thoracic extension to ~190 deg
            A["trunk_pitch"][i:j] = -22 * env
            A["retract"][i:j] = 34 * env
            A["elev_R"][i:j] = 10 + 35 * env
            A["elev_L"][i:j] = 10 + 35 * env
            mark("TE")
        elif mov == "OC":                         # fist, then opening
            A["open_" + sfx][i:j] = 0.05 + 0.9 * env
            A["elev_" + sfx][i:j] = 34 + 8 * env
            A["elb_" + sfx][i:j] = 84 + 12 * env
            mark("OC-" + ("R" if R else "L"))
        elif mov == "REACH":                      # lateral reaching, arm extends
            A["elev_" + sfx][i:j] = 20 + 150 * env
            A["horiz_" + sfx][i:j] = 20 + 60 * env
            A["elb_" + sfx][i:j] = 95 - 70 * env
        i = j

    # The movement the participant is ASKED for is the same regardless of ability.
    # Trunk compensation is driven by that intended demand, while the arm only
    # achieves a fraction of it, so a weaker participant reaches less with the
    # limb and more with the body.
    demand = (np.maximum(A["elev_R"], A["elev_L"]) / 90.0
              + np.maximum(A["horiz_R"], A["horiz_L"]) / 80.0)
    A["trunk_pitch"] += p["trunk_gain"] * 5.0 * demand + _smooth(rng, N, 0.5, 0.12)
    A["trunk_roll"] += p["trunk_gain"] * 3.2 * np.sin(demand * 2.0) + _smooth(rng, N, 0.4, 0.15)
    A["neck_pitch"] += 3.0 * demand * p["trunk_gain"] * 0.5

    # capability compresses every limb excursion toward the neutral resting pose
    comp = 0.30 + 0.70 * cap
    for k, neutral in (("elev_R", 8.0), ("elev_L", 8.0), ("horiz_R", 10.0),
                       ("horiz_L", 10.0), ("elb_R", 78.0), ("elb_L", 78.0),
                       ("open_R", 0.15), ("open_L", 0.15), ("neck_rot", 0.0)):
        A[k] = neutral + (A[k] - neutral) * comp

    # low-frequency fatigue drift across the session
    fatigue = np.linspace(0, 1, N) ** 1.5
    for k in ("elev_R", "elev_L", "horiz_R", "horiz_L"):
        A[k] = A[k] * (1.0 - 0.16 * fatigue * (1.2 - cap))
    A["trunk_pitch"] = A["trunk_pitch"] * (1.0 + 0.35 * fatigue * p["trunk_gain"] * 0.4)

    # ---- jitter, in three layers -------------------------------------
    t = np.arange(N) / FPS
    effort = np.clip(demand / 2.0, 0.0, 1.0)

    # 1. postural and tracking drift: slow, always present
    for k in ("elev_R", "elev_L", "horiz_R", "horiz_L", "elb_R", "elb_L",
              "neck_rot", "neck_pitch"):
        A[k] = A[k] + _smooth(rng, N, 1.1 * (1.5 - cap), 0.8)

    # 2. intention tremor: 3.5-6.5 Hz, growing with effort and with weakness, so
    #    part of it survives the 5 Hz filter and reaches the jerk metric
    for k, g in (("elev_R", 1.0), ("horiz_R", 1.0), ("elb_R", 1.0),
                 ("elev_L", 0.6), ("horiz_L", 0.6), ("elb_L", 0.6)):
        f = float(rng.uniform(3.5, 6.5))
        amp = g * (0.35 + 2.9 * (1.0 - cap)) * (0.25 + 0.75 * effort)
        A[k] = A[k] + amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    # 3. execution noise: broadband, scaled by how demanding the movement is
    for k in ("elev_R", "horiz_R", "elb_R", "elev_L", "horiz_L", "elb_L"):
        A[k] = A[k] + rng.normal(0, 0.55 * (1.4 - cap), N) * (0.3 + 0.7 * effort)

    return A, stage, mov_id, flags, N


# =========================================================================
# forward model: angles -> (N, 32, 3) positions in mm, +y DOWN
# =========================================================================
def build_positions(p, A, N, rng, seated=True):
    s = p["stature"]
    L = dict(pel_nav=95 * s, nav_chest=155 * s, chest_neck=125 * s, neck_head=95 * s,
             clav=45 * s, sh=105 * s, ua=205 * s, fa=185 * s, hand=72 * s,
             tip=58 * s, hip=92 * s, thigh=300 * s, shank=290 * s, foot=130 * s)

    right = np.tile([-1.0, 0.0, 0.0], (N, 1))
    fwd = np.tile([0.0, 0.0, -1.0], (N, 1))
    dn = np.tile([0.0, 1.0, 0.0], (N, 1))

    pitch = np.radians(A["trunk_pitch"])[:, None]
    roll = np.radians(A["trunk_roll"])[:, None]
    up = _u(-dn * np.cos(pitch) + fwd * np.sin(pitch) + right * np.sin(roll))
    fwd_p = _u(fwd - np.einsum("ij,ij->i", fwd, up)[:, None] * up)
    lat_r = _u(np.cross(up, fwd_p))
    lat_r = np.where((np.einsum("ij,ij->i", lat_r, right) < 0)[:, None], -lat_r, lat_r)

    P = np.zeros((N, 32, 3), np.float64)

    def put(name, arr):
        P[:, JIDX[name], :] = arr

    pelvis = np.zeros((N, 3))
    pelvis[:, 1] = 150.0 * s
    pelvis[:, 2] = 1470.0
    put("Pelvis", pelvis)
    navel = pelvis + up * L["pel_nav"]
    chest = navel + up * L["nav_chest"]
    neck = chest + up * L["chest_neck"]
    put("SpineNavel", navel)
    put("SpineChest", chest)
    put("Neck", neck)

    nf = np.radians(A["neck_pitch"])[:, None]
    nr = np.radians(A["neck_rot"])[:, None]
    head_dir = _u(up * np.cos(nf) + fwd_p * np.sin(nf))
    head = neck + head_dir * L["neck_head"]
    put("Head", head)
    face_f = _u(fwd_p * np.cos(nr) + lat_r * np.sin(nr))
    face_s = _u(np.cross(head_dir, face_f))
    put("Nose", head + face_f * 88 * s + head_dir * 8 * s)
    put("EyeLeft", head + face_f * 72 * s - face_s * 32 * s + head_dir * 26 * s)
    put("EyeRight", head + face_f * 72 * s + face_s * 32 * s + head_dir * 26 * s)
    put("EarLeft", head - face_s * 78 * s + head_dir * 18 * s)
    put("EarRight", head + face_s * 78 * s + head_dir * 18 * s)

    retract = np.radians(A["retract"])[:, None]
    for side, sgn, sfx in (("Left", -1.0, "L"), ("Right", +1.0, "R")):
        lat = lat_r * sgn
        clav = chest + lat * L["clav"] + up * (L["chest_neck"] * 0.62) \
            - fwd_p * (np.sin(retract) * 55 * s)
        sh = clav + lat * L["sh"] - fwd_p * (np.sin(retract) * 45 * s)
        put("Clavicle" + side, clav)
        put("Shoulder" + side, sh)

        # spherical shoulder: elevation from "down", plus a horizontal-plane
        # angle measured from straight-ahead toward the side
        el = np.radians(A["elev_" + sfx])[:, None]
        hzp = np.radians(A["horiz_" + sfx])[:, None]
        ua = _u(-up * np.cos(el)
                + (fwd_p * np.cos(hzp) + lat * np.sin(hzp)) * np.sin(el))
        elbow = sh + ua * L["ua"]
        put("Elbow" + side, elbow)

        flex = np.radians(A["elb_" + sfx])[:, None]
        bend = _u(fwd_p - np.einsum("ij,ij->i", fwd_p, ua)[:, None] * ua)
        fa = _u(ua * np.cos(flex) + bend * np.sin(flex))
        wrist = elbow + fa * L["fa"]
        hand = wrist + fa * L["hand"]
        put("Wrist" + side, wrist)
        put("Hand" + side, hand)

        # finger curl: open -> tip continues along the forearm,
        # closed -> tip folds back toward the palm
        curl = np.radians(115.0 * (1.0 - A["open_" + sfx]))[:, None]
        palm = _u(np.cross(fa, lat))
        tip_dir = _u(fa * np.cos(curl) - palm * np.sin(curl))
        put("HandTip" + side, hand + tip_dir * L["tip"])
        put("Thumb" + side, hand + lat * ((18 + 26 * A["open_" + sfx])[:, None] * s)
            - fa * (12 * s) + palm * (10 * s))

        hip = pelvis + lat * L["hip"]
        put("Hip" + side, hip)
        if seated:
            knee = hip + dn * (L["thigh"] * 0.30) + fwd * (L["thigh"] * 0.90)
            ankle = knee + dn * (L["shank"] * 0.95) + fwd * (L["shank"] * 0.15)
        else:
            knee = hip + dn * L["thigh"]
            ankle = knee + dn * L["shank"]
        put("Knee" + side, knee)
        put("Ankle" + side, ankle)
        put("Foot" + side, ankle + fwd * L["foot"])

    sd = np.full(32, p["noise_mm"])
    for nm in JOINTS_32:
        if nm.startswith(("Hand", "Thumb", "Wrist", "Foot")):
            sd[JIDX[nm]] += p["tremor_mm"]
    P += rng.normal(0.0, 1.0, P.shape) * sd[None, :, None]
    return P


# =========================================================================
# the intruder: a taller person walking across, further from the camera
# =========================================================================
def build_intruder(N, rng):
    q = dict(noise_mm=9.0, tremor_mm=4.0, stature=1.45)
    A = dict(
        elev_R=18 + _smooth(rng, N, 12, 0.5), horiz_R=np.full(N, 12.0),
        elb_R=40 + _smooth(rng, N, 20, 0.6), open_R=np.full(N, 0.5),
        elev_L=18 + _smooth(rng, N, 12, 0.5), horiz_L=np.full(N, 12.0),
        elb_L=40 + _smooth(rng, N, 20, 0.6), open_L=np.full(N, 0.5),
        trunk_pitch=_smooth(rng, N, 4, 0.3), trunk_roll=_smooth(rng, N, 5, 0.5),
        retract=np.zeros(N), neck_rot=_smooth(rng, N, 20, 0.4),
        neck_pitch=np.full(N, 3.0),
    )
    P = build_positions(q, A, N, rng, seated=False)
    P[:, :, 2] += 780.0                                    # further from the camera
    P[:, :, 0] += np.linspace(-950, 900, N)[:, None]       # walking across the frame
    P[:, :, 1] -= 260.0                                    # standing, so taller
    return P


# =========================================================================
# label channels
# =========================================================================
def compute_angle_channels(P, A):
    """The 15 continuous comparison values the game logs on every frame."""
    def g(nm):
        return P[:, JIDX[nm], :]

    N = P.shape[0]
    spine_up = _u(g("SpineChest") - g("SpineNavel"))
    spine_fwd = _u(np.cross(spine_up, _u(g("ShoulderRight") - g("ShoulderLeft"))))
    out = np.zeros((N, 15), np.float32)
    for k, side in ((0, "Right"), (2, "Left")):
        hand, tip, wrist = g("Hand" + side), g("HandTip" + side), g("Wrist" + side)
        out[:, k] = _ang(tip - hand, wrist - g("Elbow" + side))
        out[:, k + 1] = np.linalg.norm(tip - wrist, axis=1)
    for k, side in ((4, "Right"), (7, "Left")):
        sh, el, wr = g("Shoulder" + side), g("Elbow" + side), g("Wrist" + side)
        arm_fwd = _u((el - sh) + (wr - sh) + (wr - el))
        out[:, k] = _ang(arm_fwd, spine_fwd)
        out[:, k + 1] = _ang(el - sh, -spine_up)
        out[:, k + 2] = _ang(wr - el, spine_up)
    for k, side in ((10, "Right"), (11, "Left")):
        out[:, k] = _ang(g("Shoulder" + side) - g("Elbow" + side),
                         g("Wrist" + side) - g("Elbow" + side))
    head_fwd = _u(g("Nose") - g("Head"))
    hr = _ang(head_fwd, spine_fwd)
    out[:, 12] = hr * (A["neck_rot"] >= 0)
    out[:, 13] = hr * (A["neck_rot"] < 0)
    te = 0.5 * (_ang(g("ShoulderLeft") - g("ClavicleLeft"), g("ElbowLeft") - g("Neck"))
                + _ang(g("ShoulderRight") - g("ClavicleRight"), g("ElbowRight") - g("Neck")))
    out[:, 14] = 180.0 + te * 0.35
    return out


# =========================================================================
# assemble one session
# =========================================================================
def make_session(name, prof_key, minutes, seed, faulty, n_breaks=0):
    p = PROFILES[prof_key]
    rng = np.random.default_rng(seed)
    blocks = build_schedule(rng, minutes, faulty, n_breaks)
    A, stage, mov_id, flags, N = angle_program(p, blocks, rng)
    P = build_positions(p, A, N, rng, seated=True)

    nbody = np.ones(N, np.int8)
    bodyid = np.ones(N, np.int8)
    intruder = np.zeros(N, bool)

    if faulty:
        # a second person enters the frame and tracking follows them instead
        a = int(N * 0.46)
        b = min(N, a + int(26 * FPS))
        P[a:b] = build_intruder(b - a, rng)
        nbody[a:b] = 2
        bodyid[a:b] = 2
        intruder[a:b] = True
        for e in (a - 4, a - 2, b + 3, b + 9):   # brief identity flickers at the edges
            if 0 <= e < N:
                nbody[e] = 2

    ang_ch = compute_angle_channels(P, A)

    conf = np.ones((N, 32), np.uint8)
    for nm in JOINTS_32:
        j = JIDX[nm]
        base = 0.55 if nm in LOW_CONF_JOINTS else p["drop_rate"]
        conf[:, j] = np.where(rng.random(N) < base, 0, 1)
    if faulty and intruder.any():
        sub = conf[intruder]
        conf[intruder] = np.where(rng.random(sub.shape) < 0.45, 0, 1)

    meta = {
        "name": name, "profile": prof_key, "fps": FPS,
        "profileLabel": p["label"], "syntheticData": True, "faulty": bool(faulty),
        "demoClinical": {"HFMSE": p["hfmse"], "RULM": p["rulm"],
                         "SMA_Type": p["sma_type"], "Functional": p["functional"],
                         "note": "synthetic illustrative values, not real clinical scores"},
        "movements": MOVEMENTS, "gestureFlags": GESTURE_FLAGS,
        "angleChannels": ANGLE_CHANNELS, "joints": JOINTS_32,
        "stageNames": {str(k): v for k, v in STAGE_NAMES.items()},
    }
    return dict(P=P.astype(np.float32), stage=stage, nbody=nbody, bodyid=bodyid,
                conf=conf, flags=flags, ang=ang_ch, mov=mov_id,
                intruder=intruder, meta=meta, seed=seed)


# =========================================================================
# writers
# =========================================================================
def write_npz(rec, path):
    np.savez_compressed(
        path, P=rec["P"], stage=rec["stage"], nbody=rec["nbody"], bodyid=rec["bodyid"],
        conf=rec["conf"], flags=rec["flags"], ang=rec["ang"], mov=rec["mov"],
        intruder=rec["intruder"], meta=np.array(json.dumps(rec["meta"])))


def write_json_excerpt(rec, path, n_frames=40):
    """A short, pretty-printed, uncompressed excerpt for inspection.

    Built straight into memory: no temporary file, so nothing here can fail on a
    read-only or unusual filesystem.
    """
    import copy
    import io
    small = copy.deepcopy(rec)
    for k in ("P", "stage", "nbody", "bodyid", "conf", "flags", "ang", "mov",
              "intruder"):
        small[k] = rec[k][:n_frames]
    buf = io.StringIO()
    _write_json_stream(small, buf)
    obj = json.loads(buf.getvalue())
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_json_gz(rec, path):
    """Streams the full schema out frame by frame so memory stays flat."""
    with gzip.open(path, "wt", compresslevel=6) as f:
        _write_json_stream(rec, f)


def _write_json_stream(rec, f):
    """Writes the whole recording to an open text stream, frame by frame."""
    P, stage, nbody, bodyid = rec["P"], rec["stage"], rec["nbody"], rec["bodyid"]
    conf, flags, ang = rec["conf"], rec["flags"], rec["ang"]
    N = P.shape[0]
    rng = np.random.default_rng(rec["seed"] + 7)
    t0 = 133983073469909402 + rec["seed"] * 1_000_000

    head = {
        "patientId": f"DEMO_{rec['meta']['name']}",
        "sessionId": rec["meta"]["name"],
        "syntheticData": True,
        "profileLabel": rec["meta"]["profileLabel"],
        "demoClinical": rec["meta"]["demoClinical"],
        "loginTime": str(t0),
        "logoutTime": str(t0 + int(N / FPS * 1e7)),
        "potionsPrepared": int(rng.integers(3, 12)),
        "labelsVer": 1,
        "labelsNames": LABELS_NAMES,
        "stagesNames": STAGES_NAMES_STR,
        "gameplayData": [],
    }
    calib = {"values": [
        {"gestureId": gid, "gestureHandedness": hand,
         "gestureType": "ContinousGesture" if gid in ("HA", "OC") else "TwoStateGesture",
         "gestureExtremum": ext, "comparisonId": cid,
         "extremumValue": round(float(rng.uniform(-120, 200)), 6),
         "averageValue": round(float(rng.uniform(-60, 175)), 6)}
        for gid, hand, cid, ext in [
            ("OC", "Right", "OC_HandRightToHandTipRightRotation", "Minimum"),
            ("OC", "Left", "OC_HandLeftToHandTipLeftRotation", "Minimum"),
            ("HA", "Right", "HA_RightArmForwardToSpineForwardRotation", "Maximum"),
            ("HA", "Left", "HA_LeftArmForwardToSpineForwardRotation", "Maximum"),
            ("EF", "Right", "EF_ArmRightToForeamrRightRotation", "Maximum"),
            ("EF", "Left", "EF_ArmLeftToForeamrLeftRotation", "Maximum"),
            ("HR", "Right", "HR_HeadForwardToSpineForwardRotation", "Maximum"),
            ("HR", "Left", "HR_HeadForwardToSpineForwardRotation", "Minimum"),
            ("TE", "Undefined", "TE_LeftShoulderToRightShoulderRotation", "Maximum"),
        ]]}

    jn = JOINTS_32
    if True:
        f.write(json.dumps(head)[:-1])
        f.write("," + json.dumps({"calibration": calib})[1:-1])
        f.write(',"frameData":[')
        for i in range(N):
            if i:
                f.write(",")
            lab = (f"{stage[i]};{nbody[i]};{bodyid[i]};Central European Daylight Time;"
                   f"{2418614.3407 + i * (1000.0 / FPS):.4f};{t0 + i * 333333};"
                   f"{i / FPS:.4f};"
                   + ";".join(f"{v:.4f}" for v in ang[i]) + ";"
                   + ";".join(str(int(v)) for v in flags[i]) + ";")
            d = P[i] - P[i, 0]
            nrm = np.linalg.norm(d, axis=1)
            nrm[nrm == 0] = 1.0
            dn = d / nrm[:, None]
            w = np.sqrt(np.clip(1.0 - np.minimum(1.0, (nrm / 900.0) ** 2), 0, 1))
            js = [
                '{"jointType":"%s","position":{"x":%.2f,"y":%.2f,"z":%.2f},'
                '"orientation":{"x":%.4f,"y":%.4f,"z":%.4f,"w":%.4f},'
                '"confidenceLevel":"%s"}'
                % (jn[k], P[i, k, 0], P[i, k, 1], P[i, k, 2],
                   dn[k, 0], dn[k, 1], dn[k, 2], w[k], CONF_NAME[int(conf[i, k])])
                for k in range(32)]
            f.write('{"frameDataId":%d,"joints":[%s],"labels":"%s"}'
                    % (i + 1, ",".join(js), lab))
        f.write("]}")


# =========================================================================
def main(out_dir="data", npz_only=False, minutes=None, quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    for name, prof, mins, seed, faulty, nb in SESSION_PLAN:
        if minutes is not None:
            mins = minutes if not faulty else max(0.6, minutes * 0.13)
        rec = make_session(name, prof, mins, seed, faulty, nb)
        npz = os.path.join(out_dir, name + ".npz")
        write_npz(rec, npz)
        msg = (f"{name}: {rec['P'].shape[0]:>6} frames "
               f"({rec['P'].shape[0] / FPS / 60:5.1f} min)  "
               f"npz {os.path.getsize(npz) / 1e6:5.1f} MB")
        if not npz_only:
            ex = os.path.join(out_dir, name + "_excerpt.json")
            write_json_excerpt(rec, ex)
            msg += f"  excerpt {os.path.getsize(ex) / 1e6:4.1f} MB"
            jz = os.path.join(out_dir, name + ".json.gz")
            write_json_gz(rec, jz)
            msg += f"  json.gz {os.path.getsize(jz) / 1e6:5.1f} MB"
        if not quiet:
            print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--npz-only", action="store_true")
    ap.add_argument("--minutes", type=float, default=None)
    a = ap.parse_args()
    main(a.out, a.npz_only, a.minutes)
