"""
app.py — Markerless Motion Capture in Paediatric SMA
===================================================
Interactive viewer for Azure Kinect DK skeleton recordings captured during
home-based exergaming, showing the physiotherapy movement protocol, the game
phase structure, session quality control, and the 12 canonical upper-limb
kinematic features.

ALL DATA IS SYNTHETIC. See make_demo_data.py.

Run:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

import kinematics as K
import plotting as PL

# =========================================================================
#  CONFIGURATION - edit this block only
# =========================================================================
#
#  Leave LOCAL_DATA_DIR empty to run on the bundled synthetic demo sessions.
#
#  To work with your own recordings, point it at the folder holding them.
#  The folder is searched recursively for .zip, .json and .json.gz files, so
#  the original zipped session archives work as they are.
#
#      Windows :  LOCAL_DATA_DIR = r"C:\Users\you\Documents\video tela patients"
#      macOS   :  LOCAL_DATA_DIR = "/Users/you/Documents/video tela patients"
#      Linux   :  LOCAL_DATA_DIR = "/home/you/documents/video tela patients"
#      relative:  LOCAL_DATA_DIR = "video tela patients"
#
#  Nothing ever leaves the machine: files are read locally and converted once
#  into compact .npz arrays inside LOCAL_CACHE_DIR, which keeps later loads
#  fast. Delete that folder to force a re-import.
#
LOCAL_DATA_DIR = ""                         # <-- your recordings folder
LOCAL_CACHE_DIR = ".kinect_cache"           # where the converted arrays live

#  Optional: map the patientId inside each file to a readable label. Leave it
#  empty and the file name is used instead. Keep real identifiers out of any
#  repository you push.
PATIENT_ID_MAP = {
    # "b1a7...": "P01",
    # "c9dd...": "P02",
}

DATA_DIR = "data"                           # bundled synthetic sessions
REFRESH = 0.20                              # seconds between animation steps
# =========================================================================

USE_LOCAL = bool(LOCAL_DATA_DIR.strip())

DEMO_SESSIONS = ["P01_S1", "P01_S2", "P02_S1", "P02_S2", "BAD_S1"]
SESSION_CAPTIONS = {
    "P01_S1": "weaker profile - session 1", "P01_S2": "weaker profile - session 2",
    "P02_S1": "stronger profile - session 1", "P02_S2": "stronger profile - session 2",
    "BAD_S1": "faulty recording - excluded",
}

st.set_page_config(page_title="Kinect SMA - motion analysis demo",
                   page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
  h1, h2, h3 {color:#1f2733; letter-spacing:-0.01em;}
  .hero {background:linear-gradient(100deg,#16305a 0%,#2b5a8f 55%,#3f7fa8 100%);
         color:#fff; padding:1.5rem 1.9rem; border-radius:14px; margin-bottom:1.1rem;}
  .hero h1 {color:#fff; font-size:1.75rem; margin:0 0 .35rem 0;}
  .hero p  {color:#d7e5f5; font-size:.95rem; margin:0; max-width:62rem; line-height:1.5;}
  .pill {display:inline-block; padding:.16rem .62rem; border-radius:999px;
         font-size:.72rem; font-weight:600; letter-spacing:.02em;}
  .ok   {background:#e3f4e8; color:#1c6b34;}
  .bad  {background:#fdE6e6; color:#a32020;}
  [data-testid="stMetricValue"] {font-size:1.35rem;}
  .cap {color:#8a93a0; font-size:.8rem; line-height:1.45;}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# data
# =========================================================================
def _demo_sessions():
    missing = [s for s in DEMO_SESSIONS
               if not os.path.exists(os.path.join(DATA_DIR, s + ".npz"))]
    if missing:
        import make_demo_data
        with st.spinner("Generating synthetic sessions - first run only, ~15 s"):
            make_demo_data.main(DATA_DIR, npz_only=True, quiet=True)
    return {s: os.path.join(DATA_DIR, s + ".npz") for s in DEMO_SESSIONS}


def _local_sessions():
    """Lists the local recordings, importing any that are not cached yet."""
    folder = LOCAL_DATA_DIR.strip()
    if not os.path.isdir(folder):
        st.error(f"LOCAL_DATA_DIR does not exist:  {folder}\n\n"
                 "Fix the path in the configuration block at the top of app.py, "
                 "or set it to an empty string to use the synthetic demo data.",
                 icon="🚫")
        st.stop()

    cache = os.path.join(folder, LOCAL_CACHE_DIR)
    sources = K.list_local_sessions(folder)
    sources = [p for p in sources if LOCAL_CACHE_DIR not in p]
    if not sources:
        st.error(f"No .zip, .json or .json.gz recordings found under {folder}.",
                 icon="🚫")
        st.stop()

    cached = {K.session_key(p): os.path.join(cache, K.session_key(p) + ".npz")
              for p in sources}
    todo = [p for p in sources if not os.path.exists(cached[K.session_key(p)])]

    if todo:
        st.info(f"**{len(sources)} recordings found in `{folder}`.** "
                f"{len(todo)} still need a one-off conversion into compact arrays "
                "(roughly 20-60 s per 20-minute session). The originals are not "
                "modified and nothing is uploaded.", icon="📂")
        if not st.button(f"Import {len(todo)} recording(s)", type="primary"):
            st.stop()
        bar = st.progress(0.0, "starting…")
        failed = []
        for i, p in enumerate(todo, 1):
            bar.progress((i - 1) / len(todo), f"reading {os.path.basename(p)}")
            try:
                K.cache_recording(p, cache, patient_map=PATIENT_ID_MAP)
            except Exception as exc:                     # keep going on bad files
                failed.append(f"{os.path.basename(p)}: {exc}")
        bar.progress(1.0, "done")
        if failed:
            st.warning("Some files could not be read:\n\n- " + "\n- ".join(failed),
                       icon="⚠️")
        st.rerun()

    return {k: v for k, v in cached.items() if os.path.exists(v)}


@st.cache_data(max_entries=2, show_spinner="Loading session and computing features…")
def load_viewer(path):
    rec = K.load_npz(path)
    qc = K.qc_metrics(rec)
    prep = K.prepare(rec, alchemy_only=True)
    tr = K.compute_traces(prep)

    n_full = len(rec["stage"])
    keep = prep["keep"]
    traces = {k: np.full(n_full, np.nan, np.float32) for k in K.TARGET_12_FEATURES}
    for k in K.TARGET_12_FEATURES:
        traces[k][keep] = tr[k]

    P = rec["P"].astype(np.float32)
    raw = np.empty_like(P)
    raw[:, :, 0] = P[:, :, 0] / 1000.0
    raw[:, :, 1] = -P[:, :, 1] / 1000.0        # +y becomes UP
    raw[:, :, 2] = P[:, :, 2] / 1000.0

    sub = raw[::7]                              # axis limits from a subsample
    lims = {}
    for i, a in enumerate("xyz"):
        lo, hi = float(np.nanmin(sub[:, :, i])), float(np.nanmax(sub[:, :, i]))
        r = hi - lo
        lims[a] = (lo - 0.10 * r, hi + 0.10 * r)
    ctr = [(lims[a][0] + lims[a][1]) / 2 for a in "xyz"]
    half = max(lims[a][1] - lims[a][0] for a in "xyz") / 2
    cube = {a: (ctr[i] - half, ctr[i] + half) for i, a in enumerate("xyz")}

    mov = rec["mov"].astype(np.int16)
    if (mov >= 0).any():
        blocks = [(a, b, rec["meta"]["movements"][m])
                  for a, b, m in K.stage_segments(mov) if m >= 0]
    else:
        # real recordings carry no movement-block track, so the completed-movement
        # flags are used to locate the blocks instead
        blocks = K.blocks_from_flags(rec["flags"], rec["meta"]["gestureFlags"])

    return dict(raw=raw, conf=rec["conf"], stage=rec["stage"], mov=rec["mov"],
                flags=rec["flags"], nbody=rec["nbody"], traces=traces, cube=cube,
                meta=rec["meta"], qc=qc, keep=keep, blocks=blocks, n=n_full)


@st.cache_data(show_spinner="Computing session summaries…")
def get_summary(path):
    rec = K.load_npz(path)
    prep = K.prepare(rec, alchemy_only=True)
    tr = K.compute_traces(prep)
    return dict(scalars=K.session_summary(prep, tr),
                dist=K.distribution_table(tr),
                qc={k: v for k, v in K.qc_metrics(rec).items() if k != "checks"})


# =========================================================================
# header
# =========================================================================
st.markdown("""
<div class="hero">
  <h1>Markerless motion capture in paediatric SMA</h1>
  <p>Azure Kinect DK skeleton recordings from home-based exergaming, with the
  physiotherapy movement protocol, session quality control, and the twelve
  canonical upper-limb kinematic features. Every recording below is
  <strong>synthetic</strong> - procedurally generated to imitate two contrasting
  motor profiles plus one deliberately faulty session. No real participant data,
  identifiers or clinical scores appear anywhere in this application.</p>
</div>
""", unsafe_allow_html=True)

PATHS = _local_sessions() if USE_LOCAL else _demo_sessions()
SESSIONS = list(PATHS)

# =========================================================================
# sidebar
# =========================================================================
with st.sidebar:
    st.subheader("Recording")
    if USE_LOCAL:
        st.caption(f"{len(SESSIONS)} local recording(s)")
        name = st.selectbox("Session", SESSIONS, index=0)
    else:
        name = st.radio("Session", SESSIONS, index=0,
                        captions=[SESSION_CAPTIONS.get(s, "") for s in SESSIONS])

    st.divider()
    st.subheader("Features")
    if "feat_sel" not in st.session_state:
        st.session_state["feat_sel"] = ["shoulder_flexext_rom_R", "elbow_rom_R",
                                        "trunk_comp_R"]
    ca, cb = st.columns(2)
    if ca.button("All 12", use_container_width=True):
        st.session_state["feat_sel"] = list(K.TARGET_12_FEATURES)
    if cb.button("Clear", use_container_width=True):
        st.session_state["feat_sel"] = []
    selected = st.multiselect("Overlay on the graph", K.TARGET_12_FEATURES,
                              key="feat_sel",
                              format_func=lambda k: K.FEATURE_DISPLAY_NAMES[k])
    normalise = st.toggle("Normalise each feature to 0-1", value=True,
                          help="Turn off to read raw units. Only useful with a "
                               "single feature, or features sharing a scale.")

    st.divider()
    st.subheader("Playback")
    speed = st.select_slider("Speed", options=[0.5, 1, 2, 4, 8], value=2,
                             format_func=lambda v: f"{v}x real time")
    loop = st.checkbox("Repeat", value=True)
    zoom = st.checkbox("Zoom graph to a window around the cursor", value=False)
    zoom_s = st.slider("Window (seconds)", 5, 120, 30, disabled=not zoom)

V = load_viewer(PATHS[name])
N = V["n"]
meta, qc = V["meta"], V["qc"]

if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
st.session_state.frame = int(np.clip(st.session_state.frame, 0, N - 1))

# =========================================================================
# headline metrics
# =========================================================================
badge = ('<span class="pill ok">RETAINED</span>' if qc["retained"]
         else '<span class="pill bad">EXCLUDED</span>')
m = st.columns([1.1, 1, 1, 1, 1.2])
m[0].metric("Session", name, help=meta["profileLabel"])
m[1].metric("Duration", f"{qc['minutes']:.1f} min", f"{qc['n_frames']:,} frames")
m[2].metric("Gameplay share", f"{qc['alchemy_pct']:.1f} %")
m[3].metric("Unexpected phases", f"{qc['unexpected_pct']:.1f} %")
with m[4]:
    st.markdown("**Retention**")
    st.markdown(badge, unsafe_allow_html=True)
    if qc["multi_body_pct"] > 0:
        st.markdown(f'<span class="cap">{qc["multi_body_pct"]:.1f} % of frames '
                    "contain a second body</span>", unsafe_allow_html=True)

tab_view, tab_qc, tab_dist, tab_cmp = st.tabs(
    ["Session viewer", "Quality control", "Feature distributions",
     "Cross-session comparison"])

# =========================================================================
# TAB 1 - viewer
# =========================================================================
with tab_view:
    blocks = V["blocks"]
    c = st.columns([1.4, 1.4, 1, 1, 1, 3])
    mov_choice = c[0].selectbox("Jump to movement", ["(any)"] + meta["movements"],
                                label_visibility="collapsed")

    def _jump(direction):
        f = st.session_state.frame
        cand = [b for b in blocks if mov_choice in ("(any)", b[2])]
        nxt = [b[0] for b in cand if b[0] > f + 2] if direction > 0 else \
              [b[0] for b in cand if b[0] < f - 2]
        if nxt:
            st.session_state.frame = nxt[0] if direction > 0 else nxt[-1]

    if c[1].button("Previous block", use_container_width=True):
        _jump(-1)
    if c[2].button("Next block", use_container_width=True):
        _jump(+1)

    def _rerun_app():
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()

    _refresh = REFRESH if st.session_state.playing else None
    _frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

    def viewer():
        b = st.columns([1, 1, 1, 6])
        if b[0].button("Pause" if st.session_state.playing else "Play",
                       use_container_width=True, key="b_play", type="primary"):
            st.session_state.playing = not st.session_state.playing
            _rerun_app()
        if b[1].button("Restart", use_container_width=True, key="b_restart"):
            st.session_state.frame = 0
            st.rerun()
        if b[2].button("End", use_container_width=True, key="b_end"):
            st.session_state.frame = N - 1
            st.rerun()

        if st.session_state.playing:
            nxt = st.session_state.frame + max(1, int(round(speed * K.FPS * REFRESH)))
            if nxt >= N:
                if loop:
                    nxt = 0
                else:
                    nxt, st.session_state.playing = N - 1, False
            st.session_state.frame = int(nxt)

        frame = st.slider("Frame", 0, N - 1, key="frame")
        t = frame / K.FPS
        stg = K.STAGE_NAMES.get(int(V["stage"][frame]), "?")
        mv = int(V["mov"][frame])
        mv = meta["movements"][mv] if mv >= 0 else "-"
        warn = " · SECOND BODY IN FRAME" if V["nbody"][frame] != 1 else ""
        st.markdown(
            f'<span class="cap">t = {int(t // 60):02d}:{t % 60:05.2f} &nbsp;|&nbsp; '
            f'phase <b>{stg}</b> &nbsp;|&nbsp; movement <b>{mv}</b>'
            f'{"&nbsp;|&nbsp; <b style=color:#a32020>" + warn + "</b>" if warn else ""}'
            "</span>", unsafe_allow_html=True)

        st.pyplot(PL.draw_skeleton(V["raw"][frame], V["conf"][frame], V["cube"],
                                   meta["joints"],
                                   f"{name}  |  frame {frame + 1:,} / {N:,}"),
                  clear_figure=True)
        st.markdown(
            '<span class="cap">Bones are drawn in the background and joint markers '
            "in the foreground. Marker opacity encodes the Kinect tracking-confidence "
            "level (Medium 0.6, Low 0.25). Only Medium-confidence joints enter the "
            "feature computation.</span>", unsafe_allow_html=True)

        st.markdown("##### Session timeline")
        st.pyplot(PL.draw_timeline(V["stage"], V["mov"], meta["movements"],
                                   V["flags"], meta["gestureFlags"], frame,
                                   V["nbody"]), clear_figure=True)

        st.markdown("##### Kinematic features over time")
        win = None
        if zoom:
            h = int(zoom_s * K.FPS / 2)
            win = (max(0, frame - h), min(N, frame + h))
        st.pyplot(PL.draw_features(V["traces"], selected, frame, normalise,
                                   window=win), clear_figure=True)

    if _frag is not None:
        viewer = _frag(run_every=_refresh)(viewer)
    viewer()

    if selected:
        st.markdown('<span class="cap">Traces shown: ' + " &nbsp;·&nbsp; ".join(
            f"<b>{K.FEATURE_DISPLAY_NAMES[k]}</b> - {K.TRACE_MEANING[k]}"
            for k in selected) + "</span>", unsafe_allow_html=True)
    st.markdown(
        '<span class="cap">Gaps in the traces are frames excluded by the '
        "frame-level filter: anything outside the gameplay phase, or where the "
        "sensor reported more than one body.</span>", unsafe_allow_html=True)

# =========================================================================
# TAB 2 - quality control
# =========================================================================
with tab_qc:
    st.markdown("#### Session retention filters")
    st.markdown(
        '<span class="cap">A recording is carried into the analysis only if it '
        "passes all three checks. They are applied to the whole file, before any "
        "frame-level or joint-level filtering.</span>", unsafe_allow_html=True)
    rows = [{"Check": c[0], "Observed": c[1], "Requirement": c[2],
             "Result": "pass" if c[3] else "FAIL"} for c in qc["checks"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if qc["retained"]:
        st.success(f"{name} is retained for analysis.", icon="✅")
    else:
        failed = [c[0] for c in qc["checks"] if not c[3]]
        st.error(f"{name} is excluded. Failed: {', '.join(failed)}.", icon="🚫")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Game phase composition")
        st.pyplot(PL.draw_stage_breakdown(qc["counts"], qc["n_frames"]),
                  clear_figure=True)
    with right:
        st.markdown("##### Frame-level filtering")
        kept = len(V["keep"])
        st.metric("Frames entering feature computation", f"{kept:,}",
                  f"{kept - qc['n_frames']:,} vs. recorded")
        st.markdown(
            '<span class="cap">Frames are kept only where the sensor reported '
            "exactly one body and the game was in the gameplay phase. Within the "
            "surviving frames, individual joints below Medium tracking confidence "
            "are set to missing, short gaps up to five frames are interpolated, the "
            "signal is low-pass filtered with a 4th-order zero-phase Butterworth at "
            "5 Hz, and samples implying a velocity above 3 m/s are rejected as "
            "tracking glitches.</span>", unsafe_allow_html=True)

    if qc["multi_body_pct"] > 0:
        st.warning(
            f"A second person is present in {qc['multi_body_pct']:.1f} % of frames. "
            "During that stretch body tracking follows the wrong person: the "
            "skeleton is taller, further from the camera and translates across the "
            "field of view. Those frames are visible in the viewer but are dropped "
            "before any feature is computed.", icon="⚠️")

    st.markdown("##### Movement protocol")
    prot = pd.DataFrame(
        [{"Code": k, "Movement": v,
          "Type": "sequential" if k in ("HR", "TE", "OC") else "continuous"}
         for k, v in K.MOVEMENT_NAMES.items()])
    st.dataframe(prot, hide_index=True, use_container_width=True)

# =========================================================================
# TAB 3 - distributions
# =========================================================================
with tab_dist:
    st.markdown(f"#### Per-session feature distributions — {name}")
    st.markdown(
        '<span class="cap">Each of the twelve features is summarised by its '
        "median, mean, standard deviation, selected percentiles, skewness and "
        "kurtosis, together with a range-based descriptor (maximum minus minimum) "
        "for the angular and excursion features.</span>", unsafe_allow_html=True)
    S = get_summary(PATHS[name])
    df = pd.DataFrame(S["dist"])
    st.dataframe(df.style.format(precision=3), hide_index=True,
                 use_container_width=True)
    st.download_button("Download this table as CSV",
                       df.to_csv(index=False).encode(),
                       file_name=f"{name}_feature_distributions.csv",
                       mime="text/csv")

# =========================================================================
# TAB 4 - cross-session comparison
# =========================================================================
with tab_cmp:
    st.markdown("#### Session-level feature values")
    st.markdown(
        '<span class="cap">Angular and excursion features are reported as maximum '
        "minus minimum; workspace volume as the convex-hull volume of the whole "
        "hand path; trunk compensation as total trunk path divided by total hand "
        "path; hand speed and log dimensionless jerk over the full trajectory.</span>",
        unsafe_allow_html=True)

    order = list(SESSIONS)
    sums = {s: get_summary(PATHS[s])["scalars"] for s in order}
    tab = pd.DataFrame(
        {s: {K.FEATURE_DISPLAY_NAMES[k]: sums[s][k] for k in K.TARGET_12_FEATURES}
         for s in order})
    tab.insert(0, "Domain", [K.FEATURE_DOMAIN[k] for k in K.TARGET_12_FEATURES])
    st.dataframe(tab.style.format(precision=3, subset=order),
                 use_container_width=True)

    st.markdown("#### Test-retest stability within a participant")
    st.markdown(
        '<span class="cap">Coefficient of variation across the two sessions of the '
        "same participant, classified as highly stable below 15 %, moderately "
        "stable from 15 to 30 %, and unstable above 30 %. With two sessions per "
        "participant this is illustrative only; an intraclass correlation "
        "coefficient needs the full cohort.</span>", unsafe_allow_html=True)
    pairs = {}
    for s in order:
        stem = s.rsplit("_", 1)[0] if "_" in s else s
        pairs.setdefault(stem, []).append(s)
    pairs = {k: v for k, v in pairs.items() if len(v) == 2}
    if not pairs:
        st.caption("Needs two sessions sharing a name stem, for example "
                   "`P01_S1` and `P01_S2`.")
    else:
        cv_rows = []
        for k in K.TARGET_12_FEATURES:
            r = {"Feature": K.FEATURE_DISPLAY_NAMES[k]}
            for stem, (a, b) in pairs.items():
                v = K.cv_percent(sums[a][k], sums[b][k])
                r[f"{stem} CV %"] = v
                r[f"{stem} stability"] = K.band(v, K.CV_BANDS)
            cv_rows.append(r)
        st.dataframe(pd.DataFrame(cv_rows).style.format(precision=2, na_rep="-"),
                     hide_index=True, use_container_width=True)

    st.markdown("#### Feature-by-feature")
    feat = st.selectbox("Feature", K.TARGET_12_FEATURES,
                        format_func=lambda k: K.FEATURE_DISPLAY_NAMES[k])
    st.pyplot(PL.draw_comparison(sums, feat, order), clear_figure=True)

with st.expander("Notes, definitions and caveats"):
    st.markdown("""
**Coordinate handling.** Raw Kinect positions are in millimetres with *+y pointing
down*. On load they are divided by 1000 and *y* is negated, so in every computation
and plot **+y is up**. The three panels show front *(x, -y)*, side *(z, -y)* and
top *(x, z)* on a shared cubic axis range, so proportions are preserved.

**Movement protocol.** From the neutral resting position (sitting upright, arms
relaxed on the thighs) the game requests elbow flexion (continuous, 30-100 deg),
horizontal abduction (continuous: arm-forward to spine-forward 0-70 deg, upper arm
to spinal down 30-80 deg, forearm near horizontal at 77-113 deg), head rotation
(sequential, about 45 deg then back to neutral), thoracic extension (sequential,
reaching about 190 deg then returning) and hand opening and closing (sequential,
fist 30-90 deg then opening 100-180 deg), plus lateral reaching. The synthetic
participants attempt the same protocol; the weaker profile achieves a smaller
fraction of it and compensates with the trunk.

**Traces vs. scalars.** The animated curves are instantaneous, rolling or cumulative
versions of the features so they can be watched frame by frame. Log dimensionless
jerk and the trunk compensation ratio use a 4-second sliding window; workspace
volume and % time above shoulder accumulate from the start of the session. The
comparison tab reports the session-level scalars instead.

**Interpretation.** More negative log dimensionless jerk means a less smooth
movement. A higher trunk compensation ratio means more trunk displacement per unit
of hand displacement, that is, the reach is being achieved by moving the body
rather than the arm.

**Files.** Each session also exists as `<name>.json.gz`, the complete recording in
the original schema, and `<name>_excerpt.json`, a short pretty-printed excerpt. An
uncompressed 20-minute session is roughly 200 MB, which is why the full files are
gzipped and the application reads compact `.npz` arrays instead.
    """)
