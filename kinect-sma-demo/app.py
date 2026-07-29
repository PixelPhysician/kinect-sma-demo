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
import streamlit.components.v1 as components

import kinematics as K
import player as PLY
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

#  Tabs that are built but not displayed. Flip either to True to bring it back.
SHOW_QUALITY_CONTROL = False
SHOW_CROSS_SESSION = False
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


@st.cache_data(max_entries=4, show_spinner=False)
def session_charts(path, feat_key):
    """The whole-session timeline and feature chart. Static, so they are drawn
    once per session instead of once per animation frame."""
    import io
    V = load_viewer(path)
    out = []
    for fig in (PL.draw_timeline(V["stage"], V["mov"], V["meta"]["movements"],
                                 V["flags"], V["meta"]["gestureFlags"],
                                 frame=None, nbody=V["nbody"]),
                PL.draw_features(V["traces"], list(feat_key), frame=None,
                                 normalise=True)):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        out.append(buf.getvalue())
        import matplotlib.pyplot as plt
        plt.close(fig)
    return out


@st.cache_data(max_entries=6, show_spinner=False)
def window_payload(path, start, stop, feat_key):
    V = load_viewer(path)
    return PLY.pack_window(V["raw"], V["conf"], V["traces"], list(feat_key),
                           start, stop, V["meta"]["joints"], V["cube"], K.FPS,
                           V["stage"], V["mov"], V["meta"]["movements"],
                           V["nbody"])


@st.cache_data(show_spinner="Computing session summaries…")
def get_summary(path):
    rec = K.load_npz(path)
    prep = K.prepare(rec, alchemy_only=True)
    tr = K.compute_traces(prep)
    return dict(scalars=K.session_summary(prep, tr),
                dist=K.distribution_table(tr),
                qc={k: v for k, v in K.qc_metrics(rec).items() if k != "checks"})



# =========================================================================
# joint colour legend
# =========================================================================
JOINT_GROUPS = [
    ("Spine and pelvis", ["Pelvis", "SpineNavel", "SpineChest", "Neck"]),
    ("Head and face", ["Head", "Nose", "EyeLeft", "EarLeft", "EyeRight", "EarRight"]),
    ("Left arm", ["ClavicleLeft", "ShoulderLeft", "ElbowLeft", "WristLeft",
                  "HandLeft", "HandTipLeft", "ThumbLeft"]),
    ("Right arm", ["ClavicleRight", "ShoulderRight", "ElbowRight", "WristRight",
                   "HandRight", "HandTipRight", "ThumbRight"]),
    ("Left leg", ["HipLeft", "KneeLeft", "AnkleLeft", "FootLeft"]),
    ("Right leg", ["HipRight", "KneeRight", "AnkleRight", "FootRight"]),
]


def joint_legend_html():
    import matplotlib.colors as mcolors
    used = set(K.ANGLE_JOINTS)
    cols = []
    for title, names in JOINT_GROUPS:
        items = ""
        for n in names:
            hexc = mcolors.to_hex(K.JOINT_COLORS.get(n, "black"))
            star = ' <b title="feeds the feature computation">&#9679;</b>' if n in used else ""
            items += (f'<div class="ji"><i class="sw" style="background:{hexc}"></i>'
                      f'{n}{star}</div>')
        cols.append(f'<div class="jg"><div class="jt">{title}</div>{items}</div>')
    return f"""
<style>
  .jwrap{{display:flex;flex-wrap:wrap;gap:1.1rem 2.1rem;margin:.3rem 0 .9rem 0;}}
  .jg{{min-width:9.5rem;}}
  .jt{{font-size:.74rem;font-weight:700;color:#1f2733;text-transform:uppercase;
       letter-spacing:.04em;margin-bottom:.32rem;}}
  .ji{{font-size:.78rem;color:#4b5561;line-height:1.55;}}
  .ji .sw{{display:inline-block;width:10px;height:10px;border-radius:3px;
          margin-right:.42rem;vertical-align:-1px;}}
  .ji b{{color:#2b5a8f;font-size:.66rem;}}
  .jnote{{font-size:.8rem;color:#5b6672;line-height:1.55;max-width:58rem;}}
  .conf{{display:flex;gap:1.4rem;margin:.5rem 0 .8rem 0;font-size:.78rem;
        color:#4b5561;align-items:center;}}
  .conf span i{{display:inline-block;width:13px;height:13px;border-radius:50%;
       background:#2b5a8f;margin-right:.4rem;vertical-align:-2px;}}
</style>
<div class="jwrap">{''.join(cols)}</div>
<div class="conf">
  <span><i style="opacity:1.00"></i>High confidence, opacity 1.00</span>
  <span><i style="opacity:0.60"></i>Medium confidence, opacity 0.60</span>
  <span><i style="opacity:0.25"></i>Low confidence, opacity 0.25</span>
</div>
<div class="jnote">
The Kinect reports all 32 joints on every frame, each with an <i>x / y / z</i>
position, an orientation quaternion and a tracking-confidence level. In the viewer
the <b>bones are drawn first, in the background</b>, and the <b>joint markers on
top, in the foreground</b>, so no marker is hidden behind a segment. A marker's
opacity is its tracking confidence, so you can see the tracker losing a joint as it
fades rather than disappearing.<br><br>
All 32 joints are drawn, but <b>only the 14 marked &#9679; are used to compute the
twelve features</b>, and only on frames where they were tracked at Medium
confidence or better. Anything below that is set to missing, gaps up to five frames
are interpolated, and the remainder is low-pass filtered at 5 Hz before any angle
is measured.
</div>
"""


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
    st.caption("Feature selection sits beside the viewer, and the playback "
               "controls are inside the player itself.")

V = load_viewer(PATHS[name])
N = V["n"]
meta, qc = V["meta"], V["qc"]

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

_names = ["Session viewer"]
if SHOW_QUALITY_CONTROL:
    _names.append("Quality control")
_names.append("Feature distributions")
if SHOW_CROSS_SESSION:
    _names.append("Cross-session comparison")
_tabs = dict(zip(_names, st.tabs(_names)))
tab_view = _tabs["Session viewer"]
tab_dist = _tabs["Feature distributions"]
# hidden tabs still execute, into a container that is never rendered
tab_qc = _tabs.get("Quality control", st.container())
tab_cmp = _tabs.get("Cross-session comparison", st.container())

# =========================================================================
# TAB 1 - viewer
# =========================================================================
with tab_view:
    blocks = V["blocks"]
    total_s = N / K.FPS

    view_col, feat_col = st.columns([4.1, 1.25], gap="medium")

    # ---- feature picker, beside the viewer rather than in the sidebar ----
    with feat_col:
        st.markdown("**Kinematic features**")
        if "feat_sel" not in st.session_state:
            st.session_state["feat_sel"] = ["shoulder_flexext_rom_R",
                                            "elbow_rom_R", "trunk_comp_R"]
        ba, bb = st.columns(2)
        if ba.button("All 12", use_container_width=True):
            st.session_state["feat_sel"] = list(K.TARGET_12_FEATURES)
        if bb.button("Clear", use_container_width=True):
            st.session_state["feat_sel"] = []
        selected = st.multiselect(
            "Overlay on the graph", K.TARGET_12_FEATURES, key="feat_sel",
            format_func=lambda k: K.FEATURE_DISPLAY_NAMES[k],
            label_visibility="collapsed")
        normalise = st.toggle("Normalise to 0-1", value=True,
                              help="Puts every selected feature on a shared "
                                   "0-1 axis so their shapes can be compared. "
                                   "Turn it off to read raw units, which is "
                                   "only useful for a single feature.")
        st.markdown('<span class="cap">The panel inside the player shows these '
                    "features across the loaded window; the chart further down "
                    "shows them across the whole recording.</span>",
                    unsafe_allow_html=True)

    # ---- viewer: window controls, scrubber and player, kept together -----
    with view_col:
        WINDOWS = [("10 s", 10), ("20 s", 20), ("30 s", 30), ("1 min", 60),
                   ("2 min", 120), ("5 min", 300), ("Full session", None)]
        c = st.columns([1.25, 1.5, 0.85, 0.85])
        win_lbl = c[0].selectbox("Zoom", [w[0] for w in WINDOWS], index=2)
        win_s = dict(WINDOWS)[win_lbl]
        mov_choice = c[1].selectbox("Jump to", ["(any)"] + meta["movements"])
        full = win_s is None
        max_start = 0 if full else max(0, int(total_s - win_s))

        if "win_start" not in st.session_state:
            st.session_state.win_start = 0
        st.session_state.win_start = int(
            np.clip(st.session_state.win_start, 0, max_start))

        LEAD_S = 2          # the window opens two seconds before the block starts

        def _jump(direction):
            # the reference point is the block we are sitting on, not the window
            # edge, otherwise the same block is selected again every time
            # the slider is whole seconds, so the reconstructed position can sit
            # up to a second before the block it is anchored on: allow for that
            cur = (st.session_state.win_start + LEAD_S) * K.FPS
            tol = K.FPS
            cand = sorted(b[0] for b in blocks if mov_choice in ("(any)", b[2]))
            nxt = ([f for f in cand if f > cur + tol] if direction > 0
                   else [f for f in cand if f < cur - tol])
            if nxt:
                tgt = nxt[0] if direction > 0 else nxt[-1]
                st.session_state.win_start = int(
                    np.clip(tgt / K.FPS - LEAD_S, 0, max_start))

        c[2].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        c[3].markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if c[2].button("Prev", use_container_width=True):
            _jump(-1)
        if c[3].button("Next", use_container_width=True):
            _jump(+1)

        if full:
            a_i, b_i, start_s = 0, N, 0
            st.markdown('<span class="cap">The whole recording is loaded. Frames '
                        "are thinned so it fits in the browser; the counter still "
                        "shows the true frame numbers.</span>",
                        unsafe_allow_html=True)
        else:
            start_s = st.slider("Window start (seconds into the recording)",
                                0, max(1, max_start), key="win_start")
            a_i = int(start_s * K.FPS)
            b_i = min(N, a_i + int(win_s * K.FPS))

        with st.spinner("Preparing…"):
            payload = window_payload(PATHS[name], a_i, b_i, tuple(selected))
        components.html(PLY.html(payload), height=700, scrolling=False)

        # say plainly where the jump landed: with a long window the start can
        # move a long way without the picture changing much
        if not full and blocks:
            def _ms(fr):
                t = int(fr / K.FPS)
                return f"{t // 60:d}:{t % 60:02d}"
            inside, seen = [], []
            for x in blocks:
                if x[0] < b_i and x[1] > a_i:
                    inside.append(x)
                    if x[2] not in seen:
                        seen.append(x[2])
            nxt = next((x for x in blocks if x[0] >= a_i), None)
            bits = [f"showing {_ms(a_i)} to {_ms(b_i)}"]
            bits.append(f"{len(inside)} movement block(s) in view: " + ", ".join(seen)
                        if inside else "no movement block in view")
            if nxt is not None:
                bits.append(f"next block {_ms(nxt[0])} ({nxt[2]}, "
                            f"{blocks.index(nxt) + 1} of {len(blocks)})")
            st.markdown('<span class="cap">' + " &nbsp;·&nbsp; ".join(bits)
                        + "</span>", unsafe_allow_html=True)

        if selected:
            st.markdown('<span class="cap">Traces shown: '
                        + " &nbsp;·&nbsp; ".join(
                            f"<b>{K.FEATURE_DISPLAY_NAMES[k]}</b> - "
                            f"{K.TRACE_MEANING[k]}" for k in selected)
                        + "</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Whole recording")
    st.markdown(f'<span class="cap">The shaded band marks the {win_s} s window '
                "loaded into the player above.</span>", unsafe_allow_html=True)
    tl_png, ft_png = session_charts(PATHS[name], tuple(selected))
    st.image(tl_png, use_container_width=True)
    st.image(ft_png, use_container_width=True)

    with st.expander("All 32 tracked joints, and how the skeleton is drawn"):
        st.markdown(joint_legend_html(), unsafe_allow_html=True)

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
    st.dataframe(df.style.format(precision=3, na_rep="-"), hide_index=True,
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
    st.dataframe(tab.style.format(precision=3, na_rep="-", subset=order),
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

**Gaps in the traces.** A trace goes blank wherever a frame was removed by the
frame-level filter: anything outside the gameplay phase, or any frame where the
sensor reported more than one body. Nothing is interpolated across those stretches,
and the player shades them grey so a blank stretch reads as excluded rather than
missing.

**Reading the player.** Bones are drawn in the background and joint markers in the
foreground, with marker opacity encoding the Kinect tracking-confidence level
(Medium 0.6, Low 0.25). Only Medium-confidence joints enter the feature
computation. The two thin ribbons beneath the feature panel are the game phase and
the movement block. Playback is drawn in the browser, so it costs no server
round-trips.

**Long spans.** Windows up to two minutes play every recorded frame. Longer spans,
including the whole recording, are thinned by an integer step so the data still
fits in the browser: a 23-minute session loads as about 5,000 frames, one in eight.
The clock and the frame counter always refer to the true frame numbers of the
original recording, and the readout states how much thinning is in effect.

**Playback.** Speed runs from 0.25x to 10x real time, default 2x, with an optional
repeat. The space bar toggles play, the arrow buttons step one frame, Restart
returns to the top of the span, and dragging the scrubber pauses first.

**Files.** Each session also exists as `<name>.json.gz`, the complete recording in
the original schema, and `<name>_excerpt.json`, a short pretty-printed excerpt. An
uncompressed 20-minute session is roughly 200 MB, which is why the full files are
gzipped and the application reads compact `.npz` arrays instead.
    """)
