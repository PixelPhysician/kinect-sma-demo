"""
player.py — client-side playback component.

The server prepares one window of the session once, ships it to the browser as
packed binary, and the browser animates it on a canvas at the real frame rate.
Nothing is redrawn on the server while the animation runs, so playback costs no
round-trips at all.

Packing:
    positions   int16, millimetres, shape (F, J, 3), -32768 marks missing
    confidence  uint8, shape (F, J), 0 = Low, 1 = Medium, 2 = High
    features    uint16, 0..65535 mapped onto each feature's own min..max,
                shape (F, S), 65535 alone is not used as a sentinel because a
                separate validity mask is cheaper to read
"""

import base64
import json

import matplotlib.colors as mcolors
import numpy as np

import kinematics as K

MISSING = -32768


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


MAX_PLAYER_FRAMES = 5400        # about 3 min at full rate; longer spans decimate


def pack_window(raw, conf, traces, selected, start, stop, joints, cube, fps,
                stage=None, mov=None, movements=None, nbody=None,
                max_frames=MAX_PLAYER_FRAMES):
    """
    Builds the JSON payload for one span of the recording.

    Spans longer than `max_frames` are decimated by an integer step so that the
    whole session can be loaded without shipping tens of megabytes. The step is
    reported back so the clock and the frame counter still refer to the true
    frame numbers of the original recording.
    """
    step = max(1, -(-(stop - start) // max_frames))
    sl = slice(start, stop, step)
    pos = raw[sl] * 1000.0                               # metres -> millimetres
    bad = ~np.isfinite(pos).all(axis=2)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.clip(np.rint(pos), -32767, 32767).astype(np.int16)
    pos[bad] = MISSING

    jidx = {n: i for i, n in enumerate(joints)}
    bones = [[jidx[a], jidx[b]] for a, b in K.SKELETON_EDGES
             if a in jidx and b in jidx]
    bone_col = [mcolors.to_hex(K.JOINT_COLORS.get(a, "black"))
                for a, b in K.SKELETON_EDGES if a in jidx and b in jidx]
    joint_col = [mcolors.to_hex(K.JOINT_COLORS.get(n, "black")) for n in joints]

    feats, fmeta = [], []
    for key in selected:
        v = np.asarray(traces[key][sl], float)
        lo = np.nanmin(traces[key]) if np.isfinite(traces[key]).any() else 0.0
        hi = np.nanmax(traces[key]) if np.isfinite(traces[key]).any() else 1.0
        span = hi - lo if hi - lo > 1e-12 else 1.0
        norm = np.clip((v - lo) / span, 0, 1)
        ok = np.isfinite(v)
        feats.append(np.where(ok, np.rint(norm * 65534), 65535).astype(np.uint16))
        fmeta.append({"key": key, "name": K.FEATURE_DISPLAY_NAMES[key],
                      "colour": K.FEATURE_COLORS[key], "lo": float(lo),
                      "span": float(span),
                      "unit": ""})
    feats = (np.stack(feats, axis=1) if feats
             else np.zeros((stop - start, 0), np.uint16))

    band = []
    if stage is not None:
        seg = K.stage_segments(np.asarray(stage[sl]))
        band = [{"a": int(a), "b": int(b),
                 "c": K.STAGE_COLORS.get(int(s), "#cccccc"),
                 "n": K.STAGE_NAMES.get(int(s), str(s))} for a, b, s in seg]
    mband = []
    if mov is not None and movements:
        seg = K.stage_segments(np.asarray(mov[sl]).astype(np.int16))
        mband = [{"a": int(a), "b": int(b),
                  "c": K.MOVEMENT_COLORS.get(movements[m], "#999"),
                  "n": K.MOVEMENT_LABELS.get(movements[m], movements[m])}
                 for a, b, m in seg if m >= 0]

    # frames the pipeline discards: outside gameplay, or more than one body
    xband = []
    if stage is not None:
        excl = np.asarray(stage[sl]) != K.ALCHEMY_STAGE
        if nbody is not None:
            excl |= np.asarray(nbody[sl]) != 1
        xband = [{"a": int(a), "b": int(b)}
                 for a, b, v in K.stage_segments(excl.astype(np.int8)) if v]

    return {
        "F": int(pos.shape[0]), "J": len(joints),
        "fps": float(fps) / step, "srcFps": float(fps), "step": int(step),
        "offset": int(start),
        "pos": _b64(pos), "conf": _b64(conf[sl].astype(np.uint8)),
        "feat": _b64(feats), "S": int(feats.shape[1]),
        "bones": bones, "boneCol": bone_col, "jointCol": joint_col,
        "cube": {k: [float(v[0]), float(v[1])] for k, v in cube.items()},
        "fmeta": fmeta, "band": band, "mband": mband, "xband": xband,
    }


# =========================================================================
_HTML = r"""
<div id="wrap">
  <div id="bar">
    <button id="play" class="prim">Play</button>
    <button id="back">&#8592;</button>
    <button id="fwd">&#8594;</button>
    <button id="rst">Restart</button>
    <label class="lb">Speed
      <select id="spd">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1">1x</option>
        <option value="2" selected>2x</option>
        <option value="4">4x</option>
        <option value="10">10x</option>
      </select>
    </label>
    <label class="lb"><input type="checkbox" id="loop" checked> Loop</label>
    <span id="clock" class="clock">00:00.00</span>
    <span id="phase" class="ph"></span>
  </div>
  <input type="range" id="scrub" min="0" max="1" value="0" step="1">
  <canvas id="skel"></canvas>
  <canvas id="feat"></canvas>
  <div id="legend"></div>
</div>
<style>
  #wrap{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2733;}
  #bar{display:flex;align-items:center;gap:.55rem;margin:0 0 .45rem 0;flex-wrap:wrap;}
  #bar button{border:1px solid #d3d9e0;background:#fff;border-radius:7px;
    padding:.3rem .8rem;font-size:.83rem;cursor:pointer;color:#1f2733;}
  #bar button:hover{background:#f2f5f9;}
  #bar button.prim{background:#2b5a8f;border-color:#2b5a8f;color:#fff;min-width:92px;}
  #bar button.prim:hover{background:#224975;}
  .lb{font-size:.78rem;color:#6b7683;display:flex;align-items:center;gap:.3rem;}
  .lb select{font-size:.78rem;padding:.16rem .3rem;border-radius:6px;
    border:1px solid #d3d9e0;background:#fff;color:#1f2733;}
  .clock{font-variant-numeric:tabular-nums;font-size:.83rem;color:#1f2733;
    margin-left:.3rem;font-weight:600;}
  .ph{font-size:.76rem;color:#6b7683;}
  #scrub{width:100%;margin:.1rem 0 .5rem 0;accent-color:#2b5a8f;}
  canvas{width:100%;display:block;border-radius:9px;background:#fff;}
  #feat{margin-top:.45rem;}
  #legend{display:flex;flex-wrap:wrap;gap:.55rem 1.1rem;margin-top:.4rem;
    font-size:.76rem;color:#5b6672;}
  #legend span b{font-variant-numeric:tabular-nums;color:#1f2733;}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;
    margin-right:.32rem;vertical-align:-1px;}
</style>
<script>
const D = __PAYLOAD__;

function unb64(s, Type){
  const bin = atob(s), n = bin.length, u = new Uint8Array(n);
  for (let i=0;i<n;i++) u[i]=bin.charCodeAt(i);
  return new Type(u.buffer);
}
const MISSING = -32768;
const POS  = unb64(D.pos , Int16Array);
const CONF = unb64(D.conf, Uint8Array);
const FEAT = unb64(D.feat, Uint16Array);
const ALPHA = [0.25, 0.60, 1.00];

const skel = document.getElementById('skel');
const feat = document.getElementById('feat');
const sc = skel.getContext('2d'), fc = feat.getContext('2d');
const scrub = document.getElementById('scrub');
const playB = document.getElementById('play');
const clock = document.getElementById('clock');
const phase = document.getElementById('phase');
const legend = document.getElementById('legend');
scrub.max = D.F - 1;

let frame = 0, playing = false, last = 0, acc = 0;
let W = 0, PH = 0, FH = 150;
const BANDH = 6;      // strip reserved at the foot of the feature panel

function resize(){
  const cssW = skel.parentElement.clientWidth || 900;
  const dpr = window.devicePixelRatio || 1;
  W = cssW; PH = Math.round(Math.min(cssW/3, 360));
  skel.width = cssW*dpr; skel.height = PH*dpr;
  skel.style.height = PH+'px';
  sc.setTransform(dpr,0,0,dpr,0,0);
  feat.width = cssW*dpr; feat.height = FH*dpr;
  feat.style.height = FH+'px';
  fc.setTransform(dpr,0,0,dpr,0,0);
  drawFeatBase(); draw();
}
window.addEventListener('resize', resize);

// ---- three orthogonal projections, shared cubic range ----
const VIEWS = [
  {t:'Front view (x vs -y)', ax:'x', ay:'y', gx:o=>o[0], gy:o=>o[1],
   xl:'Left \u2190 x \u2192 Right', yl:'Down \u2190 -y \u2192 Up'},
  {t:'Side view (z vs -y)',  ax:'z', ay:'y', gx:o=>o[2], gy:o=>o[1],
   xl:'Front \u2190 z \u2192 Back', yl:'Down \u2190 -y \u2192 Up'},
  {t:'Top view (x vs z)',    ax:'x', ay:'z', gx:o=>o[0], gy:o=>o[2],
   xl:'Left \u2190 x \u2192 Right', yl:'Back \u2190 z \u2192 Front'}
];

function draw(){
  sc.clearRect(0,0,W,PH);
  const pw = W/3, ML = 38, MR = 10, MT = 19, MB = 31;
  const base = frame*D.J*3;

  for (let v=0; v<3; v++){
    const V = VIEWS[v], ox = v*pw;
    const rx = D.cube[V.ax], ry = D.cube[V.ay];
    const x0 = ox + ML, y0 = MT;
    const iw = pw - ML - MR, ih = PH - MT - MB;
    const s = Math.min(iw/(rx[1]-rx[0]), ih/(ry[1]-ry[0]));   // equal aspect
    const cx = x0 + iw/2, cy = y0 + ih/2;
    const mx = (rx[0]+rx[1])/2, my = (ry[0]+ry[1])/2;
    const PX = u => cx + (u - mx)*s;
    const PY = u => cy - (u - my)*s;                          // canvas y grows down

    // grid and numbered ticks
    sc.strokeStyle='#e8ecf1'; sc.lineWidth=1;
    sc.fillStyle='#98a1ad'; sc.font='9px sans-serif';
    for (let g=0; g<=4; g++){
      const gx = rx[0]+(rx[1]-rx[0])*g/4, gy = ry[0]+(ry[1]-ry[0])*g/4;
      sc.beginPath(); sc.moveTo(PX(gx), y0); sc.lineTo(PX(gx), y0+ih); sc.stroke();
      sc.beginPath(); sc.moveTo(x0, PY(gy)); sc.lineTo(x0+iw, PY(gy)); sc.stroke();
      sc.textAlign='center'; sc.textBaseline='top';
      sc.fillText(gx.toFixed(1), PX(gx), y0+ih+4);
      sc.textAlign='right'; sc.textBaseline='middle';
      sc.fillText(gy.toFixed(1), x0-4, PY(gy));
    }
    // axis spines
    sc.strokeStyle='#c8cfd8'; sc.lineWidth=1;
    sc.beginPath(); sc.moveTo(x0, y0); sc.lineTo(x0, y0+ih); sc.lineTo(x0+iw, y0+ih);
    sc.stroke();

    sc.save();
    sc.beginPath(); sc.rect(x0, y0, iw, ih); sc.clip();

    const gxv = o => V.gx(o)/1000, gyv = o => V.gy(o)/1000;
    const P = j => {
      const k = base + j*3;
      if (POS[k]===MISSING) return null;
      return [POS[k], POS[k+1], POS[k+2]];
    };
    // bones first: background
    sc.lineCap='round'; sc.lineWidth=2.4;
    for (let b=0; b<D.bones.length; b++){
      const A = P(D.bones[b][0]), B = P(D.bones[b][1]);
      if (!A || !B) continue;
      sc.globalAlpha = 0.55; sc.strokeStyle = D.boneCol[b];
      sc.beginPath();
      sc.moveTo(PX(gxv(A)), PY(gyv(A)));
      sc.lineTo(PX(gxv(B)), PY(gyv(B)));
      sc.stroke();
    }
    // joints on top: foreground, opacity = tracking confidence
    for (let j=0; j<D.J; j++){
      const A = P(j); if (!A) continue;
      sc.globalAlpha = ALPHA[CONF[frame*D.J + j]] ?? 0.25;
      sc.fillStyle = D.jointCol[j];
      sc.beginPath(); sc.arc(PX(gxv(A)), PY(gyv(A)), 3.3, 0, 6.2832); sc.fill();
    }
    sc.globalAlpha = 1;
    sc.restore();

    sc.fillStyle='#1f2733'; sc.font='600 11px sans-serif';
    sc.textAlign='center'; sc.textBaseline='alphabetic';
    sc.fillText(V.t, ox+pw/2, 13);
    sc.fillStyle='#8a93a0'; sc.font='9px sans-serif';
    sc.fillText(V.xl, x0+iw/2, PH-5);
    sc.save();
    sc.translate(ox+10, y0+ih/2); sc.rotate(-Math.PI/2);
    sc.textAlign='center'; sc.textBaseline='middle';
    sc.fillText(V.yl, 0, 0);
    sc.restore();
  }
  drawCursor();
  updateReadout();
}

// ---- feature panel: static traces drawn once, cursor drawn per frame ----
let featBase = null;
function drawFeatBase(){
  const c = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  c.width = W*dpr; c.height = FH*dpr;
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,W,FH);
  // frames the pipeline discards are shaded, so a blank stretch reads as
  // "excluded" rather than "missing"
  const XX = i => (D.F<2) ? 0 : i*(W-2)/(D.F-1) + 1;
  g.fillStyle='#eef1f5';
  for (const b of (D.xband||[])) g.fillRect(XX(b.a), 0, Math.max(1, XX(b.b)-XX(b.a)), FH-BANDH-2);
  g.strokeStyle='#eef1f5'; g.lineWidth=1;
  for (let k=0;k<=4;k++){ const y=8+(FH-18-BANDH)*k/4;
    g.beginPath(); g.moveTo(0,y); g.lineTo(W,y); g.stroke(); }
  const X = i => (D.F<2) ? 0 : i*(W-2)/(D.F-1) + 1;
  const Y = v => 8 + (FH-18-BANDH)*(1-v);
  for (let s=0; s<D.S; s++){
    g.strokeStyle = D.fmeta[s].colour; g.lineWidth = 1.5;
    g.beginPath(); let pen = false;
    for (let i=0;i<D.F;i++){
      const raw = FEAT[i*D.S + s];
      if (raw === 65535){ pen = false; continue; }
      const x = X(i), y = Y(raw/65534);
      if (!pen){ g.moveTo(x,y); pen = true; } else g.lineTo(x,y);
    }
    g.stroke();
  }
  featBase = c;
}
function drawCursor(){
  fc.clearRect(0,0,W,FH);
  if (featBase) fc.drawImage(featBase, 0, 0, W, FH);
  const x = (D.F<2) ? 0 : frame*(W-2)/(D.F-1) + 1;

  // game phase along the bottom. The movement ribbon is deliberately not drawn
  // any more, though D.mband is still carried so the readout can name the task.
  for (const b of D.band){
    fc.fillStyle=b.c; fc.globalAlpha=0.5;
    fc.fillRect((D.F<2?0:b.a*(W-2)/(D.F-1)), FH-BANDH,
                Math.max(1,(b.b-b.a)*(W-2)/Math.max(1,D.F-1)), 4);
  }
  fc.globalAlpha=1;
  fc.strokeStyle='#1f2733'; fc.lineWidth=1.2;
  fc.beginPath(); fc.moveTo(x,0); fc.lineTo(x,FH-BANDH-2); fc.stroke();
  for (let s=0;s<D.S;s++){
    const raw = FEAT[frame*D.S + s];
    if (raw === 65535) continue;
    fc.fillStyle = D.fmeta[s].colour;
    fc.beginPath();
    fc.arc(x, 8 + (FH-18-BANDH)*(1-raw/65534), 3.4, 0, 6.2832); fc.fill();
    fc.strokeStyle='#fff'; fc.lineWidth=1; fc.stroke();
  }
}

function trueFrame(i){ return D.offset + i*D.step; }
function updateReadout(){
  const t = trueFrame(frame)/D.srcFps;
  const m = Math.floor(t/60), s = t - m*60;
  clock.textContent = String(m).padStart(2,'0')+':'+s.toFixed(2).padStart(5,'0');
  let ph = '';
  for (const b of D.band) if (frame>=b.a && frame<b.b) ph = b.n;
  let mv = '';
  for (const b of D.mband) if (frame>=b.a && frame<b.b) mv = b.n;
  phase.textContent = 'frame ' + (trueFrame(frame)+1).toLocaleString()
                    + (ph ? '  \u00b7  ' + ph : '') + (mv ? '  \u00b7  ' + mv : '');
  let h = '';
  for (let s2=0;s2<D.S;s2++){
    const raw = FEAT[frame*D.S + s2];
    const m2 = D.fmeta[s2];
    // no placeholder dash when the trace has no value here: the name alone
    const val = (raw===65535) ? '' : (m2.lo + (raw/65534)*m2.span).toFixed(3);
    h += '<span><i class="sw" style="background:'+m2.colour+'"></i>'
       + m2.name + (val ? ' <b>' + val + '</b>' : '') + '</span>';
  }
  legend.innerHTML = h;
}

// ---- transport ----
function setPlaying(p){
  playing = p; playB.textContent = p ? 'Pause' : 'Play';
  if (p){ if (frame >= D.F-1) frame = 0;
          last = performance.now(); acc = 0; requestAnimationFrame(tick); }
}
function tick(now){
  if (!playing) return;
  const spd = parseFloat(document.getElementById('spd').value);
  acc += (now - last)/1000 * D.fps * spd;
  last = now;
  if (acc >= 1){
    const step = Math.floor(acc); acc -= step;
    frame += step;
    if (frame >= D.F){
      if (document.getElementById('loop').checked){ frame %= D.F; }
      else { frame = D.F-1; scrub.value = frame; draw(); setPlaying(false); return; }
    }
    scrub.value = frame;
    draw();
  }
  requestAnimationFrame(tick);
}
playB.onclick = () => setPlaying(!playing);
document.getElementById('back').onclick = () => { setPlaying(false); frame=Math.max(0,frame-1); scrub.value=frame; draw(); };
document.getElementById('fwd').onclick  = () => { setPlaying(false); frame=Math.min(D.F-1,frame+1); scrub.value=frame; draw(); };
document.getElementById('rst').onclick  = () => { setPlaying(false); frame=0; scrub.value=0; draw(); };
scrub.oninput = e => { setPlaying(false); frame = +e.target.value; draw(); };
document.addEventListener('keydown', e => {
  if (e.code === 'Space'){ e.preventDefault(); setPlaying(!playing); }
});

resize();
</script>
"""


def html(payload):
    return _HTML.replace("__PAYLOAD__", json.dumps(payload))
