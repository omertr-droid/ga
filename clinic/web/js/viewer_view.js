// viewer_view.js — the 3-panel reader + a single "GA area (OCT)" card. OCT-only: no PLEX.
//   LEFT   IR localizer with REAL per-B-scan locator lines (current one highlighted green).
//   MIDDLE B-scan scroller + 2 checkboxes: show BM, highlight predicted-GA A-scans.
//   RIGHT  the en-face projection with the predicted GA (translucent green) toggle.
// Clicking the projection jumps the B-scan to that location (probe) without occluding the B-scan.
// The Bruch's-membrane source was chosen at upload, so this screen shows one fixed number (no DL toggle).
import { api } from './api.js';
import { el, loadImage, fmtArea } from './lib.js';

const MMPP = 6 / 512;                 // en-face mm/px (matches the pipeline's ENFACE_MMPP)

export function mountViewer(root, vid, meta) {
  const N = meta.n_bscans, W = meta.W, H = meta.H, OUT = meta.enface_out;
  const AX = meta.axial_um_per_px;
  let idx = Math.floor(N / 2);        // land on the central B-scan (crosses the fovea)
  let showBM = true, showGA = true;   // B-scan overlays
  let showProjGA = false;             // en-face overlay OFF by default (legend toggle)
  let probe = null;                   // { bscan, col, fx, fy }

  // data (loaded once)
  let lines = [], bmRows = [], gaIv = [], slab = meta.slab_um || [10, 340];
  let fieldInvalid = [];                               // per-B-scan [start,end] saturated-band runs
  let locImg = null, projImg = null, gaImg = null;
  const bscanCache = new Map();

  // ---------------- DOM ----------------
  const dOac = el('b', {}, '');
  const dash = el('div', { class: 'dash' },
    el('div', { class: 'dash-card oac' }, el('span', {}, 'GA area (OCT)'), dOac));

  // left: localizer
  const locCanvas = el('canvas', { class: 'loc-canvas' });
  const lctx = locCanvas.getContext('2d');

  // middle: b-scan
  const imgCanvas = el('canvas', { width: W, height: H, class: 'bscan-img' });
  const ovCanvas = el('canvas', { width: W, height: H, class: 'bscan-ov' });
  const bwrap = el('div', { class: 'bscan-wrap' }, imgCanvas, ovCanvas);
  const ictx = imgCanvas.getContext('2d'), octx = ovCanvas.getContext('2d');
  const numLabel = el('span', { class: 'bnum' });
  const slider = el('input', { type: 'range', min: 0, max: N - 1, value: idx, step: 1 });
  const prev = el('button', { class: 'ghost' }, '‹');
  const next = el('button', { class: 'ghost' }, '›');
  const cbBM = el('input', { type: 'checkbox' }); cbBM.checked = showBM;
  const cbGA = el('input', { type: 'checkbox' }); cbGA.checked = showGA;
  const checks = el('div', { class: 'checks' },
    el('label', { class: 'toggle' }, cbBM, el('span', {}, 'BM segmentation')),
    el('label', { class: 'toggle' }, cbGA, el('span', {}, 'Highlight predicted GA')));

  // right: projection (+ its legend = show/hide toggle)
  const projCanvas = el('canvas', { width: OUT, height: OUT, class: 'proj-canvas' });
  const pctx = projCanvas.getContext('2d');
  const cbPGA = el('input', { type: 'checkbox' }); cbPGA.checked = showProjGA;

  const layout = el('section', { class: 'screen viewer' }, dash,
    el('div', { class: 'panels' },
      el('div', { class: 'panel' }, el('h3', {}, 'Localizer'), locCanvas,
        el('div', { class: 'hint' }, 'green line = current B-scan · click a line or scroll to navigate')),
      el('div', { class: 'panel' }, el('div', { class: 'panel-head' }, el('h3', {}, 'B-scan'), numLabel),
        checks, bwrap,
        el('div', { class: 'foot' }, prev, slider, next)),
      el('div', { class: 'panel' }, el('h3', {}, 'En-face projection'), projCanvas,
        el('div', { class: 'legend' },
          el('label', { class: 'legend-item' }, cbPGA, el('span', { class: 'sw green' }), el('span', {}, 'predicted GA'))),
        el('div', { class: 'hint' }, 'click anywhere to jump the B-scan there'))));
  root.append(layout);

  // ---------------- drawing: localizer ----------------
  function drawLocalizer() {
    if (!locImg) return;
    const LW = locImg.naturalWidth, LH = locImg.naturalHeight;
    if (locCanvas.width !== LW) { locCanvas.width = LW; locCanvas.height = LH; }
    lctx.clearRect(0, 0, LW, LH);
    lctx.drawImage(locImg, 0, 0);
    for (let i = 0; i < lines.length; i++) {
      const [x1, y1, x2, y2] = lines[i];
      lctx.strokeStyle = (i === idx) ? 'rgb(0,255,0)' : 'rgba(255,255,255,0.20)';
      lctx.lineWidth = (i === idx) ? 2.2 : 1;
      lctx.beginPath(); lctx.moveTo(x1, y1); lctx.lineTo(x2, y2); lctx.stroke();
    }
  }
  function locClick(e) {
    if (!locImg || !lines.length) return;
    const r = locCanvas.getBoundingClientRect();
    const y = (e.clientY - r.top) * (locImg.naturalHeight / r.height);
    let best = Infinity, bi = idx;
    for (let i = 0; i < lines.length; i++) {
      const my = (lines[i][1] + lines[i][3]) / 2, d = Math.abs(my - y);
      if (d < best) { best = d; bi = i; }
    }
    setIndex(bi);
  }
  locCanvas.addEventListener('click', locClick);
  locCanvas.addEventListener('wheel', (e) => { e.preventDefault(); setIndex(idx + (e.deltaY > 0 ? 1 : -1)); }, { passive: false });

  // ---------------- drawing: B-scan overlay ----------------
  function drawBscanOverlay() {
    octx.clearRect(0, 0, W, H);
    const bm = bmRows[idx];
    for (const [s, e] of (fieldInvalid[idx] || [])) {   // shade saturated out-of-field columns (excluded)
      octx.fillStyle = 'rgba(255,70,70,0.20)';
      octx.fillRect(s, 0, (e - s + 1), H);
    }
    if (showGA && bm) {
      octx.fillStyle = 'rgba(0,210,0,0.30)';
      for (const [s, e] of (gaIv[idx] || [])) {
        octx.beginPath(); let started = false;
        for (let x = s; x < e; x++) {
          const b = bm[x]; if (b == null) continue;
          const yy = b + slab[0] / AX;
          if (!started) { octx.moveTo(x, yy); started = true; } else octx.lineTo(x, yy);
        }
        for (let x = e - 1; x >= s; x--) { const b = bm[x]; if (b != null) octx.lineTo(x, b + slab[1] / AX); }
        octx.closePath(); octx.fill();
      }
    }
    if (showBM && bm) {
      octx.strokeStyle = 'rgba(255,235,0,0.85)'; octx.lineWidth = 1.3;
      octx.beginPath(); let st = false;
      for (let x = 0; x < W; x++) {
        const b = bm[x];
        if (b == null) { st = false; continue; }
        if (!st) { octx.moveTo(x, b); st = true; } else octx.lineTo(x, b);
      }
      octx.stroke();
    }
    if (probe && probe.bscan === idx) drawEdgeTick(probe.col);
  }
  function drawEdgeTick(col) {                          // non-occluding: short ticks at top + bottom only
    const x = col + 0.5, len = Math.max(10, H * 0.07);
    octx.strokeStyle = 'rgb(0,229,255)'; octx.lineWidth = 1.6;
    octx.beginPath();
    octx.moveTo(x, 0); octx.lineTo(x, len);
    octx.moveTo(x, H - len); octx.lineTo(x, H);
    octx.stroke();
  }
  async function renderBscan() {
    let im = bscanCache.get(idx);
    if (!im) { im = await loadImage(api.bscanUrl(vid, idx)); bscanCache.set(idx, im); }
    ictx.clearRect(0, 0, W, H); ictx.drawImage(im, 0, 0, W, H);
    drawBscanOverlay();
  }

  // ---------------- drawing: projection ----------------
  function drawProjection() {
    pctx.clearRect(0, 0, OUT, OUT);
    if (projImg) pctx.drawImage(projImg, 0, 0, OUT, OUT);
    if (showProjGA && gaImg) pctx.drawImage(gaImg, 0, 0, OUT, OUT);
    if (probe) {
      const X = probe.fx * OUT, Y = probe.fy * OUT, r = Math.max(6, OUT / 36);
      pctx.strokeStyle = 'rgb(0,229,255)'; pctx.lineWidth = Math.max(1.5, OUT / 240);
      pctx.beginPath();
      pctx.moveTo(X - r, Y); pctx.lineTo(X + r, Y); pctx.moveTo(X, Y - r); pctx.lineTo(X, Y + r);
      pctx.stroke();
      pctx.beginPath(); pctx.arc(X, Y, r, 0, Math.PI * 2); pctx.stroke();
    }
  }
  function probeFromClick(e) {
    const r = projCanvas.getBoundingClientRect();
    if (r.width < 2) return null;
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
    const sx = (meta.fov_mm[0] / W) / MMPP, sy = (meta.fov_mm[1] / N) / MMPP;
    const cd = (OUT - 1) / 2;
    const xsrc = (fx * OUT - cd) / sx + (W - 1) / 2;
    const ysrc = (fy * OUT - cd) / sy + (N - 1) / 2;
    const col = Math.max(0, Math.min(W - 1, Math.round(xsrc)));
    // The en-face rows were reversed on the way out (the usual raster), so un-flip. A reverse-scanned
    // raster (003-016 / 003-130) is already fundus-ordered and must map straight through.
    const flipped = meta.enface_flip !== false;
    const row = flipped ? (N - 1) - ysrc : ysrc;
    const bscan = Math.max(0, Math.min(N - 1, Math.round(row)));
    return { bscan, col, fx, fy };
  }
  projCanvas.addEventListener('click', (e) => {
    const p = probeFromClick(e); if (!p) return;
    probe = p; setIndex(p.bscan); drawProjection();
  });

  // ---------------- navigation ----------------
  function updateNum() { numLabel.textContent = `${idx + 1} / ${N}`; }
  function setIndex(i) {
    const ni = Math.max(0, Math.min(N - 1, i | 0));
    if (ni === idx) { drawBscanOverlay(); drawLocalizer(); return; }
    idx = ni; slider.value = idx;
    updateNum(); renderBscan(); drawLocalizer();
  }
  slider.addEventListener('input', () => setIndex(+slider.value));
  prev.addEventListener('click', () => setIndex(idx - 1));
  next.addEventListener('click', () => setIndex(idx + 1));
  bwrap.addEventListener('wheel', (e) => { e.preventDefault(); setIndex(idx + (e.deltaY > 0 ? 1 : -1)); }, { passive: false });
  cbBM.addEventListener('change', () => { showBM = cbBM.checked; drawBscanOverlay(); });
  cbGA.addEventListener('change', () => { showGA = cbGA.checked; drawBscanOverlay(); });
  cbPGA.addEventListener('change', () => { showProjGA = cbPGA.checked; drawProjection(); });

  const onKey = (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { setIndex(idx - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { setIndex(idx + 1); e.preventDefault(); }
  };
  window.addEventListener('keydown', onKey);

  // ---------------- init ----------------
  (async () => {
    dOac.textContent = fmtArea(meta.oac_area_mm2);
    updateNum();
    try {
      const [ll, bmd, gn] = await Promise.all([api.locLines(vid), api.bm(vid), api.gaNative(vid)]);
      lines = ll.lines || [];
      bmRows = bmd.bm || []; slab = bmd.slab_um || slab;
      fieldInvalid = bmd.field_invalid || [];
      gaIv = gn.intervals || [];
    } catch (e) { /* panels still render what they can */ }
    [locImg, projImg, gaImg] = await Promise.all([
      loadImage(api.localizerUrl(vid)).catch(() => null),
      loadImage(api.projectionUrl(vid)).catch(() => null),
      loadImage(api.gaOverlayUrl(vid)).catch(() => null),
    ]);
    await renderBscan();
    drawLocalizer();
    drawProjection();
  })();

  return () => { window.removeEventListener('keydown', onKey); };
}
