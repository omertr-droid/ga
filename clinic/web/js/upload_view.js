// upload_view.js — the multi-step upload modal (a small finite state machine over one overlay):
//   [staging ->] choosing_file -> listing_scans -> choosing_scan -> choosing_bm -> processing -> done | error
// Step 1 (open) decodes the E2E and lists ONLY its 6x6-measurable scans (no GA yet). After the user
// picks one OR BOTH eyes and a Bruch's-membrane source per eye, step 2 (process) computes GA once per
// eye, records each, and opens the first.
// A dropped file enters at `staging`, which uploads the bytes and yields a server path; from there the
// flow is identical to a pasted/browsed path.
import { api } from './api.js';
import { el } from './lib.js';
import { openScan, notify } from './main.js';

let openModal = null;                 // singleton guard: only one upload modal at a time

function fmtSize(n) {
  if (n == null) return '';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + ' KB';
  return n + ' B';
}

export function openUpload() {
  if (openModal) return;
  openModal = buildModal();
}

// Entry point for a drag-and-drop: open (or reuse) the modal and jump straight to uploading the bytes.
// Returns false if a scan is mid-processing — replacing the modal then would strand that work.
export function openUploadWithFile(file) {
  if (openModal && openModal.busy()) return false;
  if (!openModal) openModal = buildModal();
  openModal.startWithFile(file);
  return true;
}

function buildModal() {
  const titleEl = el('h3', {}, 'Upload E2E');
  const body = el('div', { class: 'modal-body' });
  const foot = el('div', { class: 'modal-foot' });
  const closeBtn = el('button', { class: 'icon-x', title: 'Close', 'aria-label': 'Close' }, '✕');
  const modal = el('div', { class: 'modal' },
    el('div', { class: 'modal-head' }, titleEl, closeBtn), body, foot);
  const scrim = el('div', { class: 'modal-scrim' }, modal);
  document.body.append(scrim);

  // selected: volume indices to process (one or both eyes). bmByIndex: that eye's BM choice — it must be
  // per-eye, because the two eyes can carry different bm_prompts and pipeline._resolve_bm silently falls
  // back to the self-seg if you ask for "device" on a scan that has none.
  const data = { path: '', scans: [], dlAvailable: false, selected: new Set(), bmByIndex: {},
                 xhr: null, busy: false };

  function close() {
    if (data.xhr) { try { data.xhr.abort(); } catch (_) {} data.xhr = null; }
    scrim.remove();
    window.removeEventListener('keydown', onKey);
    openModal = null;
  }
  closeBtn.addEventListener('click', close);
  scrim.addEventListener('mousedown', (e) => { if (e.target === scrim) close(); });
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  window.addEventListener('keydown', onKey);

  function setFoot(...btns) { foot.innerHTML = ''; foot.append(...btns.filter(Boolean)); }
  const btn = (label, cls, onclick, attrs = {}) =>
    el('button', { class: cls, onclick, ...attrs }, label);

  // ---------------- step: stage a dropped file ----------------
  // The browser gives us bytes, never a path, so upload them to a content-addressed file first. No
  // Cancel button: ✕ / Escape / clicking the scrim already close(), which aborts the transfer.
  function stepStaging(file) {
    titleEl.textContent = 'Uploading';
    body.innerHTML = '';
    const fill = el('div', { class: 'up-bar-fill' });
    const pct = el('div', { class: 'up-pct' }, '0%');
    body.append(el('div', { class: 'up-wrap' },
      el('div', { class: 'up-name' }, file.name),
      el('div', { class: 'up-bar' }, fill),
      el('div', { class: 'up-row' }, el('span', { class: 'muted' }, fmtSize(file.size)), pct)));
    setFoot();

    api.uploadStage(file, {
      onStart: (xhr) => { data.xhr = xhr; },
      onProgress: (loaded, total) => {
        const p = total ? Math.round((100 * loaded) / total) : 0;
        fill.style.width = p + '%';
        pct.textContent = p + '%';
      },
    }).then((res) => {
      data.xhr = null;
      data.path = res.path;
      openScans();
    }).catch((e) => {
      data.xhr = null;
      if (e.aborted) return;                                 // the modal is already gone
      stepFile('');
      const err = body.querySelector('.err');
      if (err) { err.textContent = e.message || String(e); err.hidden = false; }
    });
  }

  // ---------------- step: choose file ----------------
  function stepFile(prefill = '') {
    titleEl.textContent = 'Upload E2E';
    const input = el('input', { type: 'text', class: 'path-input', value: prefill,
      placeholder: 'paste an absolute .E2E path…', spellcheck: 'false' });
    const err = el('div', { class: 'err', hidden: true });
    body.innerHTML = '';
    body.append(
      el('p', { class: 'muted' }, 'Browse for a Spectralis .E2E file, or paste its full path on this computer.'),
      input, err);
    const go = () => {
      const p = input.value.trim();
      if (!p) { input.focus(); return; }
      data.path = p; err.hidden = true; openScans(err);
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
    setFoot(btn('Browse…', 'btn', () => stepBrowse(input.value.trim() || data.path)),
            btn('Open', 'primary', go));
    setTimeout(() => input.focus(), 30);
  }

  // ---------------- step: browse the filesystem ----------------
  function stepBrowse(startPath) {
    titleEl.textContent = 'Choose an E2E file';
    body.innerHTML = '';
    // editable address bar: type/paste a folder path and Enter (or Go) jumps there.
    const addr = el('input', { type: 'text', class: 'path-input fs-addr', spellcheck: 'false',
      placeholder: 'type or paste a folder path, then press Enter' });
    const addrRow = el('div', { class: 'fs-addr-row' }, addr, btn('Go', 'btn', () => load(addr.value.trim())));
    const note = el('div', { class: 'fs-note muted', hidden: true });
    const drives = el('div', { class: 'fs-drives' });
    const listEl = el('div', { class: 'fs-list' });
    body.append(addrRow, drives, note, listEl);
    setFoot(btn('Back', 'ghost', () => stepFile(data.path)));
    addr.addEventListener('keydown', (e) => { if (e.key === 'Enter') load(addr.value.trim()); });

    const chip = (label, path, on) =>
      el('button', { class: 'fs-chip' + (on ? ' on' : ''), type: 'button', onclick: () => load(path) }, label);
    const rowDir = (label, path, cls = 'dir') =>
      el('button', { class: 'fs-row ' + cls, type: 'button', onclick: () => load(path) }, label);
    const rowFile = (f) =>
      el('button', { class: 'fs-row file', type: 'button',
        onclick: () => { data.path = f.path; openScans(); } },
        el('span', { class: 'fs-name' }, f.name),
        el('span', { class: 'fs-size muted' }, fmtSize(f.size)));

    async function load(path) {
      listEl.innerHTML = ''; listEl.append(el('div', { class: 'muted' }, 'Loading…'));
      let res;
      try { res = await api.fsList(path); }
      catch (e) { listEl.innerHTML = ''; listEl.append(el('div', { class: 'err' }, e.message || String(e))); return; }
      addr.value = res.path;                                   // reflect the resolved folder
      note.hidden = !res.not_found;
      if (res.not_found) note.textContent = 'That path was not found — showing your home folder instead.';
      drives.innerHTML = '';
      drives.append(chip('Home', res.home, false));
      (res.roots || []).forEach((r) => {
        const on = res.path.toUpperCase().startsWith(r.toUpperCase());
        drives.append(chip(r.replace(/[\\/]+$/, '') || r, r, on));   // "C:\" → "C:" ; "/" → "/"
      });
      listEl.innerHTML = '';
      if (res.parent) listEl.append(rowDir('⬆  ..', res.parent, 'up'));
      (res.dirs || []).forEach((d) => listEl.append(rowDir('📁  ' + d.name, d.path)));
      (res.files || []).forEach((f) => listEl.append(rowFile(f)));
      if (!(res.dirs || []).length && !(res.files || []).length) {
        listEl.append(el('div', { class: 'fs-empty muted' }, 'No sub-folders or .E2E files here.'));
      }
    }
    load(startPath || '');
  }

  // ---------------- step: list scans ----------------
  async function openScans(errSink) {
    body.innerHTML = '';
    body.append(el('div', { class: 'modal-loading' }, el('div', { class: 'spinner' }), el('div', { class: 'muted' }, 'Reading E2E…')));
    setFoot();
    try {
      const res = await api.uploadOpen(data.path);
      data.scans = res.scans || [];
      data.dlAvailable = !!res.dl_available;
      data.selected = new Set();                             // a new file: never inherit the old picks
      data.bmByIndex = {};
      const who = res.patient || {};
      titleEl.textContent = (who.patient_name || who.patient_id) ? `Upload — ${who.patient_name || who.patient_id}` : 'Upload E2E';
      if (!data.scans.length) {
        body.innerHTML = '';
        body.append(el('div', { class: 'muted' }, 'This E2E has no 6×6-measurable scan (a wide macular volume is required to measure a 6×6 mm GA area).'));
        setFoot(btn('Back', 'ghost', () => stepFile(data.path)), btn('Close', 'btn', close));
        return;
      }
      stepScan();
    } catch (e) {
      if (errSink) { /* came from the file step */ }
      const msg = (e.status === 423) ? 'The E2E file is open or locked in another program; close it and try again.'
        : (e.status === 400 ? 'File not found on this machine.' : (e.message || String(e)));
      stepFile(data.path);
      const err = body.querySelector('.err');
      if (err) { err.textContent = msg; err.hidden = false; }
    }
  }

  // ---------------- step: choose scan(s) ----------------
  const selectedScans = () =>
    data.scans.filter((s) => data.selected.has(s.index)).sort((a, b) => a.index - b.index);

  // Default to every eye. If an E2E somehow holds two 6x6 volumes of the SAME eye, take the richer one
  // and leave the other selectable rather than silently processing both.
  function preselectAllEyes() {
    const best = new Map();
    data.scans.forEach((s) => {
      const cur = best.get(s.eye || '?');
      if (!cur || (s.n_bscans || 0) > (cur.n_bscans || 0)) best.set(s.eye || '?', s);
    });
    best.forEach((s) => data.selected.add(s.index));
  }

  function stepScan() {
    titleEl.textContent = 'Choose scans';
    if (!data.selected.size) preselectAllEyes();
    body.innerHTML = '';
    body.append(el('p', { class: 'muted' },
      'Only 6×6-measurable volumes are shown (one per eye). Select one or both eyes — GA area is computed for each.'));
    // role=listbox/option: aria-selected is ignored by assistive tech on a bare <button>. The existing
    // .scan-row[aria-selected="true"] styling is unchanged, and Space/Enter already fire click.
    const listEl = el('div', { class: 'scan-list', role: 'listbox',
      'aria-multiselectable': 'true', 'aria-label': 'scans to process' });
    let contBtn;
    data.scans.forEach((s) => {
      const fov = (s.fov_mm && s.fov_mm.length === 2) ? `${s.fov_mm[0].toFixed(1)}×${s.fov_mm[1].toFixed(1)} mm` : '';
      const chip = s.has_device_bm
        ? el('span', { class: 'dev-chip on' }, '✓ device BM')
        : el('span', { class: 'dev-chip' }, 'DL BM');
      const rowEl = el('button', { class: 'scan-row', type: 'button', role: 'option',
        'aria-selected': data.selected.has(s.index) ? 'true' : 'false',
        onclick: () => {
          if (data.selected.has(s.index)) data.selected.delete(s.index);
          else data.selected.add(s.index);
          rowEl.setAttribute('aria-selected', data.selected.has(s.index) ? 'true' : 'false');
          if (contBtn) contBtn.disabled = data.selected.size === 0;
        } },
        el('span', { class: 'scan-eye' }, s.eye || '—'),
        el('span', { class: 'scan-kind' }, s.kind || ''),
        el('span', { class: 'scan-fov muted' }, fov),
        el('span', { class: 'scan-n muted' }, `${s.n_bscans} B-scans`),
        chip);
      listEl.append(rowEl);
    });
    body.append(listEl);
    contBtn = btn('Continue →', 'primary', () => stepBM());
    contBtn.disabled = data.selected.size === 0;
    setFoot(btn('Back', 'ghost', () => stepFile(data.path)), contBtn);
  }

  // ---------------- step: choose BM (one block per selected eye) ----------------
  function bmBlock(s) {
    const wrap = el('div', { class: 'bm-block' });
    wrap.append(el('p', { class: 'muted' },
      `${s.eye || '—'} · ${(s.fov_mm || []).map((x) => x.toFixed(1)).join('×')} mm · ${s.n_bscans} B-scans`));

    const prompt = s.bm_prompt;       // "choice" | "dl_only" | "auto_only"
    if (prompt === 'choice') {
      data.bmByIndex[s.index] = data.bmByIndex[s.index] || 'dl';
      const cardDL = el('button', { class: 'bm-card', type: 'button',
        onclick: () => { data.bmByIndex[s.index] = 'dl'; refreshBM(); } },
        el('div', { class: 'bm-card-title' }, 'DL BM segmentation', el('span', { class: 'badge' }, 'recommended')),
        el('div', { class: 'bm-card-sub muted' }, 'our trained model'));
      const cardDev = el('button', { class: 'bm-card', type: 'button',
        onclick: () => { data.bmByIndex[s.index] = 'device'; refreshBM(); } },
        el('div', { class: 'bm-card-title' }, 'Device BM'),
        el('div', { class: 'bm-card-sub muted' }, 'as exported by the Spectralis'));
      const refreshBM = () => {
        cardDL.setAttribute('aria-selected', data.bmByIndex[s.index] === 'dl' ? 'true' : 'false');
        cardDev.setAttribute('aria-selected', data.bmByIndex[s.index] === 'device' ? 'true' : 'false');
      };
      refreshBM();
      wrap.append(el('p', {}, 'This scan has a device Bruch’s-membrane. Which should we use?'),
                  el('div', { class: 'bm-grid' }, cardDL, cardDev));
    } else if (prompt === 'dl_only') {
      data.bmByIndex[s.index] = 'dl';
      wrap.append(el('div', { class: 'bm-note' }, 'This scan has no device Bruch’s-membrane. It will be ',
        el('b', {}, 'DL BM-segmented'), ' automatically.'));
    } else {
      data.bmByIndex[s.index] = 'auto';
      wrap.append(el('div', { class: 'bm-note' }, 'This scan has no device Bruch’s-membrane and no DL model is available. It will be ',
        el('b', {}, 'automatically segmented'), ' (classical).'));
    }
    return wrap;
  }

  function stepBM() {
    titleEl.textContent = 'Bruch’s-membrane segmentation';
    body.innerHTML = '';
    selectedScans().forEach((s) => body.append(bmBlock(s)));
    setFoot(btn('Back', 'ghost', () => stepScan()),
            btn('Process →', 'primary', () => process()));
  }

  // ---------------- step: process (once per selected eye) ----------------
  // Sequential, not parallel: each eye is CPU-bound (DL inference or ~5-12 s self-seg) and would only
  // contend. The E2E is decoded once — store.open caches the RawE2E, so the second eye is a cache hit.
  async function process() {
    const chosen = selectedScans();
    data.busy = true;                                        // a drop mid-process must not hijack the modal
    titleEl.textContent = 'Processing';
    body.innerHTML = '';
    const label = el('div', { class: 'muted' }, 'Segmenting layers and detecting GA…');
    const status = el('div', { class: 'proc-status' });
    body.append(el('div', { class: 'modal-loading' }, el('div', { class: 'spinner' }), label, status));
    setFoot();

    const lockMsg = 'The E2E file is open or locked in another program; close it and try again.';
    const done = [], failed = [];
    try {
      for (let i = 0; i < chosen.length; i++) {
        const s = chosen[i];
        const eye = s.eye || 'scan';
        if (chosen.length > 1) label.textContent = `Processing ${eye} — ${i + 1} of ${chosen.length}…`;
        try {
          const res = await api.uploadProcess(data.path, s.index, data.bmByIndex[s.index] || 'auto');
          done.push({ s, res });
          if (res.warning) notify(res.warning, 'warn', 7000);
          if (res.db && !res.db.saved && res.db.warning) notify(res.db.warning, 'warn', 7000);
          if (chosen.length > 1) status.append(el('span', { class: 'proc-ok' }, `${eye} ✓`));
        } catch (e) {
          failed.push({ s, e });
          if (chosen.length > 1) status.append(el('span', { class: 'proc-bad' }, `${eye} ✗`));
        }
      }
    } finally {
      // Keep the compact processed eyes, but release the decoded E2E and ONNX model after the batch.
      try { await api.uploadFinish(data.path); } catch (_) {}
      data.busy = false;
    }

    if (!done.length) {                                      // every eye failed: keep the error panel
      const e = failed[0].e;
      body.innerHTML = '';
      body.append(el('div', { class: 'err' }, (e.status === 423) ? lockMsg : (e.message || String(e))));
      setFoot(btn('Back', 'ghost', () => stepBM()), btn('Close', 'btn', close));
      return;
    }

    const saved = done.filter((d) => d.res.db && d.res.db.saved).map((d) => d.s.eye || 'scan');
    if (saved.length) notify(`Saved ${saved.join(' + ')} to the database ✓`, 'ok');
    if (failed.length) {                                     // partial: the rest are recorded and open
      const e = failed[0].e;
      const why = (e.status === 423) ? 'the file is locked' : (e.message || String(e));
      notify(`${failed.map((f) => f.s.eye || 'scan').join(', ')} could not be processed (${why}).`, 'warn', 7000);
    }
    close();
    openScan(done[0].res.vid, done[0].res.meta);             // land on the first eye that succeeded
  }

  stepFile('');
  return {
    close,
    busy: () => data.busy,
    startWithFile(file) {
      if (data.xhr) { try { data.xhr.abort(); } catch (_) {} data.xhr = null; }   // supersede an upload
      data.selected = new Set();
      data.bmByIndex = {};
      stepStaging(file);
    },
  };
}
