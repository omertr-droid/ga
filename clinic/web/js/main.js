// main.js — the 1 menu + screen router. Screens: Database (home), Patient detail, 3-panel Viewer.
// "Upload E2E" is a button that opens a modal (it is not a screen).
import { api } from './api.js';
import { mountHome } from './home_view.js';
import { mountPatient } from './patient_view.js';
import { mountViewer } from './viewer_view.js';
import { openUpload, openUploadWithFile } from './upload_view.js';

const $ = (s) => document.querySelector(s);
let cleanup = null;
let last = null;                       // { vid, meta } of the currently/last-loaded scan

// The identity belongs to the Viewer: it names the scan on screen. route() clears it everywhere else,
// so the Database and Patient screens never claim a scan is loaded when none is being viewed.
function setIdentity(meta) {
  const lab = $('#identity');
  if (!meta) { lab.className = 'identity muted'; lab.textContent = 'no scan loaded'; return; }
  lab.className = 'identity';
  const date = (meta.acq_date || '').split(' ')[0];      // '03-Apr-2022 09:14:02' -> '03-Apr-2022'
  const fov = meta.fov_mm ? `${meta.fov_mm[0].toFixed(1)}×${meta.fov_mm[1].toFixed(1)} mm` : '';
  lab.textContent = [meta.identity || meta.patient_id || '', date, fov].filter(Boolean).join('  ·  ');
}

function setActive(name) {
  document.querySelectorAll('#menu .menu-item[data-screen]').forEach(
    (t) => t.classList.toggle('active', t.dataset.screen === name));
}

export function route(name, arg) {
  if (name === 'viewer' && !arg && !last) name = 'home';
  if (cleanup) { cleanup(); cleanup = null; }
  const root = $('#screen-root');
  root.innerHTML = '';
  if (name === 'viewer') {
    setActive('viewer');
    const a = arg || last;
    setIdentity(a.meta);
    cleanup = mountViewer(root, a.vid, a.meta);
  } else if (name === 'patient') {
    setActive('home');                 // patient detail belongs to the Database section
    setIdentity(null);
    cleanup = mountPatient(root, arg);
  } else {
    setActive('home');
    setIdentity(null);
    cleanup = mountHome(root);
  }
}

// Navigate to a patient's detail screen (called from the Home list).
export function routePatient(patientId) { route('patient', patientId); }

// Called when a scan is chosen/processed (upload modal or a patient-detail Open).
export function openScan(vid, meta) {
  last = { vid, meta };
  const vbtn = document.querySelector('#menu .menu-item[data-screen="viewer"]');
  if (vbtn) vbtn.disabled = false;
  route('viewer', last);               // route() sets the identity
}

export function showLoading(msg) {
  $('#loading-label').textContent = msg || 'Working…';
  $('#loading').hidden = false;
}
export function hideLoading() { $('#loading').hidden = true; }

// Transient corner toast (non-blocking feedback, e.g. "Saved ✓" / a soft database warning).
let toastTimer = null;
export function notify(msg, kind = 'ok', ms = 4200) {
  const host = $('#toast');
  if (!host) return;
  host.className = 'toast ' + kind;
  host.textContent = msg;
  host.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { host.hidden = true; }, ms);
}

// ---------- drag & drop an .E2E anywhere in the window ----------
// Window-scoped for correctness, not just reach: without preventDefault on BOTH dragover and drop the
// browser navigates away and opens the dropped file. Every handler is gated on the drag actually
// carrying files, so dragging text into a path input keeps its native behavior.
const hasFiles = (e) => {
  const t = e.dataTransfer && e.dataTransfer.types;
  return !!t && Array.prototype.indexOf.call(t, 'Files') !== -1;
};

let dragDepth = 0;                     // dragleave also fires when the cursor crosses onto a CHILD
function showDrop() {
  const o = $('#drop-overlay');
  if (o) o.hidden = false;
  const live = $('#drop-live');
  if (live) live.textContent = 'Release to add an E2E scan';
}
function hideDrop() {
  const o = $('#drop-overlay');
  if (o) o.hidden = true;
  const live = $('#drop-live');
  if (live) live.textContent = '';
}

// Reject folders, multi-drops and non-E2E files before a single byte leaves the browser.
function pickDroppedE2E(dt) {
  const items = dt.items ? Array.from(dt.items) : [];
  const entries = items
    .filter((it) => it.kind === 'file' && typeof it.webkitGetAsEntry === 'function')
    .map((it) => it.webkitGetAsEntry());
  // A dropped directory otherwise arrives as a 0-byte File and fails confusingly downstream.
  if (entries.some((en) => en && en.isDirectory)) {
    notify('That’s a folder — drop a single .E2E file.', 'warn'); return null;
  }
  const files = dt.files ? Array.from(dt.files) : [];
  if (!files.length) { notify('No file detected in that drop.', 'warn'); return null; }
  if (files.length > 1) { notify('Drop a single .E2E file at a time.', 'warn'); return null; }
  const f = files[0];
  if (!/\.e2e$/i.test(f.name || '')) { notify('Only Spectralis .E2E files are supported.', 'warn'); return null; }
  if (!f.size) { notify('That file is empty.', 'warn'); return null; }
  return f;
}

window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  if (++dragDepth === 1) showDrop();
});
window.addEventListener('dragover', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();                  // required, or the drop never fires
  e.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', (e) => {
  if (!hasFiles(e)) return;
  if (--dragDepth <= 0) { dragDepth = 0; hideDrop(); }
});
window.addEventListener('dragend', () => { dragDepth = 0; hideDrop(); });   // safety: never stick open
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();                  // stop the browser opening the file
  dragDepth = 0;                       // drop emits no matching dragleave
  hideDrop();
  const f = pickDroppedE2E(e.dataTransfer);
  if (!f) return;
  if (!openUploadWithFile(f)) { notify('A scan is still processing — wait for it to finish.', 'warn'); return; }
  const live = $('#drop-live');
  if (live) live.textContent = `Reading ${f.name}…`;
});

document.querySelectorAll('#menu .menu-item[data-screen]').forEach((t) =>
  t.addEventListener('click', () => { if (!t.disabled) route(t.dataset.screen); }));
$('#menu-upload').addEventListener('click', () => openUpload());

api.health().catch(() => {});
route('home');
