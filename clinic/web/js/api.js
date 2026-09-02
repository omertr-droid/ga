// api.js — the only place that knows endpoint URLs. JSON endpoints return parsed objects;
// PNG endpoints return URL strings (for <img>/Image()). No PLEX endpoints (OCT-only).
const J = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    const e = new Error(detail); e.status = r.status; throw e;
  }
  return r.json();
};
const POST = (url, body) => J(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});
const vp = (vid) => encodeURIComponent(vid);

export const api = {
  health: () => J('/api/health'),
  config: () => J('/api/config'),

  // upload flow
  fsList: (path) => J('/api/fs/list?path=' + encodeURIComponent(path || '')),
  // A dropped file is raw bytes with no path, so stage it server-side first and use the path it returns.
  // XHR, not fetch: only XHR reports upload progress, and an E2E is ~300 MB. Passing the File as the
  // body streams it — it is never read into a JS string. onStart hands back the xhr so the caller can abort.
  uploadStage: (file, { onProgress, onStart } = {}) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload/stage?name=' + encodeURIComponent(file.name || ''));
    xhr.responseType = 'json';
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    });
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) { resolve(xhr.response); return; }
      const detail = (xhr.response && xhr.response.detail) || xhr.statusText || 'upload failed';
      const e = new Error(detail); e.status = xhr.status; reject(e);
    });
    xhr.addEventListener('error', () => reject(new Error('Network error while uploading.')));
    xhr.addEventListener('abort', () => { const e = new Error('upload cancelled'); e.aborted = true; reject(e); });
    if (onStart) onStart(xhr);
    xhr.send(file);
  }),
  uploadOpen: (path) => POST('/api/upload/open', { path }),
  uploadProcess: (path, index, bmChoice) => POST('/api/upload/process', { path, index, bm_choice: bmChoice }),
  uploadFinish: (path) => POST('/api/upload/finish', { path }),
  reopen: (recordId) => POST('/api/reopen', { record_id: recordId }),

  // patient database
  db: () => J('/api/db'),
  patient: (pid) => J(`/api/db/${encodeURIComponent(pid)}`),
  dbDelete: (id, deleteStaged = false) => J(
    `/api/db?record_id=${encodeURIComponent(id)}&delete_staged=${deleteStaged ? 'true' : 'false'}`,
    { method: 'DELETE' }),
  dbXlsxUrl: () => '/api/db.xlsx',
  dbCsvUrl: () => '/api/db.csv',

  // viewer panels (per vid)
  meta: (vid) => J(`/api/scan/${vp(vid)}/meta`),
  locLines: (vid) => J(`/api/scan/${vp(vid)}/loc_lines`),
  bm: (vid) => J(`/api/scan/${vp(vid)}/bm`),
  gaNative: (vid) => J(`/api/scan/${vp(vid)}/ga_native`),
  dashboard: (vid) => J(`/api/scan/${vp(vid)}/dashboard`),
  bscanUrl: (vid, idx) => `/api/scan/${vp(vid)}/bscan/${idx}.png`,
  localizerUrl: (vid) => `/api/scan/${vp(vid)}/localizer.png`,
  projectionUrl: (vid) => `/api/scan/${vp(vid)}/projection.png`,
  gaOverlayUrl: (vid) => `/api/scan/${vp(vid)}/ga_overlay.png`,
};
