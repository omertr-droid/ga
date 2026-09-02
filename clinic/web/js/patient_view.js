// patient_view.js — one patient's visits: date · eye · GA area · Δ vs previous (same eye) · BM source.
// "Open" re-opens the scan into the Viewer (re-processed from its stored E2E + index + BM choice).
import { api } from './api.js';
import { el, fmtArea } from './lib.js';
import { route, routePatient, openScan, showLoading, hideLoading, notify } from './main.js';

const dateOf = (r) => (r.acq_date || '').split(' ')[0] || (r.logged_at || '').split('T')[0] || '—';
const sortKey = (r) => r.acq_iso || r.record_id || '';
const numGA = (r) => (r.ga_area_mm2 === '' || r.ga_area_mm2 == null) ? null : Number(r.ga_area_mm2);

function deltaCell(curr, prev) {
  if (prev == null || curr == null || isNaN(curr) || isNaN(prev)) return el('span', { class: 'muted' }, '—');
  const d = curr - prev;
  if (Math.abs(d) < 0.005) return el('span', { class: 'delta flat' }, '±0.00');
  const up = d > 0;                                  // GA growth (up) is the clinically adverse direction
  return el('span', { class: 'delta ' + (up ? 'up' : 'down') },
    (up ? '▲ +' : '▼ −') + Math.abs(d).toFixed(2));
}

export function mountPatient(root, patientId) {
  const title = el('h2', {}, 'Patient');
  const sub = el('span', { class: 'muted' }, '');
  const tableWrap = el('div', {}, el('div', { class: 'muted' }, 'Loading…'));

  root.append(el('section', { class: 'screen patient' },
    el('div', { class: 'patient-head' },
      el('button', { class: 'ghost back', onclick: () => route('home') }, '‹ Database'),
      el('div', { class: 'patient-title' }, title, sub)),
    tableWrap));

  function openRow(rec) {
    if (!rec.record_id) { alert('This row cannot be opened (no record id).'); return; }
    showLoading('Re-opening the scan (segmenting layers and detecting GA)…');
    api.reopen(rec.record_id)
      .then((res) => openScan(res.vid, res.meta))
      .catch((e) => alert('Could not open: ' + (e.message || e)))
      .finally(hideLoading);
  }

  async function removeRow(rec) {
    const fileNote = rec.staged_copy
      ? 'GA Clinic has a managed drag-and-drop copy; you can choose whether to delete it next.'
      : 'The original E2E file on disk is NOT deleted.';
    if (!confirm(`Remove this scan from the database?\n\n${rec.eye} · ${dateOf(rec)}\n\n${fileNote}`)) return;
    let deleteStaged = false;
    if (rec.staged_copy) {
      deleteStaged = confirm('This scan came from a drag-and-drop copy stored by GA Clinic.\n\n' +
        'Also delete that stored E2E copy when no other eye/record uses it?');
    }
    try {
      const res = await api.dbDelete(rec.record_id, deleteStaged);
      if (res.warning) notify(res.warning, 'warn', 7000);
      else if (res.staged_deleted) notify('Removed from the database and deleted the stored E2E copy', 'ok');
      else notify('Removed from the database', 'ok');
      load();
    } catch (e) { alert('Could not remove: ' + (e.message || e)); }
  }

  function buildTable(visits) {
    // group by eye, sort each eye chronologically, compute Δ vs the previous scan of the same eye
    const byEye = {};
    for (const r of visits) (byEye[r.eye || '—'] ||= []).push(r);
    const rows = [];
    Object.keys(byEye).sort().forEach((eye) => {
      const list = byEye[eye].slice().sort((a, b) => (sortKey(a) < sortKey(b) ? -1 : 1));
      let prev = null;
      list.forEach((r) => {
        const ga = numGA(r);
        rows.push(el('tr', {},
          el('td', {}, dateOf(r)),
          el('td', {}, el('span', { class: 'eye-tag' }, r.eye || '—')),
          el('td', { class: 'num' }, fmtArea(ga)),
          el('td', { class: 'num' }, deltaCell(ga, prev)),
          el('td', { class: 'bm muted' }, r.bm_source || '—'),
          el('td', { class: 'acts' },
            el('button', { class: 'primary sm', onclick: () => openRow(r) }, 'Open'),
            el('button', { class: 'ghost sm danger', onclick: () => removeRow(r) }, 'Remove'))));
        prev = ga;
      });
    });
    return el('table', { class: 'visits-table' },
      el('thead', {}, el('tr', {},
        el('th', {}, 'Date'), el('th', {}, 'Eye'),
        el('th', { class: 'num' }, 'GA area'), el('th', { class: 'num' }, 'Δ vs prev'),
        el('th', {}, 'BM'), el('th', {}, ''))),
      el('tbody', {}, ...rows));
  }

  async function load() {
    tableWrap.innerHTML = ''; tableWrap.append(el('div', { class: 'muted' }, 'Loading…'));
    try {
      const p = await api.patient(patientId);
      const name = p.patient_name || p.patient_id || 'unknown';
      title.textContent = name;
      sub.textContent = p.patient_name ? `  ${p.patient_id}` : '';
      tableWrap.innerHTML = '';
      tableWrap.append(buildTable(p.visits || []));
    } catch (e) {
      tableWrap.innerHTML = '';
      if (e.status === 404) {
        tableWrap.append(el('div', { class: 'muted' }, 'This patient has no scans. ',
          el('a', { href: '#', onclick: (ev) => { ev.preventDefault(); route('home'); } }, 'Back to the database.')));
      } else {
        tableWrap.append(el('div', { class: 'err' }, 'Could not load: ' + (e.message || e)));
      }
    }
  }
  load();

  return () => {};
}
