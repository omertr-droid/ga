// home_view.js — the landing screen: a searchable patient DATABASE (not a card grid) + Upload + Export.
// Loads the patient list once, filters client-side by id/name, and navigates to a patient on click.
import { api } from './api.js';
import { el } from './lib.js';
import { routePatient, notify } from './main.js';
import { openUpload } from './upload_view.js';

// Download .xlsx via fetch+blob (not a bare <a download>): a 404 (empty) or 423 (locked) shows a toast
// instead of a silently broken download.
async function exportXlsx() {
  try {
    const r = await fetch(api.dbXlsxUrl());
    if (!r.ok) {
      let d = r.statusText; try { d = (await r.json()).detail || d; } catch (_) {}
      notify(r.status === 404 ? 'Nothing to export yet.' : ('Export failed: ' + d), 'warn');
      return;
    }
    const url = URL.createObjectURL(await r.blob());
    const a = el('a', { href: url, download: 'ga_clinic.xlsx' });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { notify('Export failed: ' + (e.message || e), 'warn'); }
}

export function mountHome(root) {
  let all = [];                                        // every patient (loaded once, filtered client-side)

  const subtitle = el('p', { class: 'muted home-sub' }, 'Loading…');
  const uploadBtn = el('button', { class: 'primary' }, 'Upload E2E');
  const exportBtn = el('button', { class: 'btn', disabled: true }, 'Export Excel');
  uploadBtn.addEventListener('click', () => openUpload());
  exportBtn.addEventListener('click', exportXlsx);

  const searchInput = el('input', { type: 'search', class: 'search-input',
    placeholder: 'Search by patient id or name…', autocomplete: 'off', spellcheck: 'false' });
  const list = el('div', { class: 'patient-list' });

  const section = el('section', { class: 'screen home' },
    el('div', { class: 'home-head' },
      el('div', {}, el('h2', {}, 'Patient database'), subtitle),
      el('div', { class: 'home-actions' }, uploadBtn, exportBtn)),
    el('div', { class: 'search' }, searchInput),
    list);
  root.append(section);

  function row(p) {
    const name = p.patient_name || p.patient_id || 'unknown';
    const visits = `${p.n_visits} visit${p.n_visits === 1 ? '' : 's'}`;
    const eyes = (p.eyes && p.eyes.length) ? ' · ' + p.eyes.join('/') : '';
    const when = p.latest_date ? ' · last ' + p.latest_date : '';
    const r = el('button', { class: 'patient-row', type: 'button',
      onclick: () => routePatient(p.patient_id) },
      el('span', { class: 'pat-name' }, name),
      p.patient_name ? el('span', { class: 'pat-id muted' }, p.patient_id) : null,
      el('span', { class: 'pat-meta muted' }, `${visits}${eyes}${when}`));
    return r;
  }

  function render(items) {
    list.innerHTML = '';
    if (!items.length) {
      list.append(el('div', { class: 'empty muted' },
        all.length ? 'No patients match your search.'
                   : 'No patients yet. Click “Upload E2E” to add the first scan.'));
      return;
    }
    items.forEach((p) => list.append(row(p)));
  }

  function applyFilter() {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) { render(all); return; }
    render(all.filter((p) =>
      (p.patient_id || '').toLowerCase().includes(q) ||
      (p.patient_name || '').toLowerCase().includes(q)));
  }
  let t = null;
  searchInput.addEventListener('input', () => { clearTimeout(t); t = setTimeout(applyFilter, 110); });

  (async () => {
    try {
      const data = await api.db();
      if (data.locked) {
        subtitle.textContent = '';
        list.append(el('div', { class: 'banner' },
          el('span', {}, '⚠ ' + (data.message || 'The patient database is open in another program.')),
          el('button', { class: 'ghost sm', onclick: () => mountHome((root.innerHTML = '', root)) }, 'Retry')));
        return;
      }
      all = data.patients || [];
      const totalVisits = all.reduce((s, p) => s + (p.n_visits || 0), 0);
      subtitle.textContent = all.length
        ? `${all.length} patient${all.length === 1 ? '' : 's'} · ${totalVisits} scan${totalVisits === 1 ? '' : 's'}`
        : 'No patients yet.';
      exportBtn.disabled = !all.length;
      render(all);
    } catch (e) {
      subtitle.textContent = '';
      list.append(el('div', { class: 'err' }, 'Could not load the database: ' + (e.message || e)));
    }
  })();

  return () => { clearTimeout(t); };
}
