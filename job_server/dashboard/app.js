// Job Queue Dashboard — vanilla JS, polls the FastAPI server on same origin.

const API = '';  // same origin
const POLL_QUEUE_MS = 3000;
const POLL_OUTPUT_MS = 2000;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  view: 'queue',          // 'queue' | 'detail'
  detailJobId: null,
  outputOffset: 0,
  outputComplete: false,
  autoRefresh: true,
  queueTimer: null,
  detailTimer: null,
};

// ----- utilities -----

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function statusBadge(status) {
  const cls = `status status-${status}`;
  return `<span class="${cls}">${status}</span>`;
}

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).detail || ''; } catch (_) {}
    throw new Error(`${r.status} ${r.statusText}${detail ? ': ' + detail : ''}`);
  }
  return r.json();
}

// ----- health -----

async function pollHealth() {
  try {
    const h = await api('/health');
    const el = $('#health');
    el.textContent = `${h.status} · pending: ${h.pending_jobs} · running: ${h.running_jobs}`;
    el.className = `health ${h.status}`;
  } catch (e) {
    const el = $('#health');
    el.textContent = `server unreachable: ${e.message}`;
    el.className = 'health unhealthy';
  }
}

// ----- queue + history -----

async function pollQueue() {
  try {
    const queue = await api('/jobs/queue');
    renderRunning(queue.running_job);
    renderPending(queue.pending_jobs);
    $('#pending-count').textContent = queue.total_pending ? `(${queue.total_pending})` : '';
  } catch (e) {
    console.error('queue poll failed', e);
  }
  try {
    const limit = Math.max(1, Math.min(500, parseInt($('#filter-limit').value, 10) || 30));
    const user = $('#filter-user').value.trim();
    const status = $('#filter-status').value;
    const params = new URLSearchParams({ limit });
    if (user) params.set('user', user);
    if (status) params.set('status', status);
    const hist = await api('/jobs/history?' + params.toString());
    renderHistory(hist);
  } catch (e) {
    console.error('history poll failed', e);
  }
}

function renderRunning(job) {
  const el = $('#running-job');
  if (!job) {
    el.textContent = 'none';
    el.classList.remove('has-job');
    return;
  }
  el.classList.add('has-job');
  el.innerHTML = `
    <a href="#/job/${job.job_id}" class="job-id">${job.job_id}</a>
    &nbsp;·&nbsp; <strong>${escapeHtml(job.experiment_class)}</strong>
    &nbsp;·&nbsp; user: ${escapeHtml(job.user)}
    &nbsp;·&nbsp; started: ${fmtTime(job.started_at)}
  `;
}

function renderPending(jobs) {
  const tbody = $('#pending-table tbody');
  if (!jobs || jobs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty">queue empty</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => `
    <tr data-job="${j.job_id}">
      <td><a href="#/job/${j.job_id}">${j.job_id}</a></td>
      <td>${escapeHtml(j.user)}</td>
      <td>${escapeHtml(j.experiment_class)}</td>
      <td>${j.priority}</td>
      <td>${fmtTime(j.created_at)}</td>
      <td><button class="danger" data-cancel="${j.job_id}">cancel</button></td>
    </tr>
  `).join('');
}

function renderHistory(jobs) {
  const tbody = $('#history-table tbody');
  if (!jobs || jobs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty">no jobs</td></tr>`;
    return;
  }
  tbody.innerHTML = jobs.map(j => `
    <tr data-job="${j.job_id}">
      <td><a href="#/job/${j.job_id}">${j.job_id}</a></td>
      <td>${escapeHtml(j.user)}</td>
      <td>${escapeHtml(j.experiment_class || '')}</td>
      <td>${statusBadge(j.status)}</td>
      <td>${fmtTime(j.created_at)}</td>
      <td>${fmtTime(j.completed_at)}</td>
    </tr>
  `).join('');
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function cancelJob(jobId) {
  if (!confirm(`Cancel pending job ${jobId}?`)) return;
  try {
    await api(`/jobs/${jobId}`, { method: 'DELETE' });
    await pollQueue();
  } catch (e) {
    alert(`Cancel failed: ${e.message}`);
  }
}

// ----- detail view -----

async function loadDetail(jobId) {
  state.detailJobId = jobId;
  state.outputOffset = 0;
  state.outputComplete = false;
  $('#detail-job-id').textContent = jobId;
  $('#detail-output').textContent = '(waiting for output…)';
  $('#detail-meta').innerHTML = 'loading…';
  $('#detail-cancel').hidden = true;
  await pollDetail();
}

async function pollDetail() {
  if (!state.detailJobId) return;
  const jobId = state.detailJobId;

  try {
    const job = await api(`/jobs/${jobId}`);
    renderDetailMeta(job);
    $('#detail-cancel').hidden = (job.status !== 'pending');
  } catch (e) {
    $('#detail-meta').innerHTML = `<dt>error</dt><dd>${escapeHtml(e.message)}</dd>`;
  }

  if (state.outputComplete) return;

  try {
    const out = await api(`/jobs/${jobId}/output?offset=${state.outputOffset}`);
    if (out.output) {
      const pre = $('#detail-output');
      if (state.outputOffset === 0) pre.textContent = '';
      const wasAtBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
      pre.append(document.createTextNode(out.output));
      if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
    }
    if (out.line_count > state.outputOffset) {
      state.outputOffset = out.line_count;
    }
    if (out.is_complete) {
      state.outputComplete = true;
      const pre = $('#detail-output');
      if (!pre.textContent) pre.textContent = '(no output captured)';
    }
  } catch (e) {
    console.error('output poll failed', e);
  }
}

function renderDetailMeta(job) {
  const rows = [
    ['status', statusBadge(job.status)],
    ['user', escapeHtml(job.user)],
    ['experiment', escapeHtml(job.experiment_class)],
    ['priority', job.priority],
    ['created', fmtTime(job.created_at)],
    ['started', fmtTime(job.started_at) || '—'],
    ['completed', fmtTime(job.completed_at) || '—'],
    ['data file', job.data_file_path ? escapeHtml(job.data_file_path) : '—'],
    ['expt pickle', job.expt_pickle_path ? escapeHtml(job.expt_pickle_path) : '—'],
    ['hardware cfg', job.hardware_config_version_id || '—'],
    ['multiphoton cfg', job.multiphoton_config_version_id || '—'],
    ['floquet cfg', job.floquet_storage_version_id || '—'],
    ['man1 cfg', job.man1_storage_version_id || '—'],
  ];
  if (job.error_message) rows.push(['error', escapeHtml(job.error_message)]);
  $('#detail-meta').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ----- routing -----

function route() {
  const hash = window.location.hash || '#/';
  const m = hash.match(/^#\/job\/(.+)$/);
  if (m) {
    showView('detail');
    loadDetail(m[1]);
  } else {
    showView('queue');
    state.detailJobId = null;
  }
}

function showView(view) {
  state.view = view;
  $('#view-queue').hidden = (view !== 'queue');
  $('#view-detail').hidden = (view !== 'detail');
}

// ----- timers -----

function startTimers() {
  stopTimers();
  if (!state.autoRefresh) return;
  state.queueTimer = setInterval(() => {
    pollHealth();
    if (state.view === 'queue') pollQueue();
  }, POLL_QUEUE_MS);
  state.detailTimer = setInterval(() => {
    if (state.view === 'detail') pollDetail();
  }, POLL_OUTPUT_MS);
}

function stopTimers() {
  if (state.queueTimer) clearInterval(state.queueTimer);
  if (state.detailTimer) clearInterval(state.detailTimer);
  state.queueTimer = state.detailTimer = null;
}

// ----- init -----

window.addEventListener('hashchange', route);
window.addEventListener('DOMContentLoaded', () => {
  $('#autorefresh').addEventListener('change', (e) => {
    state.autoRefresh = e.target.checked;
    if (state.autoRefresh) startTimers(); else stopTimers();
  });
  $('#refresh-btn').addEventListener('click', () => {
    pollHealth();
    if (state.view === 'queue') pollQueue();
    if (state.view === 'detail') pollDetail();
  });
  // Re-poll history when filters change
  ['#filter-user', '#filter-status', '#filter-limit'].forEach(sel => {
    $(sel).addEventListener('change', () => { if (state.view === 'queue') pollQueue(); });
    $(sel).addEventListener('input',  () => { if (state.view === 'queue') pollQueue(); });
  });
  // Delegate cancel-button clicks
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-cancel]');
    if (btn) {
      e.preventDefault();
      cancelJob(btn.getAttribute('data-cancel'));
    }
  });
  // Detail cancel button
  $('#detail-cancel').addEventListener('click', () => {
    if (state.detailJobId) cancelJob(state.detailJobId);
  });

  route();
  pollHealth();
  if (state.view === 'queue') pollQueue();
  startTimers();
});
