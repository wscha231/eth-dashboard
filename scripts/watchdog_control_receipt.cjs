/* Read-only, bounded telemetry after the existing watchdog decision. */
const fs = require('node:fs');
const REPOSITORY = 'wscha231/eth-dashboard';
const INFERENCE = 'Collect closed bars, settle matured outcomes and infer from saved models';
const LIMITATIONS = [
  'Absence of a producer execution is not proof that a scheduled trigger was never accepted or that GitHub scheduler caused the miss.',
  'An absent watchdog execution at an expected :18/:38 opportunity cannot produce a repository-local receipt.',
  'A recovery dispatch request is not proof that publication recovered; no dispatched producer run ID is inferred.',
  'Active-worker evidence records the recovery guard, not a proven shared-lock cause.',
  'Producer observation is bounded to 50 recent runs and 10 exact-attempt job queries; older attempts and unobserved executions remain unknown.',
  'Public status is a separate post-decision observation, not the payload of the unchanged health check.',
];
function time(value) {
  if (typeof value !== 'string' || !/Z$|[+-]\d\d:\d\d$/.test(value) || !Number.isFinite(Date.parse(value))) throw Error('Invalid timestamp');
  return Date.parse(value);
}
function integer(value) {
  if (!Number.isSafeInteger(Number(value)) || Number(value) < 1) throw Error('Invalid identity');
  return Number(value);
}
function summary(run) {
  return Object.fromEntries(['id', 'run_attempt', 'event', 'created_at', 'run_started_at', 'updated_at', 'status', 'conclusion', 'head_sha', 'path']
    .map(k => [k === 'id' ? 'run_id' : k, run[k] ?? null]));
}
function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, stable(value[k])]));
  return value;
}
function build({run, job, env, checkedAt, runs = [], publicData = null, recovery = null, limitations = []}) {
  const id = integer(env.GITHUB_RUN_ID), attempt = integer(env.GITHUB_RUN_ATTEMPT);
  if (env.GITHUB_REPOSITORY !== REPOSITORY || run.repository?.full_name !== REPOSITORY
      || run.head_repository?.full_name !== REPOSITORY || run.head_branch !== 'main'
      || run.path !== '.github/workflows/event_watchdog.yml' || run.id !== id || run.run_attempt !== attempt
      || run.event !== env.GITHUB_EVENT_NAME || run.head_sha !== env.GITHUB_SHA
      || !/^[a-f0-9]{40}$/.test(run.head_sha) || job.run_id !== id || job.run_attempt !== attempt || job.name !== 'freshness') {
    throw Error('Watchdog run/attempt identity mismatch');
  }
  const checked = time(checkedAt), started = time(job.started_at), created = time(run.created_at);
  if (created > started || started > checked) throw Error('Watchdog timestamps out of order');
  const health = job.steps.filter(s => s.name === 'Independently check public timestamps and delayed status');
  // Use the real health-check start, not the producer creation time or artifact time.
  const healthAt = health.length === 1 && health[0].started_at ? time(health[0].started_at) : started;
  if (healthAt < started || healthAt > checked) throw Error('Invalid health check time');
  const origin = Math.floor(healthAt / 3600000) * 3600000;
  const extra = [...limitations];
  if (new Date(healthAt).getUTCMinutes() < 15) extra.push('Public readiness grace may still accept the preceding slot; this receipt keys the actual check hour.');
  if (health.length !== 1 || !health[0].started_at) extra.push('Health check start unavailable; slot uses watchdog job start.');
  const outcome = env.HEALTH_OUTCOME || 'unavailable';
  if (!['success', 'failure', 'cancelled', 'skipped', 'unavailable'].includes(outcome)) throw Error('Invalid health outcome');
  const decision = outcome === 'success' ? 'healthy' : recovery?.decision ?? null;
  if (decision !== null && !['healthy', 'invalid_hosting_policy', 'hosting_cooldown', 'active_worker', 'cooldown', 'issuance_deadline', 'dispatched'].includes(decision)) throw Error('Invalid recovery decision');
  if (outcome === 'success' && recovery) throw Error('Healthy recovery evidence conflict');
  if (!decision) extra.push('Recovery decision unavailable; no dispatch outcome is inferred.');
  if (recovery && (typeof recovery.dispatch_requested !== 'boolean'
      || (decision === 'dispatched') !== recovery.dispatch_requested)) throw Error('Invalid dispatch evidence');
  if (decision === 'dispatched') time(recovery.dispatch_requested_at);
  for (const [key, expected] of [['blocking_worker', 'active_worker'], ['cooldown_run', 'cooldown']]) {
    if (decision === expected && !recovery?.[key]) throw Error('Missing decision evidence');
    if (recovery?.[key]) integer(recovery[key].id);
  }
  const matching = [];
  for (const candidate of runs) {
    const r = candidate.run;
    if (r.repository?.full_name !== REPOSITORY || r.head_repository?.full_name !== REPOSITORY
        || r.head_branch !== 'main' || r.path !== '.github/workflows/event_hourly.yml') continue;
    integer(r.id); integer(r.run_attempt);
    for (const j of candidate.jobs) {
      if (j.run_id !== r.id || j.run_attempt !== r.run_attempt || j.name !== 'forecast') continue;
      const steps = j.steps.filter(s => s.name === INFERENCE && s.started_at && (s.status === 'in_progress' || (s.status === 'completed' && ['success', 'failure'].includes(s.conclusion))));
      if (steps.some(s => { const t = time(s.started_at); return origin <= t && t < origin + 3600000 && t <= checked; })) {
        matching.push({...summary(r), match_basis: 'observed_inference_step_start_in_slot'});
        break;
      }
    }
  }
  matching.sort((a, b) => a.run_id - b.run_id || a.run_attempt - b.run_attempt);
  return stable({schema: 1, task_key: 'EF-OPS-02-SLOT-KEYED-CONTROL-TELEMETRY', repository: REPOSITORY,
    watchdog: {run_id: id, run_attempt: attempt, event: run.event, head_sha: run.head_sha,
      created_at: run.created_at, started_at: job.started_at, checked_at: checkedAt},
    slot: {expected_origin: new Date(origin).toISOString(), expected_primary_trigger_at: new Date(origin + 480000).toISOString(),
      issuance_deadline: new Date(origin + 3300000).toISOString(), basis: 'actual_health_check_hour', health_checked_at: new Date(healthAt).toISOString()},
    health: {outcome, public_status: typeof publicData?.status === 'string' ? publicData.status.slice(0, 128) : null,
      observed_release_id: typeof publicData?.release_id === 'string' ? publicData.release_id.slice(0, 256) : null},
    producer_observation: {observed: matching.length > 0, matching_runs: matching},
    recovery: {decision, blocking_worker: recovery?.blocking_worker ? summary(recovery.blocking_worker) : null,
      cooldown_run: recovery?.cooldown_run ? summary(recovery.cooldown_run) : null,
      dispatch_requested: outcome === 'success' ? false : recovery?.dispatch_requested ?? null,
      dispatch_requested_at: recovery?.dispatch_requested_at ?? null},
    limitations: [...LIMITATIONS, ...extra]});
}
async function collect({github, context, env, fetchPublic = async () => {
  const response = await fetch('https://etherforecast.live/signals.json', {signal: AbortSignal.timeout(5000)});
  if (!response.ok) throw Error('Public observation unavailable');
  const reader = response.body.getReader(); const chunks = []; let bytes = 0;
  try {
    for (;;) { const {done, value} = await reader.read(); if (done) break; bytes += value.length;
      if (bytes > 2 * 1024 * 1024) throw Error('Public observation exceeds limit'); chunks.push(value); }
  } finally { await reader.cancel(); }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}}) {
  const id = integer(env.GITHUB_RUN_ID), attempt = integer(env.GITHUB_RUN_ATTEMPT);
  const {data: run} = await github.rest.actions.getWorkflowRunAttempt({...context.repo, run_id: id, attempt_number: attempt, request: {timeout: 5000}});
  const {data: jobs} = await github.rest.actions.listJobsForWorkflowRunAttempt({...context.repo, run_id: id, attempt_number: attempt, per_page: 100, request: {timeout: 5000}});
  const own = jobs.jobs.filter(j => j.name === 'freshness');
  if (own.length !== 1) throw Error('Exact watchdog job required');
  const limitations = [], runs = []; let publicData = null;
  try { publicData = await fetchPublic(); } catch { limitations.push('Post-decision public observation unavailable.'); }
  try {
    const {data} = await github.rest.actions.listWorkflowRuns({...context.repo, workflow_id: 'event_hourly.yml', branch: 'main', per_page: 50, request: {timeout: 5000}});
    if (data.total_count > data.workflow_runs.length) limitations.push('Producer list truncated; absence is not exhaustive.');
    for (const r of data.workflow_runs.slice(0, 10)) {
      const {data: detail} = await github.rest.actions.listJobsForWorkflowRunAttempt({...context.repo, run_id: r.id, attempt_number: r.run_attempt, per_page: 100, request: {timeout: 5000}});
      runs.push({run: r, jobs: detail.jobs});
    }
  } catch { limitations.push('Producer metadata lookup incomplete; absence is not exhaustive.'); }
  const receipt = build({run, job: own[0], env, checkedAt: new Date().toISOString(), runs, publicData,
    recovery: env.RECOVERY_EVIDENCE ? JSON.parse(env.RECOVERY_EVIDENCE) : null, limitations});
  const bytes = JSON.stringify(receipt) + '\n';
  if (Buffer.byteLength(bytes) > 65536) throw Error('Receipt exceeds compact size limit');
  fs.mkdirSync('watchdog-control', {recursive: true}); fs.writeFileSync('watchdog-control/receipt.json', bytes);
  return receipt;
}
module.exports = {build, collect, stable};
