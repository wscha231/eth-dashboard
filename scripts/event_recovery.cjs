/* Recovery dispatch only; no model restoration, ledger mutation or training. */
const ACTIVE = ['queued', 'in_progress', 'requested', 'waiting', 'pending'];
// Include every writer in the shared daily-forecast group and the model builder.
// GitHub concurrency keeps only one pending job; another dispatch can replace it.
const WORKERS = new Set([
  'event_hourly.yml', 'event_research.yml', 'daily_forecast.yml',
  'forward_research.yml', 'hybrid_daily.yml', 'hybrid_replay.yml',
  'full_history_backtest.yml', 'weekly_live_review.yml', 'site_search.yml',
]);
const COOLDOWN_MS = 20 * 60 * 1000;

module.exports = async function recover({github, context, core, now, clock = () => now ?? new Date()}) {
  async function finish(decision, message) {
    core.setOutput('recovery', decision);
    core.notice(message);
    await core.summary.addHeading('Hourly forecast recovery')
      .addRaw(`Decision: ${decision}\n\n${message}\n\nPublic freshness has not yet passed verification.\n`).write();
    // Dispatch is not proof that publication recovered. Leave the incident visible.
    core.setFailed('Hourly site is delayed or unavailable; ' + message);
    return decision;
  }
  for (const status of ACTIVE) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRunsForRepo,
      {...context.repo, branch: 'main', status, per_page: 100});
    const active = runs.find(run => WORKERS.has(run.path?.split('/').at(-1)));
    if (active) return finish('active_worker', `Recovery deferred: ${active.name} run ${active.id} is ${status}.`);
  }

  const {data} = await github.rest.actions.listWorkflowRuns({
    ...context.repo, workflow_id: 'event_hourly.yml', branch: 'main', event: 'workflow_dispatch', per_page: 50,
  });
  // Read after the API calls: queued work and network time consume the issuance window.
  const dispatchTime = clock();
  // A re-run keeps created_at, so consider the actual attempt start and completion.
  const recent = data.workflow_runs.find(run => run.event === 'workflow_dispatch' &&
    [run.created_at, run.run_started_at, run.updated_at].some(value =>
      Number.isFinite(Date.parse(value)) && dispatchTime.getTime() - Date.parse(value) < COOLDOWN_MS));
  if (recent) return finish('cooldown', `Recovery deferred: manually dispatched hourly run ${recent.id} was active within the last 20 minutes.`);
  // Reserve five minutes for setup and inference before the existing :55 cutoff.
  if (clock().getUTCMinutes() >= 50) return finish('issuance_deadline', 'Recovery deferred: fewer than five minutes remain before the :55 issuance cutoff.');

  await github.rest.actions.createWorkflowDispatch({...context.repo, workflow_id: 'event_hourly.yml', ref: 'main'});
  return finish('dispatched', 'One fresh hourly run requested on main; its external release must still be verified.');
};
