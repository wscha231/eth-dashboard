/* Research inputs and resumable history come only from approved repository runs. */
module.exports = async function select({github, context, core}) {
  const repoId = context.payload.repository.id;
  const allowed = new Map([['main', null]]);
  const pr = context.payload.pull_request;
  if (pr?.head?.repo?.id === repoId) allowed.set(pr.head.ref, pr.head.sha);
  if (context.eventName === 'push') {
    const {data: prs} = await github.rest.repos.listPullRequestsAssociatedWithCommit({...context.repo, commit_sha: context.sha});
    for (const item of prs) if (item.merged_at && item.merge_commit_sha === context.sha && item.head.repo?.id === repoId) {
      allowed.set(item.head.ref, item.head.sha);
    }
  }
  for (const [name, key, file] of [['event-research-state', 'source', 'event_research.yml'], ['event-historical-state', 'cache', 'event_historical_study.yml']]) {
    const {data} = await github.rest.actions.listArtifactsForRepo({...context.repo, name, per_page: 100});
    for (const artifact of data.artifacts) {
      const info = artifact.workflow_run;
      if (artifact.expired || info?.head_repository_id !== repoId || !allowed.has(info.head_branch)) continue;
      if (key === 'source' && info.head_branch !== 'main') continue;
      const {data: run} = await github.rest.actions.getWorkflowRun({...context.repo, run_id: info.id});
      if (!run.path.endsWith('/'+file) || run.status !== 'completed') continue;
      if (key === 'source' && (run.conclusion !== 'success' || run.event === 'pull_request')) continue;
      const expected = allowed.get(info.head_branch);
      if (expected && run.head_sha !== expected) {
        const {data: ancestry} = await github.rest.repos.compareCommitsWithBasehead({...context.repo, basehead: run.head_sha+'...'+expected});
        if (!['ahead', 'identical'].includes(ancestry.status)) continue;
      }
      if (!expected && run.event === 'pull_request') continue;
      core.setOutput(key, String(info.id)); break;
    }
  }
};
