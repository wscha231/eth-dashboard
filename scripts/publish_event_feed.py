"""Publish public forecast data independently of the static website deployment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.copy_event_evidence import copy_evidence
from scripts.deployment_policy import configure, cooldown
from scripts.export_event_segments import export
from scripts.publish_search_assets import publish as publish_assets, render_snapshot
from scripts.publish_v2_assets import publish as publish_v2_assets

FEED_BRANCH='data/event-feed'
SITE_BRANCH='data/daily-forecast'
DYNAMIC=('signals.json','signals_replay.json','signals_segments.csv','outlook.html','archive_retention.json')


def git(*args, cwd=None):
    return subprocess.check_output(['git',*map(str,args)],cwd=cwd,text=True).strip()


def worktree(branch, target, fallback):
    exists=git('ls-remote','--heads','origin','refs/heads/'+branch)
    base=branch if exists else fallback
    git('fetch','origin',f'{base}:refs/remotes/origin/{base}')
    git('worktree','add','--detach',target,'origin/'+base)
    return Path(target)


def commit(target, branch, message):
    changed=git('diff','--cached','--name-only',cwd=target)
    if not changed:return None
    git('commit','-m',message,cwd=target)
    sha=git('rev-parse','HEAD',cwd=target)
    git('push','origin',f'HEAD:refs/heads/{branch}',cwd=target)
    return sha


def publish(root, runner, source=None):
    source=Path(source or Path(__file__).resolve().parents[1]);root=Path(root);runner=Path(runner)
    payload=json.loads((root/'signals.json').read_text())
    if not payload.get('release_id'):raise ValueError('missing public release')
    feed=worktree(FEED_BRANCH,runner/'event-feed',SITE_BRANCH)
    public=feed/'forecast_site/public';public.mkdir(parents=True,exist_ok=True)
    copy_evidence(root,public,retain_previous=True)
    shutil.copyfile(root/'signals.json',public/'signals.json')
    shutil.copyfile(root/'replay.json',public/'signals_replay.json')
    export(json.loads((root/'replay.json').read_text()),public/'signals_segments.csv')
    (public/'outlook.html').write_text(render_snapshot(payload))
    config=feed/'forecast_site/vercel.json';config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps({'git':{'deploymentEnabled':{'**':False}}})+'\n')
    git('add','-f','--',*[f'forecast_site/public/{p}' for p in DYNAMIC],
        'forecast_site/public/archive','forecast_site/vercel.json',cwd=feed)
    feed_sha=commit(feed,FEED_BRANCH,'chore(data): publish immutable ETH feed without a site build')
    site=worktree(SITE_BRANCH,runner/'event-site',SITE_BRANCH)
    publish_assets(site,source=source,stage=True,include_snapshot=False)
    publish_v2_assets(site,source=source,stage=True)
    configure(site,source=source,stage=True)
    # Remove obsolete static copies so routing cannot silently serve old values.
    for name in (*DYNAMIC,'archive'):
        path=site/'forecast_site/public'/name
        if path.is_dir():shutil.rmtree(path)
        elif path.exists():path.unlink()
        if git('ls-files','--',f'forecast_site/public/{name}',cwd=site):
            git('add','-u','--',f'forecast_site/public/{name}',cwd=site)
    changed=bool(git('diff','--cached','--name-only',cwd=site))
    policy=cooldown(source)
    site_sha=None
    if changed and policy:
        print('Static website update deferred by the provider cooldown; feed data preserved.',flush=True)
    else:
        site_sha=commit(site,SITE_BRANCH,'chore(site): update static EtherForecast interface and data routing')
    result={'release_id':payload['release_id'],'feed_commit':feed_sha or git('rev-parse','HEAD',cwd=feed),
            'static_site_commit':site_sha,'static_site_changed':bool(site_sha),
            'external_verification':'required','data_branch':FEED_BRANCH}
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    git('config','user.name','eth-forecast-bot');git('config user.email','eth-forecast-bot@users.noreply.github.com')
    publish('lake/signals',os.environ['RUNNER_TEMP'])
