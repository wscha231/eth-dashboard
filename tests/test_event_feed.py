"""Exercise two actual git publications: data changes must not rebuild the site."""
import json
from pathlib import Path
import subprocess
import pytest
from scripts.publish_event_feed import publish
from signal_pipeline.diagnostics import export_archive


@pytest.mark.parametrize('blocked', [False, True])
def test_two_publications_update_feed_but_only_one_static_deployment(tmp_path,monkeypatch,blocked):
    source=Path(__file__).resolve().parents[1]
    remote=tmp_path/'remote.git';working=tmp_path/'working';working.mkdir()
    def git(*args):return subprocess.check_output(['git',*map(str,args)],text=True).strip()
    git('init','--bare','-q',remote);git('init','-q',working)
    git('-C',working,'config','user.name','test');git('-C',working,'config','user.email','test@example.invalid')
    git('-C',working,'remote','add','origin',remote)
    pub=working/'forecast_site/public';pub.mkdir(parents=True)
    (pub/'signals.json').write_text('{"old":true}');(pub/'donations.json').write_text('{"preserved":true}')
    git('-C',working,'add','.');git('-C',working,'commit','-qm','existing public state')
    git('-C',working,'push','origin','HEAD:refs/heads/data/daily-forecast')
    root=working/'lake/signals';root.mkdir(parents=True)
    ref=export_archive(root,{'6':[]},kind='range_shadow',as_of='2026-09-14T07:00Z')
    payload={'release_id':'one','evidence_archives':{'range_shadow':ref}}
    (root/'signals.json').write_text(json.dumps(payload));(root/'replay.json').write_text('{"generated_at":"fixture","horizons":{}}')
    monkeypatch.chdir(working)
    if blocked:
        monkeypatch.setattr('scripts.publish_event_feed.cooldown', lambda _: {'blocked': True})
    original_static=git('--git-dir',remote,'rev-parse','refs/heads/data/daily-forecast')
    for run in (1,2):
        runner=tmp_path/f'runner{run}';runner.mkdir()
        payload['release_id']=str(run);(root/'signals.json').write_text(json.dumps(payload))
        result=publish(root,runner,source)
        if blocked:
            assert result['static_site_commit'] is None
            static=original_static
        elif run==1:
            static=result['static_site_commit'];assert static
        else:
            assert result['static_site_commit'] is None
        assert git('--git-dir',remote,'rev-parse','refs/heads/data/daily-forecast')==static
        current=json.loads(git('--git-dir',remote,'show','refs/heads/data/event-feed:forecast_site/public/signals.json'))
        assert current['release_id']==str(run)
        assert git('--git-dir',remote,'show','refs/heads/data/daily-forecast:forecast_site/public/donations.json')=='{"preserved":true}'
        config=json.loads(git('--git-dir',remote,'show','refs/heads/data/event-feed:forecast_site/vercel.json'))
        assert config['git']['deploymentEnabled']=={'**':False}
    files=git('--git-dir',remote,'ls-tree','-r','--name-only','refs/heads/data/event-feed').splitlines()
    assert not any(p.endswith('.db') or 'range_seed' in p for p in files)
    assert ('forecast_site/public/signals.json' in git('--git-dir',remote,'ls-tree','-r','--name-only','refs/heads/data/daily-forecast').splitlines()) == blocked


def test_routes_use_only_the_public_feed_and_bound_current_cache():
    config=json.loads(Path('forecast_site/vercel.json').read_text())
    routes={r['source']:r['destination'] for r in config['rewrites']}
    for name in ('signals.json','signals_replay.json','signals_segments.csv','outlook.html'):
        assert routes['/'+name]==f'https://raw.githubusercontent.com/wscha231/eth-dashboard/data/event-feed/forecast_site/public/{name}'
    assert 'drive.google' not in json.dumps(config)
    assert routes['/archive/:path*'].endswith('/archive/:path*')
