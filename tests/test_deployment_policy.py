"""Provider waiting periods must preserve data without claiming delivery."""
from datetime import datetime
import json
from pathlib import Path
import subprocess

import pytest

from scripts.deployment_policy import configure, cooldown

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('at,blocked', [
    ('2026-09-12T20:06:30+00:00', False),
    ('2026-09-12T20:06:31+00:00', True),
    ('2026-09-13T20:09:59+00:00', True),
    ('2026-09-13T20:10:00+00:00', False),
])
def test_provider_window_boundaries(at, blocked):
    assert bool(cooldown(now=datetime.fromisoformat(at))) == blocked


def test_cooldown_configuration_preserves_headers_and_restores_production_only(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    untouched = tmp_path / 'forecast_site/public/signals.json'
    untouched.parent.mkdir(parents=True)
    untouched.write_text('{"release_id":"original"}')
    original = json.loads((ROOT/'forecast_site/vercel.json').read_text())
    for at, enabled in [('2026-09-12T21:00:00+00:00', False), ('2026-09-13T20:10:00+00:00', True)]:
        configure(tmp_path, now=datetime.fromisoformat(at), stage=True)
        actual = json.loads((tmp_path/'forecast_site/vercel.json').read_text())
        assert actual.pop('git') == {'deploymentEnabled': {'**': False, 'data/daily-forecast': enabled}}
        assert actual == {k: v for k, v in original.items() if k != 'git'}
        assert untouched.read_text() == '{"release_id":"original"}'
        staged = subprocess.check_output(['git', '-C', str(tmp_path), 'diff', '--cached', '--name-only'], text=True)
        assert staged.splitlines() == ['forecast_site/vercel.json']


def test_invalid_policy_cannot_enable_deployment(tmp_path):
    (tmp_path/'ops').mkdir()
    (tmp_path/'ops/deployment_cooldown.json').write_text(json.dumps({
        'detected_at': '2026-09-12T20:00:00Z', 'not_before': '2026-09-12T19:00:00Z',
    }))
    with pytest.raises(ValueError, match='interval'):
        configure(tmp_path, source=tmp_path)
    assert not (tmp_path/'forecast_site/vercel.json').exists()
