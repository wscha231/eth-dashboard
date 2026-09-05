"""Exercise displayed values, including timezone and as-issued history semantics."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("timezone", ["UTC", "Asia/Seoul"])
def test_forecast_display_keeps_issued_values(timezone):
    if not shutil.which("node"):
        pytest.skip("Node is required for browser JavaScript behavior checks")
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('forecast_site/public/index.html', 'utf8');
const source = html.split('<script>')[1].split('(async function main()')[0];
const context = vm.createContext({document: {getElementById: () => ({})}, window: {}});
vm.runInContext(source, context);
const evaluate = expression => vm.runInContext(expression, context);
assert.equal(evaluate('fmtDate("2026-09-05 00:00:00")'), '2026-09-05');
assert.equal(evaluate('fmtPct(null)'), '--');
assert.equal(evaluate('livePrice=9999; rollingTarget(110,100)'), 110);
const point = evaluate('liveHistoryRows([{horizon_days:30,actual_close:120,forecast_target_timestamp_utc:"2026-09-05",input_timestamp_utc:"2026-08-06",reference_price:100,regression_predicted_close:110,active_predicted_close:115,forecast_actionability:"range_only",forecast_point_price_reliable:0}],30)[0]');
assert.equal(point.predicted, 110);
assert.equal(point.raw, 100);
const range = evaluate('practicalRange(100,{},30,{regression_lower_close_10:80,regression_upper_close_90:130},{})');
assert.equal(range.lower, 80);
assert.equal(range.upper, 130);
'''
    subprocess.run(["node", "-e", script], cwd=root, env={**os.environ, "TZ": timezone}, check=True)
