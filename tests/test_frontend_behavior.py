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
const elements = {};
const element = () => ({style:{}, children:[], appendChild(child){this.children.push(child)}, replaceChildren(...children){this.children=children}});
const context = vm.createContext({document: {getElementById: id => elements[id] ||= element(), createElement:element}, window: {}});
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
evaluate('renderForwardResearch({source:{latest_source_day:"2026-09-04",expected_source_day:"2026-09-04",ready_for_current_origin:true},latest_origin:"2026-09-05",resolved_count:0,pending_count:6,latest:[{candidate:"hist_price_flow",horizon:30,target:"2026-10-05",predicted_price:110,predicted_return:.1,probability_down_flat_up:[.2,.5,.3]}],metrics:[]})');
const cells = elements['forward-table'].children[0].children[1].children;
assert.equal(cells[4].textContent, '30.0% / 50.0% / 20.0%');
assert.equal(cells[6].textContent, 'Awaiting outcomes');
assert.match(elements['forward-status'].textContent, /0 resolved \/ 6 pending/);
const payload = {horizons:{'7':{last_target:'2026-09-05',points:[{origin:'2026-08-29',target:'2026-09-05',reference_price:100,actual_price:120,actual_return:.2,returns:{extra_trees:.1,no_change_anchor:0}}]}}};
context.historyFixture=payload;
assert.equal(evaluate('fullHistoryRows(historyFixture,7,"extra_trees","all","price")[0].predicted'),110.00000000000001);
assert.equal(evaluate('fullHistoryRows(historyFixture,7,"extra_trees","recent","return")[0].predicted'),.1);
assert.equal(evaluate('fullHistoryRows(historyFixture,7,"extra_trees","recent","return")[0].raw'),0);
assert.equal(evaluate('fullHistoryRows(historyFixture,7,"no_change_anchor","all","return")[0].raw'),null);
assert.equal(evaluate('fullHistoryRows(historyFixture,7,"extra_trees","2025","price").length'),0);
'''
    subprocess.run(["node", "-e", script], cwd=root, env={**os.environ, "TZ": timezone}, check=True)
