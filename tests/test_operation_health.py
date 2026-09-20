from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json

import pytest

from scripts.build_operation_health import attribute, build, INFERENCE, REFRESH_STEPS

NOW = datetime(2026,9,19,18,0,tzinfo=timezone.utc)


def example():
    metadata={'id':123,'run_attempt':1,'event':'schedule','head_sha':'pinned',
              'created_at':'2026-09-19T17:56:30Z','jobs':[
                  {'name':'forecast','run_attempt':1,'started_at':'2026-09-19T17:56:34Z',
                   'steps':[{'name':INFERENCE,'conclusion':'success','started_at':'2026-09-19T17:57:13Z',
                             'completed_at':'2026-09-19T17:57:28Z'}]}]}
    signal={'generated_at':'2026-09-19T17:57:27Z','expected_slot':'2026-09-19T17:00:00Z',
            'status':'delayed','errors':[{'horizon':6,'reason':'issuance_deadline: the hourly :55 cutoff has passed'}]}
    return metadata,signal


def test_actual_late_creation_is_distinct_from_input_failure():
    metadata,signal=example();result=attribute(metadata,signal,run_id='123',attempt='1',now=NOW)
    assert result['causes']==['issuance_clock_guard','run_created_after_issuance_deadline']
    assert result['dispatch_to_job_seconds']==4
    assert 'unproven' in result['shared_lock_cause']


@pytest.mark.parametrize('where,expected', [
    ('job','job_started_after_issuance_deadline'),('pre','preinference_steps_crossed_deadline')])
def test_distinguishes_runner_wait_from_local_preinference_time(where,expected):
    metadata,signal=example();metadata['created_at']='2026-09-19T17:10:00Z'
    if where=='pre':metadata['jobs'][0]['started_at']='2026-09-19T17:10:05Z'
    result=attribute(metadata,signal,run_id='123',attempt='1',now=NOW)
    assert expected in result['causes']
    assert 'run_created_after_issuance_deadline' not in result['causes']


@pytest.mark.parametrize('bad', [None, {}, {'generated_at':None}, {'generated_at':'2026-09-18T17:57:27Z'}])
def test_stale_missing_or_corrupt_feed_is_not_current_evidence(bad):
    metadata,_=example()
    result=attribute(metadata,bad,run_id='123',attempt='1',now=NOW)
    assert result['causes']==['no_current_inference_report'] and result['slot'] is None


@pytest.mark.parametrize('reason,expected', [
    ('latest complete input window unavailable','input_window_unavailable'),
    ('current monthly checkpoint unavailable; replay required','checkpoint_unavailable'),
    ('other','other_inference_error')])
def test_uses_observed_reason_without_guessing(reason,expected):
    metadata,signal=example();signal['errors']=[{'reason':reason}]
    result=attribute(metadata,signal,run_id='123',attempt='1',now=NOW)
    assert expected in result['causes']


def test_wrong_run_or_attempt_rejected():
    metadata,signal=example()
    for run,attempt in [('999','1'),('123','2')]:
        with pytest.raises(ValueError):attribute(metadata,signal,run_id=run,attempt=attempt,now=NOW)


def test_retry_does_not_call_original_creation_to_job_time_queue_wait():
    metadata,signal=example();metadata['run_attempt']=2;metadata['jobs'][0]['run_attempt']=2
    assert attribute(metadata,signal,run_id='123',attempt='2',now=NOW)['dispatch_to_job_seconds'] is None


def test_preinference_failure_does_not_import_old_success():
    metadata,signal=example();metadata['jobs'][0]['steps'][0]['conclusion']='skipped'
    assert attribute(metadata,signal,run_id='123',attempt='1',now=NOW)['causes']==['inference_not_reached_or_cancelled']


def test_optional_download_failure_exposed_even_when_later_install_is_skipped():
    metadata,signal=example();outcomes=dict.fromkeys(REFRESH_STEPS,'skipped')
    outcomes['research_download']='failure'
    report=build(metadata,signal,outcomes,run_id='123',attempt='1',now=NOW)
    assert report['input_refresh_degraded']
    assert report['failed_refresh_steps']==['research_download']
    assert report['promotion']=='none' and report['historical_gaps_repaired'] is False


def test_clean_skipped_refresh_is_healthy():
    metadata,signal=example();outcomes=dict.fromkeys(REFRESH_STEPS,'skipped')
    assert not build(metadata,signal,outcomes,run_id='123',attempt='1',now=NOW)['input_refresh_degraded']
