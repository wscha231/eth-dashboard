import json

import pandas as pd
import pytest

from signal_pipeline.v2_replay import discover_current_review, score_shadow_review


def row(slot, ret, terminal, up, down, q10, q50, q90, probs, hit_up, hit_down):
    return {
        "slot": pd.Timestamp(slot).isoformat(),
        "target_end": (pd.Timestamp(slot) + pd.Timedelta(hours=7)).isoformat(),
        "return": ret,
        "terminal": terminal,
        "up": up,
        "down": down,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "p_down": probs[0],
        "p_flat": probs[1],
        "p_up": probs[2],
        "hit_up": hit_up,
        "hit_down": hit_down,
    }


def review():
    slots = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"]
    truth = [
        (0.02, 1, 0, 0),
        (0.08, 2, 1, 0),
        (-0.05, 0, 0, 1),
    ]
    candidate = []
    incumbent = []
    baseline = []
    for slot, (ret, terminal, up, down) in zip(slots, truth):
        candidate.append(row(slot, ret, terminal, up, down, -0.08, ret * 0.8, 0.10, [0.1, 0.7, 0.2] if terminal == 1 else ([0.1, 0.2, 0.7] if terminal == 2 else [0.7, 0.2, 0.1]), 0.8 if up else 0.1, 0.8 if down else 0.1))
        incumbent.append(row(slot, ret, terminal, up, down, -0.10, 0.0, 0.10, [0.2, 0.6, 0.2], 0.5, 0.5))
        baseline.append(row(slot, ret, terminal, up, down, -0.12, 0.0, 0.12, [0.25, 0.5, 0.25], 0.3, 0.3))
    return {
        "horizon_hours": 6,
        "cutoff": "2026-01-04T00:00:00+00:00",
        "heads": {"point": "candidate", "interval": "candidate", "probability": "candidate"},
        "rows": candidate,
        "incumbent_rows": incumbent,
        "baseline_rows": baseline,
    }


def test_score_shadow_review_uses_same_origins_and_baseline_only_volatility():
    r = review()
    slots = pd.DatetimeIndex(pd.to_datetime([x["slot"] for x in r["rows"]], utc=True))
    features = pd.DataFrame({"sigma": [0.01, 0.012, 0.011]}, index=slots)
    targets = pd.DataFrame({"realized_variance_total": [0.0007, 0.0010, 0.0008]}, index=slots)
    result = score_shadow_review(r, features, targets, 6)
    assert result["origins"] == 3
    assert result["head_status"]["volatility"] == "baseline_only_until_R2"
    assert result["relative_skill"]["volatility_candidate_vs_persistence_qlike"] is None
    assert result["mandatory_baselines"]["volatility_persistence"]["qlike"] >= 0
    assert result["candidate"]["center"]["rows"] == 3
    assert result["candidate"]["event"]["rows"] == 3


def test_score_shadow_review_rejects_origin_mismatch():
    r = review()
    r["baseline_rows"] = r["baseline_rows"][:-1]
    slots = pd.DatetimeIndex(pd.to_datetime([x["slot"] for x in r["rows"]], utc=True))
    features = pd.DataFrame({"sigma": [0.01, 0.012, 0.011]}, index=slots)
    targets = pd.DataFrame({"realized_variance_total": [0.0007, 0.0010, 0.0008]}, index=slots)
    with pytest.raises(ValueError, match="identical origins"):
        score_shadow_review(r, features, targets, 6)


def test_discover_current_review_matches_full_public_payload(tmp_path):
    root = tmp_path
    models = root / "shadow_models"
    models.mkdir()
    r = review()
    public = {k: v for k, v in r.items() if k not in {"rows", "incumbent_rows", "baseline_rows"}}
    (models / "review_6_good.json").write_text(json.dumps(r), encoding="utf-8")
    stale = dict(r)
    stale["cutoff"] = "2025-12-31T00:00:00+00:00"
    (models / "review_6_stale.json").write_text(json.dumps(stale), encoding="utf-8")
    assert discover_current_review(root, 6, public).name == "review_6_good.json"


def test_discover_current_review_fails_on_ambiguous_duplicate(tmp_path):
    models = tmp_path / "shadow_models"
    models.mkdir()
    r = review()
    public = {k: v for k, v in r.items() if k not in {"rows", "incumbent_rows", "baseline_rows"}}
    for name in ("review_6_a.json", "review_6_b.json"):
        (models / name).write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        discover_current_review(tmp_path, 6, public)
