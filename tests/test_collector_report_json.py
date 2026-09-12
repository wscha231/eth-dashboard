import json

import numpy as np
import pandas as pd

from eth_data_collector import report_json


def test_missing_source_measurements_are_valid_json_null_without_hiding_errors():
    payload = {"sources": [{"status": "error", "note": "HTTP 451", "age_days": np.nan}],
               "quality": {"positive_infinity": np.inf, "negative_infinity": -np.inf,
                           "missing": pd.NA, "timestamp": pd.NaT},
               "valid": [np.float64(1.5), np.int64(15), np.bool_(True), 0]}
    def reject(value):
        raise ValueError("nonstandard JSON constant: " + value)
    parsed = json.loads(report_json(payload), parse_constant=reject)
    assert parsed["sources"] == [{"status": "error", "note": "HTTP 451", "age_days": None}]
    assert all(value is None for value in parsed["quality"].values())
    assert parsed["valid"] == [1.5, 15, True, 0]
