import pandas as pd

from scripts.run_event_trading_diagnostic import complete_windows


def test_complete_windows_include_all_source_segments_and_never_bridge_missing_hours():
    grid = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    bars = pd.DataFrame({"product": "ETH-USD", "open_time": grid,
                         "open": 100., "close": 100.})
    bars = bars.drop(index=[0, 4, 5, 11])
    windows = complete_windows(bars, grid[0], grid[-1] + pd.Timedelta(hours=1))
    assert windows == [(grid[1], grid[4]), (grid[6], grid[11])]
    # Future prices cannot choose which window is presented.
    bars["close"] = [1000., 1., 5., 1., 1000., 1., 200., 1.]
    assert complete_windows(bars, grid[0], grid[-1] + pd.Timedelta(hours=1)) == windows
