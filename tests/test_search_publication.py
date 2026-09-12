"""Search publication must preserve issued evidence and reject misleading inputs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

privacy_spec = importlib.util.spec_from_file_location("site_privacy", Path(__file__).resolve().parents[1] / "scripts/verify_site_privacy.py")
privacy = importlib.util.module_from_spec(privacy_spec)
privacy_spec.loader.exec_module(privacy)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("search_assets", ROOT / "scripts/publish_search_assets.py")
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
NOW = datetime(2026, 9, 12, 8, 30, tzinfo=timezone.utc)


def fixture():
    records = []
    for h in search.HORIZONS:
        records.append({
            "forecast_id": f"issued-{h}", "horizon_seconds": h * 3600,
            "slot": "2026-09-12T08:00:00+00:00", "input_cutoff": "2026-09-12T08:00:00+00:00",
            "available_at": "2026-09-12T08:05:00+00:00", "issued_at": "2026-09-12T08:10:00+00:00",
            "window_start": "2026-09-12T09:00:00+00:00",
            "target_end": (NOW.replace(hour=9, minute=0) + timedelta(hours=h)).isoformat(),
            "reference_price": 2510.12, "price_quantiles": [2412.34, 2543.21, 2678.90],
            "terminal_down_flat_up": [.2, .5, .3], "lower_barrier_price": 2450, "upper_barrier_price": 2600,
        })
    return {"schema_version": 1, "release_id": "source-release", "status": "ready",
            "generated_at": "2026-09-12T08:15:00+00:00", "expected_slot": "2026-09-12T08:00:00+00:00",
            "current": records}


class SearchPublicationTests(unittest.TestCase):
    def test_values_are_original_and_input_is_not_mutated(self):
        payload = fixture()
        before = deepcopy(payload)
        html = search.render_snapshot(payload, NOW)
        self.assertEqual(payload, before)
        self.assertEqual(html.count('data-forecast-id="issued-'), 6)
        for value in ("$2,543.21", "$2,412.34 – $2,678.90", "20.0%", "50.0%", "30.0%", "2026-09-12 08:15 UTC"):
            self.assertIn(value, html)

    def test_stale_release_is_explicitly_archived(self):
        html = search.render_snapshot(fixture(), NOW + timedelta(days=2))
        self.assertIn("archived values, not a current outlook", html)
        self.assertIn("2026-09-12 08:15 UTC", html)

    def test_missing_or_delayed_release_cannot_show_prices(self):
        for payload in ({}, [], None, {**fixture(), "status": "delayed"}):
            html = search.render_snapshot(payload, NOW)
            self.assertEqual(html.count("Unavailable in this release"), 6)
            self.assertNotIn("$2,543.21", html)

    def test_invalid_future_metadata_cannot_show_prices(self):
        for key, value in (("generated_at", "2026-09-13T08:15:00+00:00"), ("expected_slot", "2026-09-12T08:03:00+00:00"), ("release_id", "")):
            html = search.render_snapshot({**fixture(), key: value}, NOW)
            self.assertEqual(html.count("Unavailable in this release"), 6)

    def test_corrupt_probability_prices_or_clock_omit_only_that_horizon(self):
        for key, value in (
            ("terminal_down_flat_up", [.2, .5, .4]), ("terminal_down_flat_up", [float("nan"), .5, .5]),
            ("price_quantiles", [3000, 2500, 2800]), ("price_quantiles", [2000, True, 3000]),
            ("reference_price", -1), ("issued_at", "2026-09-12T09:01:00+00:00"),
            ("target_end", "2026-09-12T09:00:00+00:00"), ("slot", "2026-09-12T07:00:00+00:00"),
        ):
            payload = fixture()
            payload["current"][0][key] = value
            html = search.render_snapshot(payload, NOW)
            self.assertEqual(html.count("Unavailable in this release"), 1, key)
            self.assertEqual(html.count('data-forecast-id="issued-'), 5, key)

    def test_duplicate_horizon_is_not_silently_selected(self):
        payload = fixture()
        payload["current"].append(deepcopy(payload["current"][0]))
        self.assertEqual(search.render_snapshot(payload, NOW).count("Unavailable in this release"), 1)

    def test_source_identifiers_are_escaped(self):
        payload = fixture()
        payload["release_id"] = '<img src=x onerror="bad()">'
        payload["current"][0]["forecast_id"] = 'bad" onclick="bad()'
        html = search.render_snapshot(payload, NOW)
        self.assertNotIn('<img src=x', html)
        self.assertNotIn(' onclick="bad()', html)
        self.assertIn('&lt;img', html)
        self.assertIn('bad&quot; onclick=&quot;', html)

    def test_publisher_preserves_data_tokens_and_stages_only_owned_assets(self):
        with TemporaryDirectory() as temp:
            target = Path(temp)
            subprocess.run(["git", "init", "-q", str(target)], check=True)
            public = target / "forecast_site/public"
            public.mkdir(parents=True)
            retained = {
                "signals.json": json.dumps(fixture()), "google-owner.html": "owner token",
                "donations.json": '{"wallets": ["original"]}', "BingSiteAuth.xml": "bing owner token",
                "unrelated.json": "pending unrelated change",
            }
            for name, contents in retained.items():
                (public / name).write_text(contents)
            search.publish(target, stage=True)
            original_html = (public / "outlook.html").read_text()
            # The deployed bundle must fix the original broken-link/404 case.
            privacy.validate((public / "index.html").read_text(), (public / "privacy.html").read_text())
            for page in ("ko/index.html", "outlook.html"):
                self.assertIn('href="https://etherforecast.live/privacy.html"', (public / page).read_text())
            search.publish(target, stage=True)
            self.assertEqual((public / "outlook.html").read_text(), original_html)
            for name, contents in retained.items():
                self.assertEqual((public / name).read_text(), contents)
            staged = subprocess.check_output(["git", "-C", str(target), "diff", "--cached", "--name-only"], text=True).splitlines()
            self.assertEqual(set(staged), {f"forecast_site/public/{name}" for name in (*search.ASSETS, "outlook.html")} | {"forecast_site/vercel.json"})

    def test_privacy_validation_rejects_broken_link_soft_404_and_draft(self):
        home = (ROOT / "forecast_site/public/index.html").read_text()
        policy = (ROOT / "forecast_site/public/privacy.html").read_text()
        for html, body in (
            (home.replace("https://etherforecast.live/privacy.html", "/missing-policy.html"), policy),
            (home, '<!doctype html><html><body>404: This page could not be found.</body></html>'),
            (home, policy.replace('id="google-data"', 'id="absent-section"')),
            (home, policy + "<!-- [Email to be confirmed] -->"),
        ):
            with self.assertRaises(ValueError):
                privacy.validate(html, body)


if __name__ == "__main__":
    unittest.main()
