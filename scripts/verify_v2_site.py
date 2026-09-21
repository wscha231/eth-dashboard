"""Verify the deployed V2 homepage and research-only availability status without credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "forecast_site/public"
BASE = "https://etherforecast.live/"
STATIC = {
    "root": ("", PUBLIC / "v2.html", "text/html"),
    "script": ("volatility.js", PUBLIC / "volatility.js", "text/javascript"),
    "korean": ("ko/", PUBLIC / "ko/index.html", "text/html"),
}
AVAILABILITY = urljoin(BASE, "availability_shadow.json")
MAX_BYTES = 2 * 1024 * 1024


def aware(value):
    if not isinstance(value, str):
        raise ValueError("timestamp string required")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return dt.astimezone(timezone.utc)


def validate_static(name, body, expected):
    if body != expected:
        raise ValueError(f"served {name} differs from current source")
    if name == "root":
        required = (
            'id="service"', 'id="release-state"', 'id="input-state"',
            "Publication, inputs and evidence are separate.",
            "Center remains 0 / 6 PASS",
            "availability fallback is a separate Center-only no-change shadow",
        )
        if any(marker not in body for marker in required):
            raise ValueError("deployed root lacks current V2 evidence markers")
    elif name == "script":
        required = ("availability_shadow.json", "Strict inputs complete",
                    "Fallback shadow eligible", "public forecast or proof of accuracy")
        if any(marker not in body for marker in required):
            raise ValueError("deployed V2 script lacks current status logic")
        if "price_quantiles" in body:
            raise ValueError("V2 volatility script reintroduced price-quantile headline logic")
    elif name == "korean":
        if "Center-only 연구 shadow" not in body or "검증된 headline 예측으로 표시하지 않습니다" not in body:
            raise ValueError("deployed Korean guidance lacks current evidence boundary")


def validate_availability(value):
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("availability status schema mismatch")
    policy = value.get("policy")
    decision = value.get("decision")
    if not isinstance(policy, dict) or not isinstance(decision, dict):
        raise ValueError("availability policy/decision required")
    expected = {
        "role": "availability_shadow_only",
        "head": "center",
        "promotion": "none",
        "public_delivery": False,
        "imputation": False,
    }
    for key, wanted in expected.items():
        if policy.get(key) != wanted:
            raise ValueError(f"availability policy changed: {key}")
    if value.get("promotion") != "none" or value.get("predictive_skill") != "not_claimed":
        raise ValueError("availability report implies unsupported promotion/skill")
    if decision.get("public_delivery") is not False:
        raise ValueError("availability decision must remain non-public")
    if decision.get("status") not in {"not_needed", "eligible", "blocked"}:
        raise ValueError("unknown availability decision")
    aware(value["generated_at"])
    aware(decision["origin"])
    if decision.get("status") == "not_needed":
        missing = decision.get("strict_missing_counts")
        if not isinstance(missing, dict) or any(missing.get(p) != 0 for p in ("ETH-USD", "BTC-USD")):
            raise ValueError("not_needed must mean the strict ETH/BTC input condition is complete")


def fetch(url, *, expected_type=None, token=None):
    query = urlencode({"deploy_verify": token or int(time.time())})
    separator = "&" if "?" in url else "?"
    request = Request(url + separator + query, headers={
        "User-Agent": "EtherForecast-deployment-verifier/1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Accept": "application/json,text/html,text/javascript,*/*;q=0.1",
    })
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {url}")
        actual = urlsplit(response.geturl())
        wanted = urlsplit(url)
        if (actual.scheme, actual.netloc, actual.path) != (wanted.scheme, wanted.netloc, wanted.path):
            raise ValueError("unexpected redirect: " + url)
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("response exceeds bounded verifier size: " + url)
        if expected_type:
            content_type = response.headers.get_content_type()
            allowed = {expected_type}
            if expected_type == "text/javascript":
                allowed.update({"application/javascript", "application/x-javascript"})
            if content_type not in allowed:
                raise ValueError(f"unexpected content type {content_type}: {url}")
        return raw.decode("utf-8")


def verify(public=PUBLIC, *, token=None):
    for name, (suffix, path, content_type) in STATIC.items():
        expected = path.read_text(encoding="utf-8")
        body = fetch(urljoin(BASE, suffix), expected_type=content_type, token=token)
        validate_static(name, body, expected)
    raw = fetch(AVAILABILITY, expected_type="application/json", token=token)
    value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    validate_availability(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--public-dir", type=Path, default=PUBLIC)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.attempts < 1 or not 0 <= args.sleep_seconds <= 30:
        parser.error("invalid retry bounds")
    if args.local:
        root = (args.public_dir / "v2.html").read_text(encoding="utf-8")
        script_body = (args.public_dir / "volatility.js").read_text(encoding="utf-8")
        korean = (args.public_dir / "ko/index.html").read_text(encoding="utf-8")
        validate_static("root", root, root)
        validate_static("script", script_body, script_body)
        validate_static("korean", korean, korean)
        print("V2 source contract verified")
        return
    last = None
    for attempt in range(1, args.attempts + 1):
        try:
            value = verify(args.public_dir, token=f"{int(time.time())}-{attempt}")
            print("V2 deployment verified:",
                  "root=source_match",
                  "script=source_match",
                  "korean=source_match",
                  "availability_status=" + str(value["decision"]["status"]),
                  "availability_public_delivery=false",
                  "checked_utc=" + datetime.now(timezone.utc).isoformat())
            return
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            last = exc
            print(f"V2 deployment verification attempt {attempt}/{args.attempts}: {exc}", flush=True)
            if attempt < args.attempts:
                time.sleep(args.sleep_seconds)
    raise SystemExit("V2 deployment verification failed") from last


if __name__ == "__main__":
    main()
