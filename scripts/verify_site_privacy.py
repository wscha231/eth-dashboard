"""Check the public HTML privacy page and its homepage link, without credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://etherforecast.live/"
POLICY = urljoin(BASE, "privacy.html")


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links, self.ids, self.text = [], set(), []
        self.canonical = None
        self.excluded = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "template"):
            self.excluded += 1
        if self.excluded:
            return
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "template"):
            self.excluded = max(0, self.excluded - 1)

    def handle_data(self, data):
        if not self.excluded:
            self.text.append(data)


def validate(homepage, policy):
    home, notice = Document(homepage), Document(policy)
    if not any(urljoin(BASE, href) == POLICY for href in home.links):
        raise ValueError("homepage privacy link missing from static HTML")
    if notice.canonical != POLICY or "privacy-policy" not in notice.ids:
        raise ValueError("canonical privacy policy page missing")
    required = {"google-data", "sharing", "security", "retention", "deletion", "limited-use", "contact"}
    if not required.issubset(notice.ids) or len(" ".join(notice.text).split()) < 350:
        raise ValueError("privacy policy body incomplete")
    if any(marker in policy for marker in ("[Operator name", "[Email to be", "Draft for operator review")):
        raise ValueError("unpublished draft placeholders present")


def fetch_html(url):
    request = Request(url, headers={"User-Agent": "ETH-Forecast-privacy-check", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=8) as response:
        if response.status != 200 or response.headers.get_content_type() != "text/html":
            raise ValueError("public HTML response required: " + url)
        actual, expected = urlsplit(response.geturl()), urlsplit(url)
        if (actual.scheme, actual.netloc, actual.path) != (expected.scheme, expected.netloc, expected.path):
            raise ValueError("canonical URL unexpectedly redirected: " + url)
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="Validate source files without network access")
    args = parser.parse_args()
    public = ROOT / "forecast_site/public"
    expected_policy = (public / "privacy.html").read_text(encoding="utf-8")
    validate((public / "index.html").read_text(encoding="utf-8"), expected_policy)
    if args.local:
        print("Privacy source verified: static homepage link and complete policy HTML")
        return
    for attempt in range(8):
        try:
            home, policy = fetch_html(BASE), fetch_html(POLICY)
            validate(home, policy)
            if policy != expected_policy:
                raise ValueError("served privacy policy differs from source")
            print("Privacy site verified: homepage_link=present policy_http=200 policy_body=source_match",
                  "url=" + POLICY, "checked_utc=" + datetime.now(timezone.utc).isoformat())
            return
        except (OSError, ValueError) as error:
            print(f"Privacy verification attempt {attempt + 1}/8: {error}", flush=True)
            if attempt == 7:
                raise SystemExit("Public privacy policy verification failed") from error
            time.sleep(15)


if __name__ == "__main__":
    main()
