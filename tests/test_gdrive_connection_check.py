import contextlib
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from scripts import check_gdrive_connection as check


ENV = {"GDRIVE_CLIENT_ID": "123-demo.apps.googleusercontent.com",
       "GDRIVE_CLIENT_SECRET": "test-secret", "GDRIVE_REFRESH_TOKEN": "test-refresh",
       "GDRIVE_FOLDER_ID": "test-folder"}


class FakeDrive:
    def __init__(self, *, mismatch=False, cleanup_fail=False, token_fail=False):
        self.calls = []
        self.mismatch, self.cleanup_fail, self.token_fail = mismatch, cleanup_fail, token_fail

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/token"):
            if self.token_fail:
                raise check.CheckError("HTTP 400: invalid_grant")
            return {"access_token": "test-access"}
        if method == "GET" and not kwargs.get("raw"):
            return {"mimeType": "application/vnd.google-apps.folder", "trashed": False,
                    "capabilities": {"canAddChildren": True}}
        if method == "POST":
            parts = kwargs["body"].split(b"\r\n\r\n")
            self.payload = parts[2].split(b"\r\n--")[0]
            meta = json.loads(parts[1].split(b"\r\n--")[0])
            assert meta["parents"] == [ENV["GDRIVE_FOLDER_ID"]]
            return {"id": "new-test-file"}
        if method == "GET":
            return b"incorrect" if self.mismatch else self.payload
        if method == "PATCH":
            assert "/new-test-file?" in url
            assert json.loads(kwargs["body"]) == {"trashed": True}
            if self.cleanup_fail:
                raise RuntimeError("test-access test-secret")
            return {"trashed": True}
        raise AssertionError("Unexpected request")


class DriveCheckTests(unittest.TestCase):
    def test_roundtrip_and_only_new_file_trashed(self):
        fake = FakeDrive()
        results = check.run(ENV, fake)
        self.assertTrue(all(value == "PASS" for value in results.values()), results)
        self.assertEqual([item[0] for item in fake.calls], ["POST", "GET", "POST", "GET", "PATCH"])

    def test_mismatch_still_cleans_up(self):
        results = check.run(ENV, FakeDrive(mismatch=True))
        self.assertTrue(results["Download match"].startswith("FAIL"))
        self.assertEqual(results["Cleanup"], "PASS")

    def test_cleanup_failure_fails_without_exception_secrets(self):
        results = check.run(ENV, FakeDrive(cleanup_fail=True))
        self.assertTrue(results["Cleanup"].startswith("FAIL"))
        self.assertNotIn("test-access", str(results))
        self.assertNotIn("test-secret", str(results))

    def test_revoked_token_never_touches_drive(self):
        fake = FakeDrive(token_fail=True)
        results = check.run(ENV, fake)
        self.assertIn("invalid_grant", results["Token refresh"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(results["Upload"], "SKIP")

    def test_invalid_config_has_no_network_requests(self):
        for value in ("", '"token"', '{"refresh_token":"token"}', "HTTP/1.1 200 OK", " token"):
            with self.subTest(value=value):
                fake = FakeDrive()
                results = check.run(dict(ENV, GDRIVE_REFRESH_TOKEN=value), fake)
                self.assertTrue(results["Secrets"].startswith("FAIL"))
                self.assertFalse(fake.calls)

    def test_raw_error_body_is_never_reported(self):
        body = io.BytesIO(b'{"error":"invalid_client","error_description":"test-secret"}')
        error = HTTPError("https://oauth2.googleapis.com/token", 401, "test-secret", {}, body)
        with patch.object(check, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(check.CheckError) as caught:
                check.request("POST", "https://oauth2.googleapis.com/token", body=b"secret")
        self.assertEqual(str(caught.exception), "HTTP 401: invalid_client")

    def test_ambiguous_create_is_not_retried(self):
        with patch.object(check, "build_opener") as opener:
            opener.return_value.open.side_effect = URLError("test-secret")
            with self.assertRaises(check.CheckError):
                check.request("POST", check.API, body=b"{}")
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_summary_and_exit_status(self):
        for success in (False, True):
            results = {stage: "PASS" if success else "SKIP" for stage in check.STAGES}
            with patch.object(check, "run", return_value=results), patch.dict(check.os.environ, {}, clear=True):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(check.main(), 0 if success else 1)
                self.assertNotIn("test-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
