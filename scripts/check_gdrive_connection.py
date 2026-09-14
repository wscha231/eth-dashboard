"""Bounded Drive OAuth round-trip check; credentials never leave the runner."""
import configparser
import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

API = "https://www.googleapis.com/drive/v3/files"
KEYS = ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN", "GDRIVE_FOLDER_ID")
STAGES = ("Secrets", "Token refresh", "Folder access", "Upload", "Download match", "Cleanup")


class CheckError(Exception):
    """Only static, non-sensitive diagnostic messages belong here."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(method, url, *, token=None, body=None, content_type="application/json", raw=False):
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = "Bearer " + token
    # Never retry creation: an uncertain response may already have created a file.
    attempts = 2 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            req = Request(url, data=body, headers=headers, method=method)
            with build_opener(NoRedirect()).open(req, timeout=30) as response:
                data = response.read(65537)
            if len(data) > 65536:
                raise CheckError("Response exceeds the test size limit")
            if raw:
                return data
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise CheckError("Unexpected response format")
            return parsed
        except HTTPError as exc:
            status = exc.code
            detail = ""
            try:
                code = json.loads(exc.read(8192)).get("error")
                if code in ("invalid_client", "invalid_grant", "invalid_scope"):
                    detail = ": " + code
            except (ValueError, TypeError, AttributeError):
                pass
            finally:
                exc.close()
            if status in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep(1)
                continue
            raise CheckError("HTTP " + str(status) + detail) from None
        except (URLError, OSError):
            if attempt + 1 < attempts:
                time.sleep(1)
                continue
            raise CheckError("Network request failed; check connectivity") from None
        except (ValueError, UnicodeError):
            raise CheckError("Unexpected response format") from None


def mask(value, env):
    if env.get("GITHUB_ACTIONS") == "true":
        escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print("::add-mask::" + escaped)


def credentials(env):
    """Accept one plain-text rclone-format secret or the original four secrets."""
    config = env.get("GDRIVE_RCLONE_CONFIG", "")
    if not config and env.get("GDRIVE_REFRESH_TOKEN", "").lstrip().startswith("[gdrive]"):
        config = env["GDRIVE_REFRESH_TOKEN"]
    if not config:
        return {key: env.get(key, "") for key in KEYS}
    try:
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(config)
        section = parser["gdrive"]
        if section.get("type") != "drive":
            raise ValueError()
        token = json.loads(section.get("token", "{}"))
        if not isinstance(token, dict):
            raise ValueError()
        values = {KEYS[0]: section.get("client_id", ""), KEYS[1]: section.get("client_secret", ""),
                  KEYS[2]: token.get("refresh_token", ""), KEYS[3]: section.get("root_folder_id", "")}
        if not all(isinstance(value, str) for value in values.values()):
            raise ValueError()
        # Structured secrets need individual masking; never log access_token either.
        for value in [*values.values(), token.get("access_token", "")]:
            if isinstance(value, str) and value:
                mask(value, env)
        return values
    except (configparser.Error, KeyError, ValueError, TypeError):
        raise CheckError("Invalid single-secret config: use [gdrive], type=drive and a JSON token") from None


def run(env=None, call=request):
    env = os.environ if env is None else env
    results = {stage: "SKIP" for stage in STAGES}
    stage, file_id, token = "Secrets", None, None
    try:
        values = credentials(env)
        for key in KEYS:
            value = values[key]
            if not value:
                raise CheckError("Missing " + key)
            if (any(c.isspace() for c in value) or value.startswith(("{", '[', '"', "'", "HTTP/"))):
                raise CheckError("Paste only the raw value into " + key)
            values[key] = value
            mask(value, env)
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.apps\.googleusercontent\.com", values[KEYS[0]]):
            raise CheckError("Invalid GDRIVE_CLIENT_ID format")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", values[KEYS[3]]):
            raise CheckError("GDRIVE_FOLDER_ID must be an ID, not a URL")
        results[stage] = "PASS"

        stage = "Token refresh"
        grant = {"client_id": values[KEYS[0]], "client_secret": values[KEYS[1]],
                 "refresh_token": values[KEYS[2]], "grant_type": "refresh_token"}
        response = call("POST", "https://oauth2.googleapis.com/token", body=urlencode(grant).encode(),
                        content_type="application/x-www-form-urlencoded")
        token = response.get("access_token")
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise CheckError("No usable access token returned")
        mask(token, env)
        results[stage] = "PASS"

        stage = "Folder access"
        folder = values[KEYS[3]]
        meta = call("GET", API + "/" + folder + "?fields=mimeType,trashed,capabilities(canAddChildren)&supportsAllDrives=true", token=token)
        if meta.get("mimeType") != "application/vnd.google-apps.folder" or meta.get("trashed") is not False:
            raise CheckError("Target is not an active folder")
        if meta.get("capabilities", {}).get("canAddChildren") is not True:
            raise CheckError("Target folder does not permit adding files")
        results[stage] = "PASS"

        stage = "Upload"
        nonce = uuid.uuid4().hex
        run_id = env.get("GITHUB_RUN_ID", "local")
        run_id = run_id if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", run_id) else "run"
        name = "etherforecast-connection-check-" + run_id + "-" + nonce + ".json"
        payload = json.dumps({"purpose": "Drive connection check", "nonce": nonce}).encode()
        metadata = json.dumps({"name": name, "parents": [folder], "mimeType": "application/json"}).encode()
        boundary = "check_" + nonce
        body = (b"--" + boundary.encode() + b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
                + metadata + b"\r\n--" + boundary.encode() + b"\r\nContent-Type: application/json\r\n\r\n"
                + payload + b"\r\n--" + boundary.encode() + b"--\r\n")
        response = call("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id&supportsAllDrives=true",
                        token=token, body=body, content_type="multipart/related; boundary=" + boundary)
        candidate = response.get("id")
        if not isinstance(candidate, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate) or candidate == folder:
            raise CheckError("Upload returned no usable new file ID; inspect test files in Drive")
        file_id = candidate
        results[stage] = "PASS"

        stage = "Download match"
        received = call("GET", API + "/" + file_id + "?alt=media&supportsAllDrives=true", token=token, raw=True)
        if received != payload:
            raise CheckError("Downloaded bytes differ from the uploaded test payload")
        results[stage] = "PASS"
    except CheckError as exc:
        results[stage] = "FAIL - " + str(exc)
    except Exception:
        # Do not expose exception reprs: third-party responses may contain secrets.
        results[stage] = "FAIL - Unexpected error (details suppressed)"
    finally:
        if file_id is not None:
            try:
                cleanup = call("PATCH", API + "/" + file_id + "?fields=trashed&supportsAllDrives=true", token=token,
                               body=b'{"trashed":true}')
                if cleanup.get("trashed") is not True:
                    raise CheckError("Trash operation not confirmed")
                results["Cleanup"] = "PASS"
            except Exception:
                results["Cleanup"] = "FAIL - Test file may remain; inspect etherforecast-connection-check files in Drive"
        elif results["Upload"].startswith("FAIL"):
            results["Cleanup"] = "SKIP - No returned file ID; an uncertain upload may leave a test file"
    return results


def main():
    results = run()
    lines = ["## Google Drive connection check", "", "| Check | Result |", "|---|---|"]
    lines.extend("| " + stage + " | " + result + " |" for stage, result in results.items())
    lines.extend(["", "Folder HTTP 404: check the account, folder ID and drive.file authorization.",
                  "invalid_grant: reauthorize and check External/Testing (7-day token expiry).",
                  "A passing check verifies this run only; it does not enable automatic backups."])
    summary = "\n".join(lines) + "\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    return 0 if all(result == "PASS" for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
