"""Copy and stage the volatility-first V2 homepage assets into the production site checkout."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("v2.html", "volatility.js")


def publish(target, source=ROOT, *, stage=False):
    target, source = Path(target).resolve(), Path(source).resolve()
    public = target / "forecast_site/public"
    public.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ASSETS:
        src = source / "forecast_site/public" / name
        dst = public / name
        if not src.is_file():
            raise FileNotFoundError(f"missing V2 site asset: {src}")
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        copied.append(f"forecast_site/public/{name}")
    if stage:
        subprocess.run(["git", "-C", str(target), "add", "-f", "--", *copied], check=True)
    return copied
