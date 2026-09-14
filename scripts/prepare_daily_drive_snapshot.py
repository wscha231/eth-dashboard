"""Make a data-only daily recovery artifact, including committed SQLite WAL rows."""
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.archive_to_drive import BRANCHES
from scripts.gdrive_store import CheckError, allowed


def prepare(root, target):
    root, target = Path(root).resolve(), Path(target).absolute()
    if target.exists():
        raise CheckError("Daily snapshot target already exists")
    if not all((root / p).is_file() for p in ("lake/gold/eth_master_daily.csv", "forecast_site/predictions.db")):
        raise CheckError("Daily recovery source is incomplete")
    target.mkdir(parents=True)
    for prefix in BRANCHES["data/daily-forecast"][1]:
        source = root / prefix
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise CheckError("Daily data contains a symlink")
            if not path.is_file() or not allowed(path.relative_to(root).as_posix()):
                continue
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".db":
                with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(destination) as dst:
                    src.backup(dst)
                    if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise CheckError("Daily SQLite snapshot failed")
                    dst.execute("PRAGMA journal_mode=DELETE")
            else:
                shutil.copy2(path, destination)


if __name__ == "__main__":
    prepare(Path.cwd(), "daily-drive-state")
