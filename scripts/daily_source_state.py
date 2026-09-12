"""Allowlisted normalized daily caches survive disposable runner restarts."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def allowed(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and
            len(path.parts) == 4 and path.parts[:2] == ("lake", "raw") and
            path.parts[2] in ("market", "vendor") and path.suffix == ".csv")


def persist(root, target):
    copied = []
    for folder in ("market", "vendor"):
        for source in sorted((Path(root) / "lake/raw" / folder).glob("*.csv")):
            name = source.relative_to(root).as_posix()
            if not allowed(name) or source.is_symlink() or not source.is_file():
                continue
            destination = Path(target) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(name)
    return copied


def restore(root, ref):
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref, "--", "lake/raw"], cwd=root, text=True).splitlines()
    restored = []
    for name in names:
        if not allowed(name):
            continue
        # Only regular tracked files; never restore a symlink as a data file.
        entry = subprocess.check_output(["git", "ls-tree", ref, "--", name], cwd=root, text=True)
        if not entry.startswith("100644 "):
            continue
        data = subprocess.check_output(["git", "show", ref + ":" + name], cwd=root)
        destination = Path(root) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        restored.append(name)
    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("restore", "persist"))
    parser.add_argument("target")
    args = parser.parse_args()
    root = Path.cwd()
    names = restore(root, args.target) if args.action == "restore" else persist(root, Path(args.target))
    print(f"Daily source cache {args.action}: {len(names)} normalized CSV files")
