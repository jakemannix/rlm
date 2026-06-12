"""Preserve Claude Code session transcripts before cleanup deletes them.

Claude Code prunes ``~/.claude/projects`` transcripts after
``cleanupPeriodDays``; this script snapshots them into a durable archive
that ``rlm.sleep.cc_traces`` can load from. Incremental and idempotent:
files already archived with the same size+mtime are skipped, changed
files are re-copied, and nothing is ever deleted from the archive.

Example:
    python scripts/sleep/extract_cc_sessions.py \
        --dest ~/Documents/cc-session-archive

The archive is a mirror of the source tree plus a ``manifest.json``
recording provenance (source path, size, sha256, archived_at). Keep the
archive OUT of any git repository — these transcripts are personal data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(source: Path, dest: Path) -> dict:
    """Copy new/changed *.jsonl transcripts from source into dest; return stats."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stats = {"copied": 0, "skipped": 0, "bytes_copied": 0}
    for src in sorted(source.rglob("*.jsonl")):
        rel = str(src.relative_to(source))
        st = src.stat()
        entry = manifest.get(rel)
        if entry and entry["size"] == st.st_size and entry["mtime"] == st.st_mtime:
            stats["skipped"] += 1
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest[rel] = {
            "source": str(src),
            "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": sha256_file(target),
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        stats["copied"] += 1
        stats["bytes_copied"] += st.st_size

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats["total_in_manifest"] = len(manifest)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="~/.claude/projects", help="Claude Code projects dir to snapshot"
    )
    parser.add_argument(
        "--dest",
        default="~/Documents/cc-session-archive",
        help="archive directory (must NOT be inside a git repo)",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    dest = Path(args.dest).expanduser()
    if not source.is_dir():
        raise SystemExit(f"source not found: {source}")
    stats = extract(source, dest)
    print(
        f"archived {stats['copied']} new/changed files "
        f"({stats['bytes_copied'] / 1e6:.1f} MB), skipped {stats['skipped']} unchanged; "
        f"manifest now tracks {stats['total_in_manifest']} files at {dest}"
    )


if __name__ == "__main__":
    main()
