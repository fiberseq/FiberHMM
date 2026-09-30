#!/usr/bin/env python
"""Regenerate the git-derived sections of fiberhmm/advisories.json.

The advisory rules (ids, text, fixes) are maintained by hand in the JSON. This
tool fills in what must come from the repository history:

``tables``
    sha256 of every version of the watched emission tables (Hia5 Nanopore and
    DddB, under ``fiberhmm/models``, the root ``models/`` mirror and
    ``legacy/``) with the paths, commits and releases that shipped it, and a
    status: ``gt_swapped`` for every version that existed before the rule's
    fix commit (plus the kept ``*_gt_swapped_legacy.json`` copies), ``fixed``
    for the version the fix introduced.
``commits``
    every commit reachable from HEAD: the ``__version__`` it reported and
    which rule fix commits it contains. A commit named in a header (declared
    ``fiberhmm_commit`` or a ``frozen_<sha>`` path in ``@PG CL``) then decides
    a rule exactly, even where the version string is shared by builds on both
    sides of a fix (dev snapshots reported 2.16.8 while containing 3.0 fixes).

Run from a git checkout:  python tools/build_advisory_index.py [--check]
``--check`` exits 1 when the committed JSON disagrees with the history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "fiberhmm" / "advisories.json"
WATCHED = {
    # table name -> file-name pattern (any models/ directory, incl. legacy/)
    "hia5_nanopore": re.compile(r"(^|/)models/(legacy/)?hia5_nanopore(_gt_swapped_legacy)?\.json$"),
    "dddb_nanopore": re.compile(r"(^|/)models/(legacy/)?dddb_nanopore(_gt_swapped_legacy)?\.json$"),
}
VERSION_RE = re.compile(r"""__version__\s*=\s*['"]([^'"]+)['"]""")


def git(*args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode()


def _table_rules(index):
    """table name -> fix commit, from the rules that watch that table."""
    out = {}
    for rule in index["advisories"]:
        table = rule.get("match", {}).get("table")
        if table:
            out[table] = rule["fixed_in"]["commit"]
    return out


def _fix_commits(index):
    out = set()
    for rule in index["advisories"]:
        fixed = rule.get("fixed_in", {})
        if fixed.get("commit"):
            out.add(fixed["commit"])
        out.update(fixed.get("by_program", {}).values())
    return sorted(out)


def build(index, through="HEAD"):
    through = git("rev-parse", through).strip()
    commits = git("rev-list", "--topo-order", "--reverse", through).split()
    reachable = set(commits)
    fix_commits = _fix_commits(index)
    containing = {}
    for fix in fix_commits:
        full = git("rev-parse", fix).strip()
        after = set(git("rev-list", "--ancestry-path", f"{full}..{through}").split())
        containing[fix] = after | {full}

    commit_table = {}
    for commit in commits:
        try:
            text = git("show", f"{commit}:fiberhmm/__init__.py")
        except subprocess.CalledProcessError:
            text = ""
        match = VERSION_RE.search(text)
        entry = {"version": match.group(1) if match else None,
                 "contains": [fix for fix in fix_commits if commit in containing[fix]]}
        commit_table[commit] = entry

    table_fix = _table_rules(index)
    blobs = defaultdict(lambda: {"paths": set(), "commits": []})
    for commit in commits:
        for line in git("ls-tree", "-r", commit).splitlines():
            meta, path = line.split("\t", 1)
            for name, pattern in WATCHED.items():
                if pattern.search(path):
                    blob = meta.split()[2]
                    blobs[(name, blob)]["paths"].add(path)
                    blobs[(name, blob)]["commits"].append(commit)
    releases = defaultdict(set)
    for tag in git("tag", "--list", "v*").split():
        if git("rev-parse", f"{tag}^{{commit}}").strip() not in reachable:
            continue  # tagged after the index was generated
        for name in WATCHED:
            for directory in ("fiberhmm/models", "models"):  # 2.0-2.5 shipped models/
                try:
                    blob = git("rev-parse", f"{tag}:{directory}/{name}.json").strip()
                except subprocess.CalledProcessError:
                    continue
                releases[(name, blob)].add(tag.lstrip("v"))
                break

    tables = []
    for (name, blob), info in blobs.items():
        data = git("cat-file", "blob", blob, binary=True)
        fix = table_fix.get(name)
        fixed_commits = containing.get(fix, set())
        before_fix = any(c not in fixed_commits for c in info["commits"])
        legacy_copy = any("_gt_swapped_legacy" in p for p in info["paths"])
        status = "gt_swapped" if (before_fix or legacy_copy) else "fixed"
        tables.append({
            "sha256": hashlib.sha256(data).hexdigest(),
            "table": name,
            "status": status,
            "paths": sorted(info["paths"]),
            "first_commit": info["commits"][0],
            "last_commit": info["commits"][-1],
            "releases": sorted(releases.get((name, blob), ()),
                               key=lambda v: tuple(int(x) for x in re.findall(r"\d+", v))),
        })
    tables.sort(key=lambda t: (t["table"], t["status"], t["first_commit"]))
    merged = dict(index)
    merged["tables"] = tables
    merged["commits"] = commit_table
    merged["commits_through"] = commits[-1]
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if the table digests in the index disagree with git history")
    parser.add_argument("--through", default="HEAD",
                        help="Last commit to index (default HEAD; --check uses the "
                             "index's commits_through)")
    args = parser.parse_args(argv)
    index = json.loads(INDEX.read_text())
    through = index.get("commits_through", args.through) if args.check else args.through
    rebuilt = build(index, through)
    if args.check:
        # Only the table digests are required to match: the commit table
        # legitimately ends at the release it was generated for.
        if rebuilt["tables"] != index.get("tables"):
            print("fiberhmm/advisories.json tables are stale: run "
                  "python tools/build_advisory_index.py", file=sys.stderr)
            return 1
        return 0
    INDEX.write_text(json.dumps(rebuilt, indent=1) + "\n")
    print(f"wrote {INDEX} ({len(rebuilt['tables'])} tables, "
          f"{len(rebuilt['commits'])} commits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
