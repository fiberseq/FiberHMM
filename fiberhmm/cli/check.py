"""fiberhmm-check: does a FiberHMM output need re-running with this version?

Reads what an output records about how it was made (BAM @PG/@CO provenance,
QC report, posteriors metadata, consensus manifest) and lists the known
FiberHMM fixes and default changes that apply to it (fiberhmm/advisories.json).

Exit status: 0 when no output needs re-running (clean, or only info-level
default changes), 3 when at least one output has a re-run advisory (affected or
possibly affected), 2 when a path could not be checked.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap

EXIT_CLEAN = 0
EXIT_ERROR = 2
EXIT_ADVISORIES = 3

_LABELS = {
    "clean": "clean",
    "info": "no re-run needed (default changes only)",
    "rerun-recommended": "RE-RUN RECOMMENDED",
    "rerun-required": "RE-RUN REQUIRED",
    "error": "ERROR",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fiberhmm-check",
        description=(
            "List the FiberHMM fixes and default changes that apply to existing "
            "outputs (BAMs, QC reports, posteriors files, consensus result "
            "directories), from the provenance they record. Exit status: 0 no "
            "re-run needed, 3 at least one re-run advisory, 2 a path could not be "
            "checked."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help="BAM/CRAM/SAM, <prefix>.qc.json, posteriors .tsv(.gz)/.h5, "
                             "or a fiberhmm-consensus output directory")
    parser.add_argument("--json", action="store_true",
                        help="Print one JSON document (schema fiberhmm.advisory_check.v1) "
                             "with a fiberhmm.advisory_report.v1 report per path")
    parser.add_argument("--scan-records", type=int, default=None, metavar="N",
                        help="BAM records to scan for read-level evidence (duplicate, "
                             "pairing and call tags); 0 reads the header only "
                             "(default: 5000)")
    parser.add_argument("--no-sidecars", action="store_true",
                        help="Do not also check the QC report fiberhmm-call writes beside "
                             "a BAM (qc/<name>.qc.json)")
    parser.add_argument("--list", action="store_true",
                        help="List every known advisory and exit")
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    return parser


def _print_report(report: dict, out) -> None:
    status = report["status"]
    label = _LABELS.get(status, status)
    if report.get("needs_rerun") and not report.get("confirmed"):
        label += " (possibly affected)"
    print(f"{report['path']}: {label}", file=out)
    if status == "error":
        print(f"  {report['error']}", file=out)
        return
    wrap = textwrap.TextWrapper(width=100, initial_indent="      ",
                                subsequent_indent="      ")
    for advisory in report["advisories"]:
        where = f" [{advisory['path']}]" if advisory.get("path") not in (None, report["path"]) else ""
        print(f"  - {advisory['id']} ({advisory['severity']}; "
              f"{advisory['status'].replace('_', ' ')}, {advisory['confidence']} confidence)"
              f"{where}", file=out)
        print(f"    {advisory['title']} [{advisory['artifact']}]", file=out)
        print("    evidence:", file=out)
        for line in advisory["evidence"]:
            print(wrap.fill(line), file=out)
        print("    why:", file=out)
        print(wrap.fill(advisory["reason"]), file=out)
        print("    fix:", file=out)
        print(wrap.fill(advisory["fix"]), file=out)


def _list_advisories(as_json: bool) -> int:
    from fiberhmm.advisories import load_index

    index = load_index()
    if as_json:
        keys = ("id", "severity", "artifact", "title", "reason", "fix", "fixed_in", "changelog")
        print(json.dumps([{k: rule.get(k) for k in keys} for rule in index["advisories"]],
                         indent=2))
        return EXIT_CLEAN
    for rule in index["advisories"]:
        print(f"{rule['id']} ({rule['severity']}, {rule['artifact']}): {rule['title']}")
    return EXIT_CLEAN


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        return _list_advisories(args.json)
    if not args.paths:
        parser.error("give at least one PATH (or --list)")
    if args.scan_records is not None and args.scan_records < 0:
        parser.error("--scan-records must be >= 0")

    from fiberhmm.advisories import DEFAULT_SCAN_RECORDS, report

    scan = DEFAULT_SCAN_RECORDS if args.scan_records is None else args.scan_records
    reports = [report(path, scan_records=scan, sidecars=not args.no_sidecars)
               for path in args.paths]
    if args.json:
        print(json.dumps({"schema": "fiberhmm.advisory_check.v1", "reports": reports},
                         indent=2))
    else:
        for item in reports:
            _print_report(item, sys.stdout)
    if any(item["status"] == "error" for item in reports):
        return EXIT_ERROR
    if any(item["needs_rerun"] for item in reports):
        return EXIT_ADVISORIES
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
