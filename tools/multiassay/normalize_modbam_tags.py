#!/usr/bin/env python3
"""
normalize_modbam_tags.py — standardize base-modification tags in a modBAM.

Different basecallers/pipelines emit non-standard MM/ML modification tags:
  - Old lowercase tag names `Mm`/`Ml` instead of the spec `MM`/`ML`.
  - Non-standard single-letter mod codes, e.g. Megalodon's `A+Y` (6mA) and
    `C+Z` (5mC) instead of the SAMtags/ChEBI standard `A+a` (6mA), `C+m` (5mC).

FiberHMM (and most modern tooling) expects standard codes: pysam.modified_bases
recognises `a` (6mA, ChEBI 21839) and `m` (5mC, ChEBI 27551). This utility rewrites
tags in place so any modBAM feeds "relatively simply" into FiberHMM.

Default remap (Megalodon -> standard):  Y->a (6mA),  Z->m (5mC)
Override with --code-map "Y:a,Z:m,...".

Only the MM header codes are rewritten; ML probabilities and MM delta lists are
untouched (they are code-agnostic). Everything else in the record is preserved.

Usage:
  normalize_modbam_tags.py -i in.bam -o out.bam [--code-map Y:a,Z:m] [--threads N]
"""
import argparse
import array
import sys

import pysam


def parse_code_map(s: str) -> dict:
    m = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        old, new = pair.split(":")
        m[old.strip()] = new.strip()
    return m


def remap_mm(mm: str, code_map: dict, stats: dict) -> str:
    """Rewrite mod codes in an MM tag string.

    MM grammar per block: <base><strand><codes>[.?] , delta , delta ... ;
    e.g. "A+Y,12,0,3;C+Zm?,1,2;" — codes are the run of code chars after the
    strand, terminated by an optional '.'/'?' and then a ','.
    """
    out_blocks = []
    for block in mm.split(";"):
        if not block:
            out_blocks.append(block)
            continue
        comma = block.find(",")
        header = block if comma < 0 else block[:comma]
        rest = "" if comma < 0 else block[comma:]
        if len(header) < 2:
            out_blocks.append(block)
            continue
        base = header[0]
        strand = header[1]
        codes = header[2:]
        # strip a trailing status flag ('.' or '?') off the codes
        flag = ""
        if codes and codes[-1] in ".?":
            flag = codes[-1]
            codes = codes[:-1]
        new_codes = []
        for c in codes:
            if c in code_map:
                stats[c] = stats.get(c, 0) + 1
                new_codes.append(code_map[c])
            else:
                new_codes.append(c)
        out_blocks.append(base + strand + "".join(new_codes) + flag + rest)
    return ";".join(out_blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--code-map", default="Y:a,Z:m",
                    help="Comma list old:new mod-code remaps (default Megalodon Y:a,Z:m)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    code_map = parse_code_map(args.code_map)
    stats = {}
    n = 0

    with pysam.AlignmentFile(args.input, "rb", check_sq=False, threads=args.threads) as bam_in:
        with pysam.AlignmentFile(args.output, "wb", template=bam_in, threads=args.threads) as bam_out:
            for rec in bam_in:
                # normalize tag names Mm/Ml -> MM/ML
                mm = ml = None
                for name in ("MM", "Mm"):
                    if rec.has_tag(name):
                        mm = rec.get_tag(name)
                        break
                for name in ("ML", "Ml"):
                    if rec.has_tag(name):
                        ml = rec.get_tag(name)
                        break
                if mm is not None:
                    new_mm = remap_mm(mm, code_map, stats)
                    # drop legacy lowercase tags, write standard ones
                    for name in ("Mm", "Ml", "MM", "ML"):
                        try:
                            rec.set_tag(name, None)
                        except Exception:
                            pass
                    rec.set_tag("MM", new_mm, "Z")
                    if ml is not None:
                        # ML is a B,C array tag; pass an array so pysam encodes it
                        # as B (not a scalar). get_tag returns array('B',...)/list.
                        rec.set_tag("ML", array.array("B", ml))
                bam_out.write(rec)
                n += 1
                if n % 200000 == 0:
                    print(f"  {n:,} records...", file=sys.stderr)

    print(f"Done. {n:,} records written to {args.output}", file=sys.stderr)
    if stats:
        print("Remapped mod codes: " +
              ", ".join(f"{k}->{code_map[k]} x{v:,}" for k, v in sorted(stats.items())),
              file=sys.stderr)
    else:
        print("No codes remapped (tags may already be standard).", file=sys.stderr)


if __name__ == "__main__":
    main()
