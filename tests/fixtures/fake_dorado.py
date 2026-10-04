"""A stand-in for ``dorado`` in the fiberhmm-pipeline basecall tests.

``fake_dorado.py --version`` prints a version on stderr (as dorado does);
``fake_dorado.py basecaller MODEL DATA [options]`` writes an unaligned BAM to
stdout with dorado 2.0.1's header records (``@PG ID:basecaller PN:dorado``,
``@RG ... DS:runid=... basecall_model=... [modbase_models=...]``) and
per-read ``RG:Z:<runid>_<model>`` tags. Reads come from the JSON file named by
``FAKE_DORADO_READS`` (``[{"name", "seq", "mm", "ml"}]``; ``mm``/``ml`` are
written only when a modification model was asked for). ``--resume-from``
copies that BAM's records first and skips their reads, as dorado does.

``FAKE_DORADO_FAIL_AFTER=N`` stops with exit 1 after N new reads, leaving a
BAM cut short (no EOF block, the last block truncated) as a killed dorado does;
``FAKE_DORADO_LOG`` gets one JSON line of argv per invocation.
"""
import array
import json
import os
import sys
import tempfile

import pysam

VERSION = "2.0.1+fake"
RUN_ID = "4e3f2a1b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f"
MODELS = {"sup": "dna_r10.4.1_e8.2_400bps_sup@v5.2.0",
          "hac": "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"}


def main(argv):
    log = os.environ.get("FAKE_DORADO_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps(argv) + "\n")
    if argv[:1] == ["--version"]:
        sys.stderr.write(f"[info] dorado\n{VERSION}\n")
        return 0
    if argv[:1] != ["basecaller"] or len(argv) < 3:
        sys.stderr.write("usage: dorado basecaller model data\n")
        return 2
    model_arg, data = argv[1], argv[2]
    opts = argv[3:]
    if not os.path.exists(data):
        sys.stderr.write(f"[error] {data} not found\n")
        return 1
    head, _, complex_mods = model_arg.partition(",")
    model = MODELS.get(head.split("@")[0], os.path.basename(head.rstrip("/")))
    mods = []
    resume = None
    i = 0
    while i < len(opts):
        if opts[i] == "--modified-bases":
            i += 1
            while i < len(opts) and not opts[i].startswith("-"):
                mods.append(f"{model}_{opts[i]}@v1")
                i += 1
            continue
        if opts[i] == "--modified-bases-models":
            mods += [os.path.basename(m) for m in opts[i + 1].split(",")]
            i += 1
        elif opts[i] == "--resume-from":
            resume = opts[i + 1]
            i += 1
        i += 1
    if complex_mods:
        mods.append(f"{model}_{complex_mods}@v1")
    rg_id = f"{RUN_ID}_{model}"
    ds = f"runid={RUN_ID} basecall_model={model}" + (
        f" modbase_models={','.join(mods)}" if mods else "")
    header = {
        "HD": {"VN": "1.6", "SO": "unknown"},
        "RG": [{"ID": rg_id, "PU": "PAW12345", "PM": "P2S-01234", "DT": "2026-09-30T10:00:00Z",
                "PL": "ONT", "DS": ds, "LB": "yw_2_4h", "SM": "yw_2_4h"}],
        "PG": [{"ID": "basecaller", "PN": "dorado", "VN": VERSION.split("+")[0],
                "CL": "dorado " + " ".join(argv)}],
    }
    reads = json.load(open(os.environ["FAKE_DORADO_READS"]))
    fail_after = int(os.environ.get("FAKE_DORADO_FAIL_AFTER", "0") or 0)
    done = set()
    handle, tmp = tempfile.mkstemp(suffix=".bam")
    os.close(handle)
    try:
        code = write(tmp, header, reads, resume, done, mods, rg_id, fail_after)
        data = open(tmp, "rb").read()
    finally:
        os.remove(tmp)
    if code:
        data = data[:-40]  # killed: no EOF block, the last block cut short
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return code


def write(path, header, reads, resume, done, mods, rg_id, fail_after):
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        if resume:
            with pysam.AlignmentFile(resume, check_sq=False) as previous:
                for read in previous.fetch(until_eof=True):
                    done.add(read.query_name)
                    record = pysam.AlignedSegment(out.header)
                    record.query_name = read.query_name
                    record.flag = 4
                    record.query_sequence = read.query_sequence
                    record.query_qualities = read.query_qualities
                    record.set_tags(read.get_tags())
                    out.write(record)
        new = 0
        for item in reads:
            if item["name"] in done:
                continue
            if fail_after and new >= fail_after:
                out.close()
                sys.stderr.write("[error] simulated crash\n")
                return 1
            record = pysam.AlignedSegment(out.header)
            record.query_name = item["name"]
            record.flag = 4
            record.query_sequence = item["seq"]
            record.query_qualities = pysam.qualitystring_to_array("5" * len(item["seq"]))
            tags = [("qs", 14, "i"), ("RG", rg_id, "Z")]
            if mods and item.get("mm"):
                tags += [("MM", item["mm"], "Z"), ("ML", array.array("B", item["ml"]))]
            record.set_tags(tags)
            out.write(record)
            new += 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
