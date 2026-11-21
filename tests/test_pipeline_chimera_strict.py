"""--chimera-detector strict (3.3) through fiberhmm-pipeline: a real minimap2 +
fiberhmm-call run records the strict detector in the called BAM's @PG and in
the chimera sidecar; a default run records the standard detector.
"""
from __future__ import annotations

import json
import shutil

import pysam
import pytest

from test_pipeline import _run_cli, plasmid_run  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(shutil.which("minimap2") is None, reason="needs minimap2")


def _call_pg(bam):
    with pysam.AlignmentFile(str(bam), check_sq=False) as handle:
        return [pg for pg in handle.header.to_dict().get("PG", []) if str(pg.get("ID", "")).startswith("fiberhmm-call")]


def _sidecars(outdir):
    return [p for p in outdir.rglob("*chimera*.json") if "error_mask" not in p.name]


@pytest.mark.parametrize("detector", [None, "strict"])
def test_pipeline_passes_the_chimera_detector_to_call(plasmid_run, detector):  # noqa: F811
    run = plasmid_run
    args = [str(run["reads"]), "--reference", str(run["map"]), "--enzyme", "dddb", "--sample", "s1", "-c", "2",
            "--min-read-length", "500", "--no-qc", "-o", str(run["out"])]
    if detector:
        args += ["--chimera-detector", detector]
    proc = _run_cli(args)
    assert proc.returncode == 0, proc.stderr[-3000:]
    called = run["out"] / "s1.fiberhmm.bam"
    (pg,) = _call_pg(called)
    ds = str(pg.get("DS", ""))
    if detector == "strict":
        assert "chimera_detector=strict" in ds, ds
    else:
        assert "chimera_detector=strict" not in ds, ds
    cars = _sidecars(run["out"])
    assert cars, sorted(str(p.relative_to(run["out"])) for p in run["out"].rglob("*"))[:40]
    detectors = {json.loads(p.read_text())["statistics"].get("detector", "standard") for p in cars}
    assert detectors == {"strict" if detector == "strict" else "standard"}, detectors
    # the run record carries the detector in its call arguments, so a Library re-call repeats it
    call_args = json.loads((run["out"] / "outputs.json").read_text())["settings"]["call_args"]
    assert ("--chimera-detector" in call_args) == (detector == "strict"), call_args
