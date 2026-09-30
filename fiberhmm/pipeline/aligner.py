"""minimap2 for ``fiberhmm-pipeline``: discovery, cached indexes and read input.

The ``minimap2`` binary on ``PATH`` is used when present, otherwise the
``mappy`` Python module (``pip install mappy``). Alignment options follow the
lab's DAF-seq standard, ``minimap2 -a -x map-ont --MD -Y`` (``-Y`` keeps the
full read sequence on supplementary records, soft-clipped).

Indexes are cached under ``~/.fiberhmm/minimap2_index/<digest>.mmi`` (override
with ``FIBERHMM_MINIMAP2_INDEX_DIR``); the digest covers the reference FASTA's
contents, the preset and the index format, so a later run on the same genome
starts aligning at once.
"""
from __future__ import annotations

import array
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import pysam

INSTALL_HELP = """\
fiberhmm-pipeline needs minimap2 to align reads, and found neither the
`minimap2` program on PATH nor the `mappy` Python module. Install one:

  macOS (Homebrew):   brew install minimap2
  conda / mamba:      conda install -c bioconda minimap2
  Debian / Ubuntu:    sudo apt install minimap2
  Python module:      pip install mappy      (no separate program needed)

Then run the command again. Reads that are already aligned (a BAM aligned to
the same reference, with MD tags) do not need minimap2."""


class AlignerNotFound(RuntimeError):
    pass


def index_cache_dir() -> str:
    env = os.environ.get("FIBERHMM_MINIMAP2_INDEX_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".fiberhmm", "minimap2_index")


PRESET_FOR_PLATFORM = {"nanopore": "map-ont", "pacbio": "map-hifi"}


@dataclass
class Aligner:
    kind: str  # "minimap2" or "mappy"
    path: Optional[str]
    version: str

    @property
    def index_format(self) -> str:
        # minimap2 index files are versioned by the program's major.minor.
        match = re.match(r"(\d+\.\d+)", self.version or "")
        return f"mm2-{match.group(1) if match else self.version}"

    def describe(self) -> str:
        where = f" ({self.path})" if self.path else ""
        return f"{self.kind} {self.version}{where}"


def find_aligner(prefer: str = "auto") -> Aligner:
    """Locate minimap2. ``prefer``: ``auto``, ``minimap2`` or ``mappy``."""
    if prefer in ("auto", "minimap2"):
        path = shutil.which("minimap2")
        if path:
            try:
                version = subprocess.run([path, "--version"], capture_output=True,
                                         text=True, timeout=30).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                version = ""
            if version:
                return Aligner("minimap2", path, version)
        if prefer == "minimap2":
            raise AlignerNotFound(INSTALL_HELP)
    if prefer in ("auto", "mappy"):
        try:
            import mappy  # noqa: F401
            version = getattr(mappy, "__version__", "") or "unknown"
            return Aligner("mappy", None, version)
        except ImportError:
            pass
    raise AlignerNotFound(INSTALL_HELP)


def index_digest(reference_sha256: str, preset: str, aligner: Aligner) -> str:
    text = f"{reference_sha256}|{preset}|{aligner.index_format}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def ensure_index(fasta: str, reference_sha256: str, preset: str, aligner: Aligner,
                 threads: int = 4, log=None) -> tuple[str, bool]:
    """Return ``(index path, built_now)``; build and cache the index if needed."""
    cache = index_cache_dir()
    os.makedirs(cache, exist_ok=True)
    digest = index_digest(reference_sha256, preset, aligner)
    target = os.path.join(cache, f"{digest}.mmi")
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target, False
    tmp = f"{target}.tmp{os.getpid()}"
    try:
        if aligner.kind == "minimap2":
            cmd = [aligner.path, "-x", preset, "-t", str(max(1, threads)), "-d", tmp, fasta]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
            if log is not None and result.stderr:
                log.write(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(
                    f"minimap2 index build failed ({result.returncode}): "
                    f"{result.stderr.strip()[-2000:]}")
        else:
            import mappy
            mappy.Aligner(fasta, preset=preset, n_threads=max(1, threads), fn_idx_out=tmp)
            if not os.path.exists(tmp):
                raise RuntimeError("mappy did not write the index")
        os.replace(tmp, target)
        with open(target + ".json", "w", encoding="utf-8") as handle:
            handle.write(
                '{"reference": %s, "reference_sha256": "%s", "preset": "%s", '
                '"aligner": "%s"}\n' % (_json_str(os.path.abspath(fasta)),
                                        reference_sha256, preset, aligner.describe()))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target, True


def _json_str(value: str) -> str:
    import json
    return json.dumps(value)


# ---------------------------------------------------------------------------
# Read input
# ---------------------------------------------------------------------------

FASTQ_EXTENSIONS = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
_SAM_TAG = re.compile(r"^[A-Za-z][A-Za-z0-9]:[AifZHB]:\S*$")


@dataclass
class ReadFile:
    path: str
    kind: str  # "fastq", "ubam" or "aligned"
    size: int


def classify_read_file(path: str) -> ReadFile:
    lower = path.lower()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"read file not found: {path}")
    size = os.path.getsize(path)
    if lower.endswith(FASTQ_EXTENSIONS):
        return ReadFile(path, "fastq", size)
    if lower.endswith((".bam", ".cram", ".sam")):
        with pysam.AlignmentFile(path, check_sq=False) as bam:
            has_sq = bool(bam.header.to_dict().get("SQ"))
            if not has_sq:
                return ReadFile(path, "ubam", size)
            for read in bam.fetch(until_eof=True):
                if not read.is_unmapped:
                    return ReadFile(path, "aligned", size)
            return ReadFile(path, "ubam", size)
    raise ValueError(
        f"{path}: unrecognised read file (expected FASTQ {', '.join(FASTQ_EXTENSIONS)} "
        "or BAM)")


def _open_binary(path: str):
    return gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")


def fastq_has_sam_tags(path: str, records: int = 50) -> bool:
    """True when FASTQ header comments are SAM tags (``MM:Z:... ML:B:C,...``).

    minimap2 ``-y`` copies them to the alignment, which is how base
    modification calls in a FASTQ reach the BAM. Plain comments (``runid=``)
    must not be copied, since they are not valid SAM.
    """
    seen = 0
    with _open_binary(path) as handle:
        for i, line in enumerate(handle):
            if i % 4 != 0:
                continue
            fields = line.decode("utf-8", "replace").rstrip("\r\n").split(None, 1)
            if len(fields) < 2:
                return False
            if not all(_SAM_TAG.match(token) for token in fields[1].split()):
                return False
            seen += 1
            if seen >= records:
                break
    return seen > 0


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv",
                                       "TGCANtgcanYRMKSWVHDByrmkswvhdb"))[::-1]


_KEEP_BAM_TAGS = ("MM", "ML", "Mm", "Ml", "MN")
# SAM B-array subtypes -> Python array type codes.
_B_ARRAY_CODES = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "f": "f"}


def bam_records_as_fastq(path: str, with_tags: bool) -> Iterator[bytes]:
    """Primary reads of a BAM as FASTQ, in sequencing orientation.

    Base-modification tags are carried as SAM-tag comments (for minimap2
    ``-y``). Hard-clipped primaries cannot be restored and are skipped.
    """
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_secondary or read.is_supplementary:
                continue
            seq = read.query_sequence
            if not seq:
                continue
            if read.cigartuples and any(op == 5 for op, _ in read.cigartuples):
                continue
            quals = read.query_qualities
            if read.is_reverse:
                seq = _revcomp(seq)
                quals = quals[::-1] if quals is not None else None
            qual = ("".join(chr(q + 33) for q in quals) if quals is not None
                    else "I" * len(seq))
            comment = ""
            if with_tags:
                parts = []
                for tag in _KEEP_BAM_TAGS:
                    if read.has_tag(tag):
                        value = read.get_tag(tag)
                        if tag in ("ML", "Ml"):
                            parts.append(f"{tag}:B:C," + ",".join(str(int(v)) for v in value))
                        elif tag == "MN":
                            parts.append(f"MN:i:{int(value)}")
                        else:
                            parts.append(f"{tag}:Z:{value}")
                if parts:
                    comment = "\t" + "\t".join(parts)
            yield f"@{read.query_name}{comment}\n{seq}\n+\n{qual}\n".encode()


def fastq_platform_votes(path: str, records: int = 200) -> dict:
    """``{"pacbio": n, "nanopore": n}`` from the MM tags in FASTQ header comments.

    PacBio Fiber-seq reports bottom-strand m6A as ``T-a``; Nanopore reports
    only ``A+a`` (the rule of ``fiberhmm-call``'s platform detection).
    """
    from fiberhmm.cli.common import _mm_spec_platform
    counts = {"pacbio": 0, "nanopore": 0}
    with _open_binary(path) as handle:
        for i, line in enumerate(handle):
            if i // 4 >= records:
                break
            if i % 4 != 0:
                continue
            for token in line.decode("utf-8", "replace").split()[1:]:
                if token.startswith(("MM:Z:", "Mm:Z:")):
                    platform = _mm_spec_platform(token[5:])
                    if platform:
                        counts[platform] += 1
                    break
    return counts


def bam_has_mod_tags(path: str, records: int = 200) -> bool:
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if read.has_tag("MM") or read.has_tag("Mm"):
                return True
            if i >= records:
                break
    return False


# Internal read names: "<serial in hex>~<original name>". Every input record
# gets its own serial, so records that share a name (the same read name in two
# input files, or twice in one) stay separate molecules through alignment;
# original_name() restores the name on the output record.
INTERNAL_NAME_SEPARATOR = "~"


def original_name(internal: str) -> str:
    """The input read name behind an internal name (unchanged if it is not one)."""
    serial, sep, name = internal.partition(INTERNAL_NAME_SEPARATOR)
    if sep and serial and all(c in "0123456789abcdef" for c in serial):
        return name
    return internal


class ReadFeeder:
    """Stream several read files as one FASTQ (minimap2 would treat two query
    files as read pairs). Tracks bytes consumed for progress.

    Each record's name is prefixed with its serial number (see
    :func:`original_name`), so alignment records group by input record, never
    by a name two molecules happen to share."""

    def __init__(self, files: list[ReadFile], carry_tags: bool):
        self.files = files
        self.carry_tags = carry_tags
        self.total_bytes = sum(f.size for f in files) or 1
        self.done_bytes = 0
        self.reads = 0
        self.error: Optional[BaseException] = None

    def _name(self, header: bytes) -> bytes:
        # header is b"@name[ comment]..."; the serial goes in front of the name.
        serial = b"%x" % self.reads + INTERNAL_NAME_SEPARATOR.encode()
        self.reads += 1
        return b"@" + serial + header[1:]

    def chunks(self) -> Iterator[bytes]:
        for rf in self.files:
            start = self.done_bytes
            if rf.kind == "fastq":
                with open(rf.path, "rb") as raw:
                    handle = gzip.GzipFile(fileobj=raw) if rf.path.lower().endswith(".gz") else raw
                    batch: list[bytes] = []
                    size = 0
                    line_no = 0
                    last = b"\n"
                    for line in handle:
                        if line_no % 4 == 0 and line.startswith(b"@"):
                            line = self._name(line)
                        elif line_no % 4 == 0 and line.strip():
                            raise ValueError(f"{rf.path}: malformed FASTQ record "
                                             f"(line {line_no + 1} does not start with '@')")
                        elif line_no % 4 == 0:
                            continue  # blank line between records
                        line_no += 1
                        batch.append(line)
                        size += len(line)
                        last = line[-1:]
                        if size >= 1 << 20:
                            self.done_bytes = start + raw.tell()
                            yield b"".join(batch)
                            batch, size = [], 0
                    if batch:
                        yield b"".join(batch)
                    if last != b"\n":
                        yield b"\n"
            else:
                for record in bam_records_as_fastq(rf.path, self.carry_tags):
                    yield self._name(record)
            self.done_bytes = start + rf.size

    def fraction(self) -> float:
        return min(1.0, self.done_bytes / self.total_bytes)


def iter_fastq_entries(chunks: Iterable[bytes]) -> Iterator[tuple[str, str, str, str]]:
    """Parse FASTQ bytes into ``(name, comment, seq, qual)`` (for mappy)."""
    buffer = b""
    lines: list[bytes] = []
    for chunk in chunks:
        buffer += chunk
        *complete, buffer = buffer.split(b"\n")
        lines.extend(complete)
        while len(lines) >= 4:
            head, seq, _plus, qual = lines[:4]
            del lines[:4]
            text = head.decode("utf-8", "replace").rstrip("\r")[1:]
            parts = text.split(None, 1)
            yield (parts[0], parts[1] if len(parts) > 1 else "",
                   seq.decode().strip(), qual.decode().strip())
    if buffer:
        lines.append(buffer)
    lines = [line for line in lines if line.strip()]
    while len(lines) >= 4:
        head, seq, _plus, qual = lines[:4]
        del lines[:4]
        text = head.decode("utf-8", "replace").rstrip("\r")[1:]
        parts = text.split(None, 1)
        yield (parts[0], parts[1] if len(parts) > 1 else "",
               seq.decode().strip(), qual.decode().strip())


# ---------------------------------------------------------------------------
# Alignment streams
# ---------------------------------------------------------------------------

def minimap2_command(aligner: Aligner, index: str, preset: str, threads: int,
                     read_group: str, copy_tags: bool) -> list[str]:
    cmd = [aligner.path, "-a", "-x", preset, "--MD", "-Y", "-t", str(max(1, threads)),
           "-R", read_group]
    if copy_tags:
        cmd.append("-y")
    cmd += [index, "-"]
    return cmd


class Minimap2Stream:
    """Run the minimap2 binary on a :class:`ReadFeeder`; iterate SAM records."""

    def __init__(self, cmd: list[str], feeder: ReadFeeder, stderr_path: str):
        self.cmd = cmd
        self.feeder = feeder
        self.stderr_handle = open(stderr_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr_handle)
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()
        self.bam = pysam.AlignmentFile(self.proc.stdout, "r")

    def _feed(self) -> None:
        try:
            for chunk in self.feeder.chunks():
                self.proc.stdin.write(chunk)
        except BrokenPipeError:
            pass
        except BaseException as exc:  # surfaced by close()
            self.feeder.error = exc
        finally:
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    @property
    def header(self):
        return self.bam.header

    def __iter__(self):
        return iter(self.bam)

    def close(self) -> None:
        try:
            self.bam.close()
        finally:
            code = self.proc.wait()
            self.thread.join(timeout=5)
            self.stderr_handle.close()
        if self.feeder.error is not None:
            raise RuntimeError(f"reading the input failed: {self.feeder.error}")
        if code != 0:
            raise RuntimeError(f"minimap2 exited with status {code}; see the log")


class MappyStream:
    """Align with the ``mappy`` module and build SAM records like minimap2 -a -Y."""

    def __init__(self, index: str, preset: str, threads: int, feeder: ReadFeeder,
                 header: dict, copy_tags: bool, read_group_id: str):
        import mappy
        self.aligner = mappy.Aligner(fn_idx_in=index, preset=preset,
                                     n_threads=max(1, threads))
        if not self.aligner:
            raise RuntimeError(f"mappy could not load the index {index}")
        self.feeder = feeder
        self.copy_tags = copy_tags
        self.read_group_id = read_group_id
        self._header = pysam.AlignmentHeader.from_dict(header)
        self._tid = {name: i for i, name in enumerate(self._header.references)}

    @property
    def header(self):
        return self._header

    def __iter__(self):
        for name, comment, seq, qual in iter_fastq_entries(self.feeder.chunks()):
            yield from self._records(name, comment, seq, qual)

    def _tags(self, comment: str) -> list:
        tags = []
        if self.copy_tags and comment:
            for token in comment.split():
                if not _SAM_TAG.match(token):
                    continue
                tag, kind, value = token.split(":", 2)
                if kind == "i":
                    tags.append((tag, int(value), "i"))
                elif kind == "B":
                    subtype, *values = value.split(",")
                    code = _B_ARRAY_CODES.get(subtype)
                    if code is None:
                        continue
                    cast = float if code == "f" else int
                    tags.append((tag, array.array(code, [cast(v) for v in values if v])))
                elif kind == "f":
                    tags.append((tag, float(value), "f"))
                else:
                    tags.append((tag, value, "Z"))
        return tags

    def _records(self, name, comment, seq, qual):
        quals = pysam.qualitystring_to_array(qual) if qual else None
        extra = self._tags(comment)
        hits = [hit for hit in self.aligner.map(seq, MD=True) if hit.is_primary]
        if not hits:
            read = pysam.AlignedSegment(self._header)
            read.query_name = name
            read.flag = 4
            read.query_sequence = seq
            read.query_qualities = quals
            read.set_tags(extra + [("RG", self.read_group_id, "Z")])
            yield read
            return
        # mappy reports the primary first; other non-secondary hits are
        # supplementary (secondary hits have is_primary False and are dropped).
        for i, hit in enumerate(hits):
            read = pysam.AlignedSegment(self._header)
            read.query_name = name
            flag = 0x800 if i else 0
            if hit.strand < 0:
                flag |= 0x10
            read.flag = flag
            read.reference_id = self._tid[hit.ctg]
            read.reference_start = hit.r_st
            read.mapping_quality = hit.mapq
            forward = hit.strand > 0
            record_seq = seq if forward else _revcomp(seq)
            record_quals = quals if (forward or quals is None) else quals[::-1]
            lead = hit.q_st if forward else len(seq) - hit.q_en
            trail = len(seq) - hit.q_en if forward else hit.q_st
            ops = [(op, n) for n, op in hit.cigar]
            cigar = ([(4, lead)] if lead else []) + ops + ([(4, trail)] if trail else [])
            read.query_sequence = record_seq
            read.query_qualities = record_quals
            read.cigartuples = cigar
            tags = [("NM", hit.NM, "i"), ("MD", hit.MD, "Z"), ("RG", self.read_group_id, "Z")]
            read.set_tags(tags + extra)
            yield read

    def close(self) -> None:
        if self.feeder.error is not None:
            raise RuntimeError(f"reading the input failed: {self.feeder.error}")


def mappy_header(contigs, read_group: dict, program: dict) -> dict:
    return {
        "HD": {"VN": "1.6", "SO": "unsorted", "GO": "query"},
        "SQ": [{"SN": c.name, "LN": c.length} for c in contigs],
        "RG": [read_group],
        "PG": [program],
    }


def progress_ticker(interval: float, callback: Callable[[], None]):
    """Call ``callback`` every ``interval`` seconds until the returned event is set."""
    stop = threading.Event()

    def run():
        while not stop.wait(interval):
            try:
                callback()
            except Exception:
                pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop

