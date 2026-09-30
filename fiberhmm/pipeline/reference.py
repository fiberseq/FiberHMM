"""Reference preparation for ``fiberhmm-pipeline``.

A reference is either a FASTA file (used as is, contig names unchanged) or a
plasmid map (SnapGene ``.dna``, GenBank ``.gb/.gbk/.genbank``, EMBL
``.embl/.emb``) that is converted to a one-contig FASTA named exactly as
FiberBrowser names the map when it loads it (:func:`fiberbrowser_contig_name`),
so the BAM and the map line up in the browser without renaming anything.

The prepared reference also carries what FiberBrowser needs to match a BAM to a
map robustly: the MD5 of every contig (written as ``@SQ M5``), its topology
(``@SQ TP:circular`` for circular contigs) and, for plasmid maps and circular
contigs, one ``@CO FIBERHMM-REFERENCE:v1:`` line (:func:`reference_comment`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote, unquote

MAP_FORMATS = {
    ".dna": "snapgene",
    ".gb": "genbank",
    ".gbk": "genbank",
    ".genbank": "genbank",
    ".embl": "embl",
    ".emb": "embl",
}
FASTA_EXTENSIONS = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")

REFERENCE_COMMENT_PREFIX = "FIBERHMM-REFERENCE:v1:"

# A FASTA reference no larger than this is copied into the output directory so
# a plasmid run's folder is self-contained; larger genomes are referenced by
# path (copying dm6 or hg38 into every run directory would waste disk).
COPY_FASTA_MAX_BYTES = 20 * 1024 * 1024

_SAFE_CHARS = re.compile(r"[^0-9A-Za-z_.-]+")


def _safe_identifier(value: str, fallback: str = "plasmid") -> str:
    stem = str(value or fallback).split()[0] if str(value or "").split() else fallback
    cleaned = _SAFE_CHARS.sub("_", stem).strip("_")
    return cleaned or fallback


def fiberbrowser_contig_name(path: str, declared_name: Optional[str] = None,
                             source_format: Optional[str] = None) -> str:
    """Return the contig name FiberBrowser gives the plasmid map at ``path``.

    This mirrors ``browser/services/plasmid_maps.py`` in FiberBrowser 3.0, so a
    BAM aligned to the FASTA written by ``fiberhmm-pipeline`` names its contig
    exactly as the browser names the map:

    * SnapGene ``.dna``: the file name without its extension, cut at the first
      whitespace, every run of characters outside ``[0-9A-Za-z_.-]`` replaced
      by ``_`` and leading/trailing ``_`` removed (``"L-HH (v2).dna"`` ->
      ``"L-HH"``; an empty result becomes ``"plasmid"``).
    * GenBank: the ``LOCUS`` name with every run of characters outside
      ``[0-9A-Za-z_.-]`` replaced by ``_`` (no stripping), or the file-name
      rule when there is no ``LOCUS`` name.
    * EMBL: the ``ID`` name, same rule as GenBank.
    * FASTA map: the first header token, file-name sanitising rule.

    ``declared_name`` is the name stored inside the file (LOCUS, ID or FASTA
    header), when the format has one.
    """
    stem = os.path.splitext(os.path.basename(path))[0] or "plasmid"
    filename_name = _safe_identifier(stem)
    fmt = source_format or MAP_FORMATS.get(os.path.splitext(path)[1].lower(), "fasta")
    if fmt in ("genbank", "embl"):
        if declared_name:
            return _SAFE_CHARS.sub("_", declared_name) or filename_name
        return filename_name
    if fmt == "fasta":
        return _safe_identifier(declared_name or filename_name)
    return filename_name


@dataclass
class PlasmidSequence:
    name: str
    sequence: str
    circular: bool
    source_format: str
    warnings: list[str] = field(default_factory=list)


def _snapgene_packets(data: bytes) -> Iterator[tuple[int, bytes]]:
    pos = 0
    while pos + 5 <= len(data):
        kind = data[pos]
        length = int.from_bytes(data[pos + 1:pos + 5], "big")
        if pos + 5 + length > len(data):
            return
        yield kind, data[pos + 5:pos + 5 + length]
        pos += 5 + length


def read_snapgene(path: str) -> PlasmidSequence:
    """Read a SnapGene ``.dna`` file.

    The DNA packet (type 0: one topology byte, bit 0 = circular, then the
    bases) is read when the file has the SnapGene cookie. FiberBrowser takes
    the longest run of ``ACGTN`` instead; the two agree on every map with an
    unambiguous sequence, and a warning is recorded when they do not.
    """
    data = Path(path).read_bytes()
    runs = re.findall(rb"[ACGTNacgtn]{50,}", data)
    browser_sequence = max(runs, key=len).decode("ascii").upper() if runs else ""
    sequence = ""
    circular = True
    if data[:1] == b"\x09" and data[5:13] == b"SnapGene":
        for kind, payload in _snapgene_packets(data):
            if kind == 0 and len(payload) > 1:
                circular = bool(payload[0] & 0x01)
                sequence = re.sub(rb"\s+", b"", payload[1:]).decode("ascii", "replace").upper()
                break
    warnings: list[str] = []
    if not sequence:
        if not browser_sequence:
            raise ValueError(f"no DNA sequence found in SnapGene file {path}")
        sequence = browser_sequence
        warnings.append("SnapGene DNA packet not found; used the longest ACGTN run")
    elif sequence != browser_sequence:
        warnings.append(
            "the SnapGene sequence contains characters other than ACGTN; "
            "FiberBrowser's map import may read a different sequence")
    return PlasmidSequence(fiberbrowser_contig_name(path, source_format="snapgene"),
                           sequence, circular, "snapgene", warnings)


def read_genbank(path: str) -> PlasmidSequence:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    locus = re.search(r"^LOCUS\s+(\S+).*", text, re.MULTILINE | re.IGNORECASE)
    circular = True
    declared = None
    if locus:
        declared = locus.group(1)
        circular = "linear" not in locus.group(0).lower()
    parts = text.split("ORIGIN", 1)
    if len(parts) < 2:
        raise ValueError(f"no ORIGIN sequence block in GenBank file {path}")
    sequence = re.sub(r"[^ACGTNacgtn]", "", parts[1].split("//", 1)[0]).upper()
    if not sequence:
        raise ValueError(f"empty ORIGIN sequence in GenBank file {path}")
    return PlasmidSequence(fiberbrowser_contig_name(path, declared, "genbank"),
                           sequence, circular, "genbank")


def read_embl(path: str) -> PlasmidSequence:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    ident = re.search(r"^ID\s+([^;\s]+).*", text, re.MULTILINE | re.IGNORECASE)
    circular = True
    declared = None
    if ident:
        declared = ident.group(1)
        circular = "linear" not in ident.group(0).lower()
    block = re.search(r"^SQ\s+.*?$([\s\S]*?)^//", text, re.MULTILINE)
    if not block:
        raise ValueError(f"no SQ sequence block in EMBL file {path}")
    sequence = re.sub(r"[^ACGTNacgtn]", "", block.group(1)).upper()
    if not sequence:
        raise ValueError(f"empty SQ sequence in EMBL file {path}")
    return PlasmidSequence(fiberbrowser_contig_name(path, declared, "embl"),
                           sequence, circular, "embl")


def read_plasmid_map(path: str) -> PlasmidSequence:
    fmt = map_format(path)
    if fmt == "snapgene":
        return read_snapgene(path)
    if fmt == "genbank":
        return read_genbank(path)
    if fmt == "embl":
        return read_embl(path)
    raise ValueError(
        f"{path}: not a plasmid map (expected {', '.join(sorted(MAP_FORMATS))})")


def map_format(path: str) -> Optional[str]:
    return MAP_FORMATS.get(os.path.splitext(str(path))[1].lower())


def is_fasta(path: str) -> bool:
    return str(path).lower().endswith(FASTA_EXTENSIONS)


def sequence_md5(sequence: str) -> str:
    """SAM ``@SQ M5``: MD5 of the upper-case sequence without whitespace."""
    return hashlib.md5(re.sub(r"\s+", "", sequence).upper().encode("ascii")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fasta(path: str, name: str, sequence: str, width: int = 60) -> None:
    """Write a one-record FASTA and its ``.fai`` atomically."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f">{name}\n")
        for i in range(0, len(sequence), width):
            handle.write(sequence[i:i + width] + "\n")
    os.replace(tmp, path)
    offset = len(f">{name}\n".encode("ascii"))
    with open(f"{path}.fai.tmp{os.getpid()}", "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{name}\t{len(sequence)}\t{offset}\t{width}\t{width + 1}\n")
    os.replace(f"{path}.fai.tmp{os.getpid()}", f"{path}.fai")


@dataclass
class Contig:
    name: str
    length: int
    md5: str
    circular: bool = False


@dataclass
class ReferenceInfo:
    """A prepared reference: the FASTA to align to and what describes it."""
    fasta: str
    contigs: list[Contig]
    source: str
    source_format: str  # fasta | snapgene | genbank | embl
    source_sha256: str
    plasmid_map: Optional[str] = None  # the copy in the output directory
    warnings: list[str] = field(default_factory=list)

    @property
    def is_plasmid(self) -> bool:
        return self.source_format != "fasta"

    @property
    def circular_contigs(self) -> dict[str, int]:
        return {c.name: c.length for c in self.contigs if c.circular}

    def to_json(self) -> dict:
        return {
            "fasta": self.fasta,
            "source": self.source,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "plasmid_map": self.plasmid_map,
            "contigs": [c.__dict__ for c in self.contigs],
            "warnings": list(self.warnings),
        }


def _open_text(path: str):
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="ascii", errors="replace")
    return open(path, "rt", encoding="ascii", errors="replace")


def scan_fasta(path: str) -> list[tuple[str, int, str]]:
    """Return ``(name, length, md5)`` for every record of a FASTA file."""
    records: list[tuple[str, int, str]] = []
    name = None
    digest = None
    length = 0
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    records.append((name, length, digest.hexdigest()))
                header = line[1:].strip()
                name = header.split()[0] if header else ""
                digest = hashlib.md5()
                length = 0
                continue
            if name is None:
                continue
            chunk = "".join(line.split()).upper()
            digest.update(chunk.encode("ascii"))
            length += len(chunk)
    if name is not None:
        records.append((name, length, digest.hexdigest()))
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def _digest_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "reference_digests.json")


def cached_fasta_digests(path: str, cache_dir: Optional[str]) -> tuple[str, list[tuple[str, int, str]]]:
    """``(file sha256, per-contig digests)`` for a FASTA, memoised by path/size/mtime.

    Hashing a genome takes seconds; the memo keeps later runs instant.
    """
    real = os.path.realpath(path)
    stat = os.stat(real)
    key = f"{real}|{stat.st_size}|{stat.st_mtime_ns}"
    memo: dict = {}
    memo_path = _digest_cache_path(cache_dir) if cache_dir else None
    if memo_path and os.path.exists(memo_path):
        try:
            with open(memo_path, encoding="utf-8") as handle:
                memo = json.load(handle)
        except (OSError, ValueError):
            memo = {}
    entry = memo.get(key)
    if entry:
        return entry["sha256"], [tuple(item) for item in entry["contigs"]]
    sha = file_sha256(real)
    contigs = scan_fasta(real)
    if memo_path:
        memo[key] = {"sha256": sha, "contigs": [list(item) for item in contigs]}
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{memo_path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(memo, handle)
            os.replace(tmp, memo_path)
        except OSError:
            pass
    return sha, contigs


def prepare_reference(reference: str, outdir: str, topology: str = "auto",
                      cache_dir: Optional[str] = None) -> ReferenceInfo:
    """Prepare ``reference`` for alignment and describe it.

    ``topology``: ``auto`` (a plasmid map's own topology; FASTA contigs are
    linear), ``circular`` (every contig is circular) or ``linear``.
    """
    if topology not in ("auto", "circular", "linear"):
        raise ValueError(f"unknown topology {topology!r}")
    reference = os.path.abspath(reference)
    if not os.path.isfile(reference):
        raise FileNotFoundError(f"reference not found: {reference}")
    os.makedirs(outdir, exist_ok=True)
    fmt = map_format(reference)
    if fmt:
        plasmid = read_plasmid_map(reference)
        circular = plasmid.circular if topology == "auto" else topology == "circular"
        fasta = os.path.join(outdir, f"{plasmid.name}.fa")
        write_fasta(fasta, plasmid.name, plasmid.sequence)
        map_copy = os.path.join(outdir, os.path.basename(reference))
        if os.path.realpath(map_copy) != os.path.realpath(reference):
            shutil.copyfile(reference, map_copy)
        return ReferenceInfo(
            fasta=fasta,
            contigs=[Contig(plasmid.name, len(plasmid.sequence),
                            sequence_md5(plasmid.sequence), circular)],
            source=reference,
            source_format=plasmid.source_format,
            source_sha256=file_sha256(reference),
            plasmid_map=map_copy,
            warnings=list(plasmid.warnings),
        )
    if not is_fasta(reference):
        raise ValueError(
            f"{reference}: unrecognised reference (expected FASTA "
            f"{', '.join(FASTA_EXTENSIONS)} or a plasmid map "
            f"{', '.join(sorted(MAP_FORMATS))})")
    sha, records = cached_fasta_digests(reference, cache_dir)
    circular = topology == "circular"
    fasta = reference
    if reference.endswith(".gz") or os.path.getsize(reference) <= COPY_FASTA_MAX_BYTES:
        # Small references (plasmids, amplicons) are copied so the run folder
        # is self-contained; gzip FASTAs are decompressed (minimap2 reads them,
        # but FiberBrowser needs a faidx-able plain FASTA).
        fasta = os.path.join(outdir, _plain_fasta_name(reference))
        if os.path.realpath(fasta) != os.path.realpath(reference):
            _copy_plain_fasta(reference, fasta)
    _ensure_fai(fasta, records)
    return ReferenceInfo(
        fasta=fasta,
        contigs=[Contig(name, length, md5, circular) for name, length, md5 in records],
        source=reference,
        source_format="fasta",
        source_sha256=sha,
    )


def _plain_fasta_name(path: str) -> str:
    name = os.path.basename(path)
    return name[:-3] if name.endswith(".gz") else name


def _copy_plain_fasta(source: str, dest: str) -> None:
    tmp = f"{dest}.tmp{os.getpid()}"
    with _open_text(source) as src, open(tmp, "w", encoding="ascii", newline="\n") as out:
        for line in src:
            out.write(line.rstrip("\r\n") + "\n")
    os.replace(tmp, dest)
    stale = dest + ".fai"
    if os.path.exists(stale):
        os.remove(stale)


def _ensure_fai(fasta: str, records: list[tuple[str, int, str]]) -> None:
    fai = fasta + ".fai"
    if os.path.exists(fai) and os.path.getmtime(fai) >= os.path.getmtime(fasta):
        return
    if not os.access(os.path.dirname(os.path.abspath(fasta)) or ".", os.W_OK):
        return  # read-only location; FiberBrowser/pysam will report it
    import pysam
    pysam.faidx(fasta)


# ---------------------------------------------------------------------------
# Header declarations
# ---------------------------------------------------------------------------

def _encode(value) -> str:
    return quote(str(value), safe="._-+")


def reference_comment(contig: Contig, info: ReferenceInfo) -> str:
    """Text of the ``@CO`` line that identifies ``contig``'s reference.

    ``FIBERHMM-REFERENCE:v1:`` followed by ``;``-separated ``key=value``
    fields whose values are percent-encoded (RFC 3986; ``[A-Za-z0-9._-+]``
    stay literal): ``contig``, ``length``, ``md5`` (as ``@SQ M5``),
    ``topology`` (``circular`` or ``linear``), ``source`` (the map or FASTA
    file name, no directory), ``source_format`` (``snapgene``, ``genbank``,
    ``embl`` or ``fasta``) and ``source_sha256`` (of the original file).
    """
    fields = [
        ("contig", contig.name),
        ("length", contig.length),
        ("md5", contig.md5),
        ("topology", "circular" if contig.circular else "linear"),
        ("source", os.path.basename(info.source)),
        ("source_format", info.source_format),
        ("source_sha256", info.source_sha256),
    ]
    return REFERENCE_COMMENT_PREFIX + ";".join(f"{k}={_encode(v)}" for k, v in fields)


def parse_reference_comment(text: str) -> Optional[dict]:
    """Parse one ``@CO`` text; ``None`` when it is not a v1 reference line."""
    if not str(text).startswith(REFERENCE_COMMENT_PREFIX):
        return None
    out: dict = {}
    for item in str(text)[len(REFERENCE_COMMENT_PREFIX):].split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = unquote(value)
    if "contig" not in out:
        return None
    if "length" in out:
        try:
            out["length"] = int(out["length"])
        except ValueError:
            return None
    return out


def declared_references(header) -> list[dict]:
    """Every valid ``FIBERHMM-REFERENCE`` declaration in a pysam header or dict."""
    data = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    parsed = (parse_reference_comment(text) for text in data.get("CO", []) or [])
    return [item for item in parsed if item]


def decorate_header(header: dict, info: ReferenceInfo) -> dict:
    """Add ``@SQ M5``/``TP`` and the ``FIBERHMM-REFERENCE`` lines to a header dict."""
    by_name = {c.name: c for c in info.contigs}
    for sq in header.get("SQ", []) or []:
        contig = by_name.get(sq.get("SN"))
        if contig is None:
            continue
        sq["M5"] = contig.md5
        if contig.circular:
            sq["TP"] = "circular"
    comments = [c for c in header.get("CO", []) or []
                if not str(c).startswith(REFERENCE_COMMENT_PREFIX)]
    for contig in info.contigs:
        if info.is_plasmid or contig.circular:
            comments.append(reference_comment(contig, info))
    if comments:
        header["CO"] = comments
    return header


def header_matches_reference(header, info: ReferenceInfo) -> tuple[bool, str]:
    """Whether a BAM header's ``@SQ`` lines describe ``info``'s contigs.

    Every ``@SQ`` must be a reference contig of the same length, and an
    ``M5``, when present, must match.
    """
    data = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    sqs = data.get("SQ", []) or []
    if not sqs:
        return False, "the BAM has no @SQ lines (unaligned)"
    by_name = {c.name: c for c in info.contigs}
    for sq in sqs:
        contig = by_name.get(sq.get("SN"))
        if contig is None:
            return False, f"contig {sq.get('SN')!r} is not in the reference"
        if int(sq.get("LN", -1)) != contig.length:
            return False, (f"contig {contig.name!r} has length {sq.get('LN')} in the "
                           f"BAM but {contig.length} in the reference")
        if sq.get("M5") and str(sq["M5"]).lower() != contig.md5:
            return False, f"contig {contig.name!r} has a different M5 checksum"
    return True, ""
