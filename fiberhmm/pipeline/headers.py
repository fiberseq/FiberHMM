"""Carry the input BAMs' header provenance into ``fiberhmm-pipeline``'s output.

minimap2 writes a fresh header, so the basecaller's ``@RG``/``@PG`` lines
(dorado's ``basecall_model=``/``modbase_models=``, PacBio ``ccs``/``jasmine``)
would be lost on alignment. :func:`carry_input_headers` merges every input's
``@RG`` lines, ``@PG`` chain and ``@CO`` lines:

* IDs stay unique across inputs: a line identical to one already carried is
  merged into it; a different line whose ID is taken (by another input, or by
  the pipeline's own read group / programs) gets ``-2``, ``-3``... appended.
  ``rg_maps[i]`` maps input ``i``'s read-group IDs to the output IDs, so the
  per-read ``RG:Z`` tags can be rewritten consistently; ``PP`` links inside a
  chain follow their program's new ID.
* FiberHMM's own ``@PG`` records and ``@CO`` declarations
  (``FIBERHMM-*``, ``MA-TYPES:``) of a realigned input are dropped: they
  describe calls the realigned records no longer carry (children of a
  dropped record are linked to its parent instead).
* ``leaf`` is the ``@PG`` the aligner's record chains to (``PP``): the last
  carried program no other carried program names as its parent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

FIBERHMM_COMMENT_PREFIXES = ("FIBERHMM-", "MA-TYPES:")


@dataclass
class CarriedHeaders:
    rg: list = field(default_factory=list)
    pg: list = field(default_factory=list)
    co: list = field(default_factory=list)
    rg_maps: list = field(default_factory=list)  # per input: {input ID: output ID}
    leaf: Optional[str] = None


def _unique(base: str, taken: set) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _is_fiberhmm_program(pg: dict) -> bool:
    return str(pg.get("PN") or pg.get("ID") or "").lower().startswith("fiberhmm")


def _topological(programs: list) -> list:
    """Programs with every parent before its children (input order otherwise)."""
    by_id = {pg.get("ID"): pg for pg in programs}
    done: set = set()
    order: list = []

    def visit(pg, trail):
        pid = pg.get("ID")
        if pid in done or pid in trail:
            return
        parent = by_id.get(pg.get("PP"))
        if parent is not None:
            visit(parent, trail | {pid})
        done.add(pid)
        order.append(pg)

    for pg in programs:
        visit(pg, frozenset())
    return order


def carry_input_headers(headers: Iterable[Optional[dict]],
                        reserved_rg: Iterable[str] = (),
                        reserved_pg: Iterable[str] = ()) -> CarriedHeaders:
    """Merge the ``@RG``/``@PG``/``@CO`` lines of ``headers`` (one per input;
    ``None`` for a FASTQ). See the module docstring."""
    out = CarriedHeaders()
    taken_rg = set(reserved_rg)
    taken_pg = set(reserved_pg)
    carried_rg: list = []  # (original line, output ID)
    carried_pg: list = []  # (line with mapped PP but original ID, output ID)
    for header in headers:
        rg_map: dict = {}
        out.rg_maps.append(rg_map)
        if not header:
            continue
        data = header.to_dict() if hasattr(header, "to_dict") else header
        for group in data.get("RG", []) or []:
            group = dict(group)
            rid = str(group.get("ID", ""))
            if not rid:
                continue
            same = next((new for orig, new in carried_rg if orig == group), None)
            if same is not None:
                rg_map[rid] = same
                continue
            new = _unique(rid, taken_rg)
            taken_rg.add(new)
            carried_rg.append((group, new))
            rg_map[rid] = new
            out.rg.append({**group, "ID": new})

        programs = [dict(pg) for pg in data.get("PG", []) or [] if pg.get("ID")]
        ids = {pg["ID"] for pg in programs}
        parent_of = {pg["ID"]: pg.get("PP") for pg in programs}
        dropped = {pg["ID"] for pg in programs if _is_fiberhmm_program(pg)}

        def live_parent(pid, seen=()):
            # The nearest kept ancestor (dropped FiberHMM records are bypassed).
            while pid in dropped and pid not in seen:
                seen = (*seen, pid)
                pid = parent_of.get(pid)
            return pid if pid in ids and pid not in dropped else None

        pg_map: dict = {}
        for pg in _topological(programs):
            if pg["ID"] in dropped:
                continue
            parent = live_parent(pg.get("PP"))
            line = {k: v for k, v in pg.items() if k != "PP"}
            if parent is not None and parent in pg_map:
                line["PP"] = pg_map[parent]
            same = next((new for orig, new in carried_pg if orig == line), None)
            if same is not None:
                pg_map[pg["ID"]] = same
                continue
            new = _unique(str(pg["ID"]), taken_pg)
            taken_pg.add(new)
            carried_pg.append((line, new))
            pg_map[pg["ID"]] = new
            out.pg.append({**line, "ID": new})

        for comment in data.get("CO", []) or []:
            text = str(comment)
            if text.startswith(FIBERHMM_COMMENT_PREFIXES) or text in out.co:
                continue
            out.co.append(text)

    parents = {pg.get("PP") for pg in out.pg if pg.get("PP")}
    leaves = [pg["ID"] for pg in out.pg if pg["ID"] not in parents]
    out.leaf = leaves[-1] if leaves else None
    return out


def missing_pp_links(header: dict) -> list[str]:
    """``@PG`` IDs whose ``PP`` names no program in ``header`` (a validity check)."""
    programs = header.get("PG", []) or []
    ids = {pg.get("ID") for pg in programs}
    return [pg.get("ID") for pg in programs if pg.get("PP") and pg["PP"] not in ids]
