"""Live states of transferable feature types on one record: ``state_at``.

A feature type whose state rule is defined at any position (``FeatureTypeDecl.transferable_state``) can be
evaluated on a record without a scan: at a scan hit (the scan's own state, for the browser's live states)
or at any other position (matched random backgrounds). In v1 one type is transferable:

``nfr.region`` (module ``nfr``): ``1`` open, ``0`` closed, ``.`` not callable, by the rule the engine
applies (``fiberhmm.inference.accessibility.gaps.read_gaps``): the record is callable at ``[start, end)``
when one of its nucleosomes (calls of at least 90 bp, on the reference through the record's CIGAR)
starts before ``start`` and one ends after ``end``; it is open when a gap of at least ``min_gap_bp``
between consecutive nucleosomes overlaps ``[start, end)``.

The nucleosomes are the ones the scan's loader uses. For Hia5 data with the scan's default loader settings
(the lattice recaller with native replay and ``families.recall_hia5_nucleosomes``), that is the record's
nucleosomes re-called by the Hia5 replay (``consensus.upstream_recall.recall_hia5_alignment``), not the
calls stored in the BAM; ``nucleosomes='calls'`` evaluates the stored calls instead. Records the loader
does not use (unmapped, secondary, supplementary, below ``input.minimum_mapq``, without modification calls,
or whose replay fails) are ``.``. Two cases differ from a scan's states by design: a scan reports states
only for the molecules it samples (``max_quantification_molecules``), and in DddA/DddB data a duplicate
record inherits its representative's state in a scan, while ``state_at`` evaluates the record itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

NUC_MIN_BP = 90
TRANSFERABLE = ('nfr.region',)


class StateReader:
    """Evaluates transferable state rules on the records of one BAM (header-bound; caches per record).

    ``settings``: the ``nfr`` module's parameters (as in ``--parameters``) or its resolved settings; the
    defaults when None. ``chemistry``: the dataset chemistry when the header declares none.
    ``nucleosomes``: ``'scan'`` (default: the nucleosomes the scan's loader uses) or ``'calls'`` (the
    record's stored calls)."""

    def __init__(self, header, *, settings: Optional[Mapping[str, Any]] = None, chemistry: Optional[str] = None,
                 nucleosomes: str = 'scan', cache_size: int = 4096):
        from fiberhmm.inference.accessibility.workflow import NFROptions
        from fiberhmm.inference.consensus.parameters import parse_options
        from .modules.nfr import NfrModule
        if nucleosomes not in ('scan', 'calls'):
            raise ValueError("nucleosomes must be 'scan' or 'calls'")
        s = dict(settings or {})
        if not ({'nfr', 'loader', 'use_fp_classes'} <= set(s)):
            s = NfrModule().resolve(s)
        self.options = NFROptions.from_params(s['nfr'])
        if self.options.clip_gaps_to_region:
            raise ValueError('nfr.region states with nfr.clip_gaps_to_region depend on the other NFRs of the run; '
                             'they are not transferable')
        self.loader = parse_options(dict(s['loader']))
        self.header = header
        self.mode = nucleosomes
        self.chemistry = chemistry or header_chemistry(header)
        self._frames = None
        self._runtime = None
        self._cache: Dict[str, Optional[List[Tuple[int, int]]]] = {}
        self._cache_size = int(cache_size)

    # ------------------------------------------------------------------ public
    def state_at(self, read, type_id: str, chrom: str, start: int, end: Optional[int] = None) -> str:
        """``'1'``, ``'0'`` or ``'.'`` for ``read`` at ``[start, end)`` (a position when ``end`` is None)."""
        if type_id not in TRANSFERABLE:
            raise ValueError(f'{type_id}: not a transferable state type (transferable: {", ".join(TRANSFERABLE)})')
        start = int(start)
        end = start + 1 if end is None else int(end)
        if end <= start:
            end = start + 1
        if read.is_unmapped or read.reference_name != chrom:
            return '.'
        nucs = self.nucleosomes(read)
        if nucs is None:
            return '.'
        return nfr_region_state(nucs, start, end, self.options.min_gap_bp)

    def nucleosomes(self, read) -> Optional[List[Tuple[int, int]]]:
        """Reference intervals of the record's nucleosomes (>= 90 bp) as the scan's loader sees them, or
        None for a record the loader does not use."""
        key = read.to_string()
        if key in self._cache:
            return self._cache[key]
        value = self._nucleosomes(read)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
        return value

    # ------------------------------------------------------------------ the loader's view of one record
    def _nucleosomes(self, read):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            return None
        if read.mapping_quality < int(self.loader['input'].minimum_mapq):
            return None
        rt = self._chemistry_runtime()
        from fiberhmm.inference.strand_rescue import hard_observations
        try:
            extracted = hard_observations(read, rt['strand_mode'], rt['mode'], rt['context'], rt['prob_threshold'])
        except ValueError:
            return None
        if extracted is None:
            return None
        nucs, msps = self._stored(read)
        if self.mode == 'scan' and rt['replay']:
            from fiberhmm.inference.consensus.upstream_recall import MOLECULE_RECALL_FAILURES, recall_hia5_alignment
            unit = dict(raw_nuc_intervals=[list(x) for x in nucs], msp_intervals=[list(x) for x in msps],
                        _region=[int(read.reference_start), int(read.reference_end or read.reference_start)],
                        aligned_blocks=[[int(a), int(b)] for a, b in read.get_blocks()])
            opt = self.loader
            try:
                recalled = recall_hia5_alignment(
                    read, unit, rt['model'], rt['strand_mode'], rt['mode'], rt['context'], rt['prob_threshold'],
                    rt['minimum_llr'], minimum_opportunities=opt['input'].native_minimum_opportunities,
                    split_minimum_llr=opt['families'].nuc_split_minimum_llr,
                    maximum_alignment_gap_bp=opt['input'].native_maximum_alignment_gap_bp,
                    minimum_nfr_length=opt['input'].minimum_nfr_length,
                    legacy_annotation_frame=rt['legacy_frame_recall'])
            except ValueError as error:
                if str(error).startswith(MOLECULE_RECALL_FAILURES):
                    return None
                raise
            if recalled.get('status') == 'recalled':
                nucs = [(int(a), int(b)) for a, b in recalled['nucleosomes']]
        return sorted((a, b) for a, b in nucs if b - a >= NUC_MIN_BP)

    def _stored(self, read):
        """(nucleosomes, MSPs) on the reference, as ``strand_rescue.load_region_evidence`` projects them."""
        from fiberhmm.core.bam_reader import cigar_to_query_ref
        from fiberhmm.inference.strand_rescue import _mapped_annotations
        parsed = self._parsed(read)
        refs = cigar_to_query_ref(read)
        nucs = _mapped_annotations(read, 'nuc', refs, retain_topology_only=True, parsed_annotations=parsed)
        msps = _mapped_annotations(read, 'msp', refs, parsed_annotations=parsed)
        return [(int(c.start), int(c.end)) for c in nucs], [(int(c.start), int(c.end)) for c in msps]

    def _parsed(self, read):
        from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
        from fiberhmm.io.ma_tags import has_annotation_tag
        f = self._frame_context()
        try:
            parsed = _parse_all_ma_annotations(read, annotation_frame=f['frame'], molecular_layers=f['layers'],
                                               group_frame=f['group_frame']) or {}
        except (KeyError, TypeError, ValueError):
            parsed = {}
        legacy = self._chemistry_runtime()['legacy_frame_loader']
        if legacy is not None and not has_annotation_tag(read):
            from fiberhmm.inference.legacy_annotations import has_legacy_tags, legacy_annotations
            frame = legacy
            if f['provenance_legacy'] is not None and has_legacy_tags(read):
                frame = f['provenance_legacy'][0]
            try:
                parsed = legacy_annotations(read, frame) or {}
            except (KeyError, TypeError, ValueError):
                parsed = {}
        return parsed

    def _frame_context(self):
        if self._frames is None:
            from fiberhmm.io.annotation_frame import (MAGroupFrame, consensus_molecular_layers,
                                                      resolve_disabled_legacy_frame)
            group_frame = MAGroupFrame(self.header)
            legacy = self._chemistry_runtime()['legacy_frame_loader']
            self._frames = dict(group_frame=group_frame, frame=group_frame.header_frame,
                                layers=consensus_molecular_layers(self.header),
                                provenance_legacy=(resolve_disabled_legacy_frame(self.header)
                                                   if legacy == 'disabled' else None))
        return self._frames

    def _chemistry_runtime(self):
        if self._runtime is None:
            from fiberhmm.inference.consensus.bam import STAGED_LIKE, _chemistry_runtime, loader_runtime
            from fiberhmm.inference.tf_recaller import ENZYME_PRESETS
            chem = self.chemistry
            if not chem:
                raise ValueError('state_at needs the dataset chemistry: the BAM header declares none; pass chemistry=')
            preset, model, context, mode, _hit, _miss = _chemistry_runtime(loader_runtime(), chem)
            opt = self.loader
            engine = opt['cr'].engine
            hia5 = chem.startswith('hia5')
            legacy_loader = opt['input'].legacy_hia5_annotation_frame if hia5 and engine in STAGED_LIKE else None
            legacy_recall = opt['input'].legacy_hia5_annotation_frame
            if legacy_recall == 'disabled':
                from fiberhmm.io.annotation_frame import resolve_disabled_legacy_frame
                legacy_recall = (resolve_disabled_legacy_frame(self.header) or ('disabled',))[0]
            enzyme = 'hia5' if hia5 else chem
            minimum_llr = getattr(opt['input'], chem.replace('-', '_') + '_minimum_llr')
            if minimum_llr < 0:
                minimum_llr = ENZYME_PRESETS[enzyme]['min_llr']
            self._runtime = dict(
                model=model, context=context, mode=mode, strand_mode=preset['strand_mode'],
                prob_threshold=preset.get('prob_threshold'), minimum_llr=minimum_llr,
                legacy_frame_loader=legacy_loader, legacy_frame_recall=legacy_recall,
                replay=bool(hia5 and opt['input'].correct_native and engine in STAGED_LIKE
                            and opt['families'].recall_hia5_nucleosomes))
        return self._runtime


def nfr_region_state(nucleosomes: Sequence[Tuple[int, int]], start: int, end: int, min_gap_bp: int) -> str:
    """The ``nfr.region`` rule on reference nucleosome intervals (>= 90 bp), exactly as
    ``accessibility.gaps.read_gaps`` decides callable and open (without ``clip_gaps_to_region``)."""
    from fiberhmm.inference.accessibility.gaps import read_gaps
    ok, gaps = read_gaps(dict(raw_nuc_intervals=[list(x) for x in nucleosomes], positions=[], hits=[],
                              raw_tf_intervals=[]), (int(start), int(end)), int(min_gap_bp), NUC_MIN_BP)
    if not ok:
        return '.'
    return '1' if gaps else '0'


def header_chemistry(header) -> Optional[str]:
    """The consensus chemistry profile a BAM header declares (the same rule as
    ``io.bam_header.bam_chemistry_profile``), or None."""
    from fiberhmm.io.bam_header import (chemistry_profile_from_declaration, declared_chemistries,
                                        infer_legacy_chemistry)
    declarations = declared_chemistries(header)
    if declarations:
        profiles = {chemistry_profile_from_declaration(d) for d in declarations} - {None}
        if len(profiles) > 1:
            return None
        return chemistry_profile_from_declaration(declarations[-1])
    legacy = infer_legacy_chemistry(header)
    return chemistry_profile_from_declaration(legacy, enzyme_recorded=False) if legacy else None


def state_at(read, type_id: str, chrom: str, start: int, end: Optional[int] = None, *, header=None,
             settings: Optional[Mapping[str, Any]] = None, chemistry: Optional[str] = None,
             nucleosomes: str = 'scan') -> str:
    """The state of a transferable feature type on one record at ``[start, end)`` (see the module docstring).
    For many records of one BAM, build one :class:`StateReader` and call its ``state_at``."""
    reader = StateReader(header if header is not None else read.header, settings=settings, chemistry=chemistry,
                         nucleosomes=nucleosomes)
    return reader.state_at(read, type_id, chrom, start, end)


def state_at_hit(read, hit: Mapping[str, Any], **kw) -> str:
    """``state_at`` at a scan hit (a ``hits.tsv.bgz`` row or a :class:`fiberhmm.scan.types.Hit`)."""
    get = hit.get if isinstance(hit, Mapping) else (lambda k: getattr(hit, k))
    return state_at(read, get('type'), get('chrom'), int(get('start')), int(get('end')), **kw)
