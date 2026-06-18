"""Slim read wrapper used by multiprocessing payload workers."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field


@dataclass(slots=True)
class PayloadRead:
    """Minimal duck-type for the pysam.AlignedSegment surface workers need."""

    query_sequence: object
    is_reverse: bool
    tags: InitVar[object]
    query_name: object = None
    daf_md_result: InitVar[object] = None
    _tags: object = field(init=False, repr=False)
    _daf_md_result: object = field(init=False, repr=False)

    def __post_init__(self, tags, daf_md_result) -> None:
        self._tags = tags
        self._daf_md_result = daf_md_result

    def has_tag(self, tag):
        return tag in self._tags

    def get_tag(self, tag):
        return self._tags[tag]
