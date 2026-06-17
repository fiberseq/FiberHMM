"""Slim read wrapper used by multiprocessing payload workers."""

from __future__ import annotations


class PayloadRead:
    """Minimal duck-type for the pysam.AlignedSegment surface workers need."""

    __slots__ = ('query_name', 'query_sequence', 'is_reverse', '_tags',
                 '_daf_md_result')

    def __init__(
        self,
        query_sequence,
        is_reverse,
        tags,
        query_name=None,
        daf_md_result=None,
    ):
        self.query_name = query_name
        self.query_sequence = query_sequence
        self.is_reverse = is_reverse
        self._tags = tags
        self._daf_md_result = daf_md_result

    def has_tag(self, tag):
        return tag in self._tags

    def get_tag(self, tag):
        return self._tags[tag]
