"""Helpers for BAM index discovery and creation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class _BamIndexPaths:
    adjacent: str
    replaced_suffix: str

    def as_tuple(self) -> tuple[str, str]:
        return self.adjacent, self.replaced_suffix


def _bam_index_paths(bam_path: str) -> _BamIndexPaths:
    bam_path = os.fspath(bam_path)
    stem, suffix = os.path.splitext(bam_path)
    suffix_index_path = stem + '.bai' if suffix.lower() == '.bam' else bam_path + '.bai'
    return _BamIndexPaths(
        adjacent=bam_path + '.bai',
        replaced_suffix=suffix_index_path,
    )


def bam_index_paths(bam_path: str) -> tuple[str, str]:
    return _bam_index_paths(bam_path).as_tuple()


def bam_index_exists(bam_path: str) -> bool:
    paths = _bam_index_paths(bam_path)
    return (
        os.path.exists(paths.adjacent)
        or os.path.exists(paths.replaced_suffix)
    )


def ensure_bam_index(bam_path: str, message: str) -> None:
    bam_path = os.fspath(bam_path)
    if bam_index_exists(bam_path):
        return
    print(message)
    import pysam
    pysam.index(bam_path)
