"""Shared read folds for cross-fitting (fiberhmm.io.folds)."""
import hashlib

import pytest

from fiberhmm.io.folds import FoldSpec, fold_of


def test_fold_is_md5_of_the_read_name_mod_k():
    for name in ('m1/100/ccs', 'read_a', 'x' * 40):
        assert fold_of(name) == int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
        assert fold_of(name, 3, 'sha1') == int(hashlib.sha1(name.encode()).hexdigest(), 16) % 3


def test_spec_parsing_and_canonical_form():
    spec = FoldSpec.parse('hash=md5,mod=10,discover=0-4')
    assert spec.discover == (0, 1, 2, 3, 4) and spec.mod == 10 and spec.hash == 'md5'
    assert spec.canonical() == 'hash=md5,mod=10,discover=0-4'
    assert FoldSpec.parse('discover=4+0-2, mod=5').canonical() == 'hash=md5,mod=5,discover=0-2+4'
    assert FoldSpec.parse(dict(mod=4, discover=[3, 1])).discover == (1, 3)
    assert FoldSpec.parse(spec.canonical()) == spec
    assert FoldSpec.parse(None) is None and FoldSpec.parse('') is None
    assert FoldSpec.parse('mod=2,discover=0-1').all_folds
    names = [f'read{i}' for i in range(200)]
    held = [n for n in names if not spec.discovers(n)]
    assert 0 < len(held) < len(names)
    assert all(spec.fold(n) >= 5 for n in held)


@pytest.mark.parametrize('bad', ['mod=10', 'discover=0-10', 'discover=4-2', 'hash=crc,discover=0', 'mod=0,discover=0',
                                 'discover=a', 'discover=0,discover=1', 'folds=1', 'discover=+'])
def test_bad_specs_are_refused(bad):
    with pytest.raises(ValueError):
        FoldSpec.parse(bad)
