"""Every reader of the legacy footprint pairs uses the shared array-pair rule.

``fiberhmm.io.legacy_tags.array_tag_pair`` decides when ``ns``/``nl`` and
``as``/``al`` form a call: both tags present, integer arrays, equal lengths.
A scalar or string tag (Dorado writes ``ns:i``, a signal count) or unequal
lengths make the pair absent. Tools that read the tags raw used to crash or
disagree with each other on such reads (recall-tfs raised a TypeError on a
scalar ``ns`` beside an ``nl`` array). This guard fails when a module outside
``legacy_tags`` reads one of these tags directly.
"""
import re
from pathlib import Path

import fiberhmm

PACKAGE = Path(fiberhmm.__file__).parent
RULE_MODULE = PACKAGE / 'io' / 'legacy_tags.py'
RAW_READ = re.compile(r"""get_tag\(\s*(?:['"](?:ns|nl|as|al)['"]|start_tag|length_tag)\s*[,)]""")


def test_no_module_reads_legacy_pairs_without_the_shared_rule():
    offenders = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path == RULE_MODULE:
            continue
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if RAW_READ.search(line):
                offenders.append(f'{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}')
    assert not offenders, ('read ns/nl/as/al through fiberhmm.io.legacy_tags.array_tag_pair:\n'
                           + '\n'.join(offenders))


def test_guard_pattern_catches_raw_reads():
    for line in ("ns = read.get_tag('ns')", 'x = read.get_tag("al")', 'v = read.get_tag(start_tag)'):
        assert RAW_READ.search(line), line
    for line in ("read.get_tag('nq')", "read.get_tag('MA')", "read.has_tag('ns')"):
        assert not RAW_READ.search(line), line
