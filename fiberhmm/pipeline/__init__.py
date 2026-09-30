"""``fiberhmm-pipeline``: reads + reference -> called BAM ready for FiberBrowser.

The package is split by concern so each piece can be tested without the others:

``reference``  plasmid-map import, contig naming, @SQ M5 / TP and the
               ``FIBERHMM-REFERENCE`` header line
``aligner``    minimap2 (binary or ``mappy``) discovery, cached indexes, read input
``circular``   origin-spanning alignments on circular references, hard clipping
``progress``   JSON-lines progress events and step completion markers
``runner``     the ordered steps (reference, index, align, call, tracks, browser)

The command-line entry point is :mod:`fiberhmm.cli.pipeline`.
"""
