"""Moved to :mod:`fiberhmm.assoc.coaccess` (the feature hub's association library).

This module is an alias kept so existing imports keep working: importing it returns the
``fiberhmm.assoc.coaccess`` module object itself, so attribute reads and monkeypatches act on the one module
that ``fiberhmm-nfr`` uses.
"""
import sys

from fiberhmm.assoc import coaccess as _coaccess

sys.modules[__name__] = _coaccess
