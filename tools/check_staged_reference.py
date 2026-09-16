"""Run the original consolidation regression suite against extracted kernels."""
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'paper/analysis/benchmark/napa_external_footprint_comparison_20260911'
sys.path.insert(0,str(ROOT))
reference=ROOT/'fiberhmm/inference/consensus/harmonized_families/reference'
for name in json.loads((reference/'SOURCE_MANIFEST.json').read_text()):
    sys.modules[name]=importlib.import_module('fiberhmm.inference.consensus.harmonized_families.reference.'+name)
sys.path.append(str(SOURCE))
names=['test_bounded_parent_prototype','test_native_cell_consolidation','test_reuse_consensus_fits',
       'test_resolve_consensus_representatives','test_native_support_nomination']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
