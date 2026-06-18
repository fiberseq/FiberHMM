from array import array

import numpy as np

from fiberhmm.core.tag_access import compact_ml_value


def test_compact_ml_value_converts_ml_like_values_to_bytes():
    assert compact_ml_value(bytes([200])) == bytes([200])
    assert compact_ml_value(bytearray([200])) == bytes([200])
    assert compact_ml_value(memoryview(bytes([200]))) == bytes([200])
    assert compact_ml_value(array("B", [200])) == bytes([200])
    assert compact_ml_value([200]) == bytes([200])
    assert compact_ml_value(np.asarray([200], dtype=np.uint8)) == bytes([200])


def test_compact_ml_value_promotes_scalar_values():
    assert compact_ml_value(200) == bytes([200])
    assert compact_ml_value(np.asarray(200, dtype=np.uint8)) == bytes([200])
