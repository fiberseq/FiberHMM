"""Portable, strict JSON artifacts for frozen consensus runs."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from fiberhmm.io.output_files import mkstemp_shared


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_bytes(value):
    """The canonical encoding that ``digest`` hashes (sorted keys, compact)."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=json_default).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path, value, *, encoded=None):
    """Write ``value``; ``encoded`` may supply its canonical bytes to avoid a
    second serialization (same JSON content, sorted key order)."""
    path = Path(path)
    # dumps uses the C encoder. dump instead iterates millions of tiny Python
    # chunks in these audit ledgers. Keep the same JSON semantics/float text;
    # compression level changes only storage, never scientific content/digests.
    if encoded is None:
        encoded = json.dumps(value, allow_nan=False, default=json_default,
                             separators=(',', ':')).encode('utf-8')
    # Never leave a truncated published artifact on cancellation/write failure.
    # Created like open() does (umask applies), not 0600 like tempfile.mkstemp.
    fd, temporary = mkstemp_shared(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            if path.suffix == '.gz':
                # Content is what is compared and digested; the compressed byte
                # stream is a container. ISA-L's igzip writes standard gzip
                # members several times faster than zlib at this level.
                try:
                    from isal import igzip
                    with igzip.IGzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0,
                                         filename='') as handle:
                        handle.write(encoded)
                except ImportError:
                    with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0,
                                       filename='') as handle:
                        handle.write(encoded)
            else:
                raw.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return json.load(handle)
