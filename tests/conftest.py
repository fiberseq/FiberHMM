"""
Shared pytest fixtures for FiberHMM tests.
"""
import pytest
import numpy as np
import pysam
import tempfile
import os


@pytest.fixture
def simple_emission_probs():
    """
    Simple 2-state, 4-symbol emission probability matrix.
    State 0: Low methylation (footprint)
    State 1: High methylation (accessible)
    """
    return np.array([
        [0.1, 0.2, 0.3, 0.4],  # State 0: prefers higher symbols (unmethylated)
        [0.4, 0.3, 0.2, 0.1],  # State 1: prefers lower symbols (methylated)
    ])


@pytest.fixture
def hexamer_emission_probs():
    """
    Realistic emission probability matrix for k=3 context.
    4^3 * 2 = 128 symbols (64 methylated + 64 unmethylated contexts)
    """
    np.random.seed(42)
    n_contexts = 64  # 4^3
    n_symbols = n_contexts * 2  # methylated + unmethylated

    # State 0 (footprint): low methylation probability
    state0_meth = np.random.uniform(0.05, 0.15, n_contexts)
    state0_unmeth = 1 - state0_meth

    # State 1 (accessible): high methylation probability
    state1_meth = np.random.uniform(0.6, 0.9, n_contexts)
    state1_unmeth = 1 - state1_meth

    emit = np.zeros((2, n_symbols))
    emit[0, :n_contexts] = state0_meth
    emit[0, n_contexts:] = state0_unmeth
    emit[1, :n_contexts] = state1_meth
    emit[1, n_contexts:] = state1_unmeth

    # Normalize
    emit /= emit.sum(axis=1, keepdims=True)

    return emit


@pytest.fixture
def simple_observations():
    """Simple observation sequence that should show clear state transitions."""
    # Pattern: low symbols (state 1) -> high symbols (state 0) -> low symbols (state 1)
    return np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0], dtype=np.int32)


@pytest.fixture
def trained_model(simple_emission_probs, simple_observations):
    """A trained FiberHMM model for testing."""
    from fiberhmm.core.hmm import FiberHMM

    model = FiberHMM(n_states=2)
    model.emissionprob_ = simple_emission_probs
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

    # Train briefly
    X = simple_observations.reshape(-1, 1)
    model.fit(X, verbose=False)

    return model


@pytest.fixture
def k3_model(hexamer_emission_probs):
    """FiberHMM model with k=3 context (128 emission symbols)."""
    from fiberhmm.core.hmm import FiberHMM

    model = FiberHMM(n_states=2)
    model.emissionprob_ = hexamer_emission_probs
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    return model


@pytest.fixture
def test_model_path():
    """Path to a real JSON model for integration tests (if available)."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dm6_dddb.json')
    if os.path.exists(path):
        return path
    return None


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_bam_data():
    """Mock BAM read data for testing bam_reader functions."""
    return {
        'read_id': 'test_read_001',
        'chrom': 'chr1',
        'start': 1000,
        'end': 2000,
        'sequence': 'ACGTACGTACGT' * 83,  # ~1000 bp
        'methylation_positions': [10, 25, 50, 75, 100, 150, 200],
        'methylation_probs': [200, 180, 220, 190, 240, 170, 210],  # 0-255 scale
    }


# ---------------------------------------------------------------------------
# Synthetic BAM generation for integration & streaming tests
# ---------------------------------------------------------------------------

def _make_mm_ml_tags(sequence, mod_positions, rng):
    """
    Build valid MM:Z and ML:B:C tags for m6A modifications at given positions.

    Args:
        sequence: DNA sequence string
        mod_positions: sorted list of query positions with modifications (must be A bases)
        rng: numpy RandomState

    Returns:
        (mm_tag_str, ml_tag_list) or (None, None) if no A bases
    """
    # Find all A positions in the sequence
    a_positions = [i for i, b in enumerate(sequence) if b == 'A']
    if not a_positions:
        return None, None

    # Filter mod_positions to only include valid A positions
    a_set = set(a_positions)
    mod_positions = [p for p in mod_positions if p in a_set]

    if not mod_positions:
        return 'A+a;', []

    # Build skip counts: for each modified A, count unmodified A's since last
    a_to_idx = {pos: i for i, pos in enumerate(a_positions)}
    skips = []
    prev_a_idx = -1
    for mp in mod_positions:
        cur_a_idx = a_to_idx[mp]
        skip = cur_a_idx - prev_a_idx - 1
        skips.append(skip)
        prev_a_idx = cur_a_idx

    mm_tag = 'A+a,' + ','.join(str(s) for s in skips) + ';'
    ml_tag = [int(rng.randint(180, 256)) for _ in mod_positions]

    return mm_tag, ml_tag


def make_synthetic_bam(output_path, n_reads=100, read_length=5000,
                       n_chroms=3, chrom_length=1_000_000,
                       mod_rate=0.05, seed=42, aligned=True):
    """
    Generate a synthetic BAM with fiber-seq-like properties.

    Reads have random DNA sequences with m6A modifications encoded as MM/ML
    tags. When aligned=True, reads are coordinate-sorted and indexed.

    Args:
        output_path: path for output BAM
        n_reads: number of reads to generate
        read_length: length of each read
        n_chroms: number of chromosomes
        chrom_length: length of each chromosome
        mod_rate: fraction of A bases to mark as modified
        seed: random seed for reproducibility
        aligned: if True, reads are aligned with coords; if False, unmapped

    Returns:
        output_path
    """
    rng = np.random.RandomState(seed)
    bases = np.array(list('ACGT'))

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate' if aligned else 'unsorted'},
        'SQ': [{'SN': f'chr{i+1}', 'LN': chrom_length} for i in range(n_chroms)],
    })

    # Write unsorted first, then sort
    unsorted_path = output_path + '.unsorted.bam'
    with pysam.AlignmentFile(unsorted_path, "wb", header=header) as outf:
        for i in range(n_reads):
            # Random sequence
            seq = ''.join(bases[rng.randint(0, 4, read_length)])

            # Find A positions and randomly modify some
            a_positions = [j for j, b in enumerate(seq) if b == 'A']
            if a_positions:
                mod_mask = rng.random(len(a_positions)) < mod_rate
                mod_positions = sorted(
                    a_positions[j] for j in range(len(a_positions)) if mod_mask[j]
                )
            else:
                mod_positions = []

            # Build MM/ML tags
            mm_tag, ml_tag = _make_mm_ml_tags(seq, mod_positions, rng)

            # Create aligned segment
            a = pysam.AlignedSegment()
            a.query_name = f'read_{i:06d}'
            a.query_sequence = seq
            a.query_qualities = pysam.qualitystring_to_array('I' * read_length)

            if aligned:
                chrom_idx = i % n_chroms
                max_pos = max(0, chrom_length - read_length)
                pos = rng.randint(0, max_pos + 1) if max_pos > 0 else 0
                a.flag = 0
                a.reference_id = chrom_idx
                a.reference_start = pos
                a.mapping_quality = 60
                a.cigar = [(0, read_length)]  # All M
            else:
                a.flag = 4  # unmapped
                a.reference_id = -1
                a.reference_start = -1
                a.mapping_quality = 0

            if mm_tag is not None:
                a.set_tag('MM', mm_tag)
                if ml_tag:
                    a.set_tag('ML', ml_tag)

            outf.write(a)

    if aligned:
        # Sort and index
        pysam.sort("-o", output_path, unsorted_path)
        os.remove(unsorted_path)
        pysam.index(output_path)
    else:
        os.replace(unsorted_path, output_path)

    return output_path


@pytest.fixture(scope="session")
def _synthetic_bam_dir():
    """Session-scoped temp directory for synthetic BAMs."""
    with tempfile.TemporaryDirectory(prefix="fiberhmm_test_") as d:
        yield d


@pytest.fixture(scope="session")
def synthetic_bam_small(_synthetic_bam_dir):
    """100 aligned reads, 3 chroms — for correctness tests."""
    path = os.path.join(_synthetic_bam_dir, "small.bam")
    return make_synthetic_bam(path, n_reads=100, n_chroms=3, read_length=5000)


@pytest.fixture(scope="session")
def synthetic_bam_medium(_synthetic_bam_dir):
    """2000 aligned reads, 5 chroms — for throughput tests."""
    path = os.path.join(_synthetic_bam_dir, "medium.bam")
    return make_synthetic_bam(path, n_reads=2000, n_chroms=5, read_length=5000)


@pytest.fixture(scope="session")
def unaligned_bam(_synthetic_bam_dir):
    """100 unaligned reads with sequences and MM/ML tags."""
    path = os.path.join(_synthetic_bam_dir, "unaligned.bam")
    return make_synthetic_bam(path, n_reads=100, n_chroms=3,
                              read_length=5000, aligned=False)


@pytest.fixture(scope="session")
def empty_bam(_synthetic_bam_dir):
    """Empty BAM file with header but no reads."""
    path = os.path.join(_synthetic_bam_dir, "empty.bam")
    return make_synthetic_bam(path, n_reads=0, aligned=True)


def make_synthetic_iupac_bam(output_path, n_reads=50, read_length=5000,
                             n_chroms=3, chrom_length=1_000_000,
                             deam_rate=0.10, seed=99, aligned=True):
    """
    Generate a synthetic DAF-seq BAM with IUPAC R/Y encoding.

    Reads have Y (deaminated C, + strand) or R (deaminated G, - strand) in
    the sequence instead of MM/ML tags.  Each read also carries an ``st:Z``
    tag (``CT`` or ``GA``).

    Args:
        output_path: path for output BAM
        n_reads: number of reads to generate
        read_length: length of each read
        n_chroms: number of chromosomes
        chrom_length: length of each chromosome
        deam_rate: fraction of C (or G) bases to mark as deaminated
        seed: random seed for reproducibility
        aligned: if True, reads are coordinate-sorted and indexed

    Returns:
        output_path
    """
    rng = np.random.RandomState(seed)
    bases = np.array(list('ACGT'))

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate' if aligned else 'unsorted'},
        'SQ': [{'SN': f'chr{i+1}', 'LN': chrom_length} for i in range(n_chroms)],
    })

    unsorted_path = output_path + '.unsorted.bam'
    with pysam.AlignmentFile(unsorted_path, "wb", header=header) as outf:
        for i in range(n_reads):
            # Random pure ACGT sequence
            seq_arr = bases[rng.randint(0, 4, read_length)]

            # Alternate strands: even reads = + strand, odd = - strand
            is_plus = (i % 2 == 0)

            if is_plus:
                # + strand: deaminate some C → Y
                st_tag = 'CT'
                target_positions = [j for j, b in enumerate(seq_arr) if b == 'C']
            else:
                # - strand: deaminate some G → R
                st_tag = 'GA'
                target_positions = [j for j, b in enumerate(seq_arr) if b == 'G']

            if target_positions:
                mask = rng.random(len(target_positions)) < deam_rate
                for idx, do_deam in zip(target_positions, mask):
                    if do_deam:
                        seq_arr[idx] = 'Y' if is_plus else 'R'

            seq = ''.join(seq_arr)

            a = pysam.AlignedSegment()
            a.query_name = f'iupac_read_{i:06d}'
            a.query_sequence = seq
            a.query_qualities = pysam.qualitystring_to_array('I' * read_length)
            a.set_tag('st', st_tag, value_type='Z')

            if aligned:
                chrom_idx = i % n_chroms
                max_pos = max(0, chrom_length - read_length)
                pos = rng.randint(0, max_pos + 1) if max_pos > 0 else 0
                a.flag = 0
                a.reference_id = chrom_idx
                a.reference_start = pos
                a.mapping_quality = 60
                a.cigar = [(0, read_length)]
            else:
                a.flag = 4
                a.reference_id = -1
                a.reference_start = -1
                a.mapping_quality = 0

            outf.write(a)

    if aligned:
        pysam.sort("-o", output_path, unsorted_path)
        os.remove(unsorted_path)
        pysam.index(output_path)
    else:
        os.replace(unsorted_path, output_path)

    return output_path


@pytest.fixture(scope="session")
def synthetic_iupac_bam(_synthetic_bam_dir):
    """50 aligned reads with IUPAC R/Y encoding + st tags."""
    path = os.path.join(_synthetic_bam_dir, "iupac_daf.bam")
    return make_synthetic_iupac_bam(path, n_reads=50, n_chroms=3, read_length=5000)


@pytest.fixture(scope="session")
def benchmark_model_path():
    """Path to a real model for integration tests."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hia5_pacbio.json')
    assert os.path.exists(path), f"Model not found: {path}"
    return os.path.abspath(path)
