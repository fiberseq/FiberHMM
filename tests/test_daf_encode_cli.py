from fiberhmm.cli import daf_encode


def test_daf_encode_parser_defaults_and_strand_choices():
    parser = daf_encode._build_daf_encode_parser()

    args = parser.parse_args(["-i", "in.bam", "-o", "out.bam"])

    assert args.input == "in.bam"
    assert args.output == "out.bam"
    assert args.reference is None
    assert args.min_mapq == 20
    assert args.min_read_length == 1000
    assert args.io_threads == 4
    assert args.strand == "auto"


def test_daf_encode_main_maps_auto_strand_to_none(monkeypatch):
    calls = []

    monkeypatch.setattr(
        daf_encode,
        "process_bam_daf_encode",
        lambda **kwargs: calls.append(kwargs),
    )

    daf_encode.main([
        "-i", "in.bam",
        "-o", "out.bam",
        "--reference", "ref.fa",
        "--min-mapq", "12",
        "--min-read-length", "500",
        "--io-threads", "2",
        "--strand", "auto",
    ])

    assert calls == [{
        "input_bam": "in.bam",
        "output_bam": "out.bam",
        "reference": "ref.fa",
        "min_mapq": 12,
        "min_read_length": 500,
        "io_threads": 2,
        "force_strand": None,
    }]


def test_daf_encode_main_passes_forced_strand(monkeypatch):
    calls = []

    monkeypatch.setattr(
        daf_encode,
        "process_bam_daf_encode",
        lambda **kwargs: calls.append(kwargs),
    )

    daf_encode.main(["-i", "in.bam", "-o", "out.bam", "--strand", "CT"])

    assert calls[0]["force_strand"] == "CT"
