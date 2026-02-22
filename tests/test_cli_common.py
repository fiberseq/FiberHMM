"""
Tests for fiberhmm.cli.common argument factories.
"""
import pytest
import argparse

from fiberhmm.cli.common import (
    add_mode_args,
    add_filter_args,
    add_parallel_args,
    add_context_args,
    add_edge_trim_args,
    add_output_args,
    add_stats_args,
    add_verbose_args,
)


class TestAddModeArgs:
    def test_default_mode(self):
        parser = argparse.ArgumentParser()
        add_mode_args(parser)
        args = parser.parse_args([])
        assert args.mode == 'pacbio-fiber'

    def test_custom_default(self):
        parser = argparse.ArgumentParser()
        add_mode_args(parser, default='daf')
        args = parser.parse_args([])
        assert args.mode == 'daf'

    def test_valid_choices(self):
        parser = argparse.ArgumentParser()
        add_mode_args(parser)
        for mode in ['pacbio-fiber', 'nanopore-fiber', 'daf']:
            args = parser.parse_args(['--mode', mode])
            assert args.mode == mode

    def test_invalid_choice(self):
        parser = argparse.ArgumentParser()
        add_mode_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(['--mode', 'invalid'])

    def test_required_mode(self):
        parser = argparse.ArgumentParser()
        add_mode_args(parser, required=True)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestAddFilterArgs:
    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_filter_args(parser)
        args = parser.parse_args([])
        assert args.min_mapq == 20
        assert args.prob_threshold == 128
        assert args.min_read_length == 1000

    def test_custom_defaults(self):
        parser = argparse.ArgumentParser()
        add_filter_args(parser, min_mapq=10, prob_threshold=200, min_read_length=500)
        args = parser.parse_args([])
        assert args.min_mapq == 10
        assert args.prob_threshold == 200
        assert args.min_read_length == 500

    def test_override_from_cli(self):
        parser = argparse.ArgumentParser()
        add_filter_args(parser)
        args = parser.parse_args(['--min-mapq', '30', '--prob-threshold', '200', '--min-read-length', '2000'])
        assert args.min_mapq == 30
        assert args.prob_threshold == 200
        assert args.min_read_length == 2000

    def test_short_flag(self):
        parser = argparse.ArgumentParser()
        add_filter_args(parser)
        args = parser.parse_args(['-q', '15'])
        assert args.min_mapq == 15


class TestAddParallelArgs:
    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_parallel_args(parser)
        args = parser.parse_args([])
        assert args.cores == 1
        assert args.region_size == 10_000_000
        assert args.skip_scaffolds is False
        assert args.chroms is None

    def test_custom_defaults(self):
        parser = argparse.ArgumentParser()
        add_parallel_args(parser, default_cores=4, default_region_size=5_000_000)
        args = parser.parse_args([])
        assert args.cores == 4
        assert args.region_size == 5_000_000

    def test_chroms_list(self):
        parser = argparse.ArgumentParser()
        add_parallel_args(parser)
        args = parser.parse_args(['--chroms', 'chr1', 'chr2', 'chrX'])
        assert args.chroms == ['chr1', 'chr2', 'chrX']

    def test_skip_scaffolds_flag(self):
        parser = argparse.ArgumentParser()
        add_parallel_args(parser)
        args = parser.parse_args(['--skip-scaffolds'])
        assert args.skip_scaffolds is True

    def test_short_cores_flag(self):
        parser = argparse.ArgumentParser()
        add_parallel_args(parser)
        args = parser.parse_args(['-c', '8'])
        assert args.cores == 8


class TestAddContextArgs:
    def test_single_default(self):
        parser = argparse.ArgumentParser()
        add_context_args(parser)
        args = parser.parse_args([])
        assert args.context_size == 3

    def test_single_override(self):
        parser = argparse.ArgumentParser()
        add_context_args(parser)
        args = parser.parse_args(['-k', '5'])
        assert args.context_size == 5

    def test_multiple_mode(self):
        parser = argparse.ArgumentParser()
        add_context_args(parser, default=[3, 4, 5], multiple=True)
        args = parser.parse_args([])
        assert args.context_sizes == [3, 4, 5]

    def test_multiple_override(self):
        parser = argparse.ArgumentParser()
        add_context_args(parser, multiple=True)
        args = parser.parse_args(['-k', '3', '5', '7'])
        assert args.context_sizes == [3, 5, 7]


class TestAddEdgeTrimArgs:
    def test_default(self):
        parser = argparse.ArgumentParser()
        add_edge_trim_args(parser)
        args = parser.parse_args([])
        assert args.edge_trim == 10

    def test_override(self):
        parser = argparse.ArgumentParser()
        add_edge_trim_args(parser)
        args = parser.parse_args(['--edge-trim', '20'])
        assert args.edge_trim == 20


class TestAddOutputArgs:
    def test_required(self):
        parser = argparse.ArgumentParser()
        add_output_args(parser, required=True)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_not_required(self):
        parser = argparse.ArgumentParser()
        add_output_args(parser, required=False)
        args = parser.parse_args([])
        assert args.output is None

    def test_short_flag(self):
        parser = argparse.ArgumentParser()
        add_output_args(parser)
        args = parser.parse_args(['-o', '/tmp/out'])
        assert args.output == '/tmp/out'


class TestAddStatsArgs:
    def test_default_false(self):
        parser = argparse.ArgumentParser()
        add_stats_args(parser)
        args = parser.parse_args([])
        assert args.stats is False

    def test_flag_true(self):
        parser = argparse.ArgumentParser()
        add_stats_args(parser)
        args = parser.parse_args(['--stats'])
        assert args.stats is True


class TestAddVerboseArgs:
    def test_default_false(self):
        parser = argparse.ArgumentParser()
        add_verbose_args(parser)
        args = parser.parse_args([])
        assert args.verbose is False

    def test_flag_true(self):
        parser = argparse.ArgumentParser()
        add_verbose_args(parser)
        args = parser.parse_args(['--verbose'])
        assert args.verbose is True

    def test_short_flag(self):
        parser = argparse.ArgumentParser()
        add_verbose_args(parser)
        args = parser.parse_args(['-v'])
        assert args.verbose is True
