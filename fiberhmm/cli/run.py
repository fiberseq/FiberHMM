#!/usr/bin/env python3
"""fiberhmm-run — removed.  Kept as a migration stub that points users to
the replacement: fiberhmm-call.

The old fiberhmm-run chained apply + recall-tfs + ft fire as separate
subprocess stages connected by pipes.  fiberhmm-call fuses the Python
stages in-process and is 2–9× faster on the same hardware.  There is no
meaningful use case where fiberhmm-run is preferable.
"""
import sys


_MESSAGE = """\
fiberhmm-run has been removed.  Use fiberhmm-call instead.

Old command:
  fiberhmm-run -i in.bam -o out.bam --enzyme hia5 --seq pacbio --fire -c 8

Equivalent with fiberhmm-call (sorted + indexed input):
  fiberhmm-call -i in.bam -o recalled.bam --enzyme hia5 --seq pacbio \\
                -c 8 --io-threads 16 --region-parallel --skip-scaffolds
  ft fire recalled.bam out.bam                 # only if you need FIRE

Equivalent with fiberhmm-call (unaligned or stdin input, pipe to FIRE):
  fiberhmm-call -i in.bam -o - --enzyme hia5 --seq pacbio \\
                -c 8 --io-threads 16 \\
    | ft fire - out.bam

Full docs: fiberhmm-call --help
"""


def main():
    sys.stderr.write(_MESSAGE)
    sys.exit(2)


if __name__ == '__main__':
    main()
