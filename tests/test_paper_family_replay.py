"""Archived biological examples replayed through the public transfer command.

These are numerical regression fixtures, not an independent accuracy estimate.
"""
from pathlib import Path
import pytest
from fiberhmm.inference.consensus.transfer_cli import main
from fiberhmm.inference.consensus.artifacts import read_json

ROOT=Path(__file__).parent/'fixtures/paper_families'
CASES=sorted(ROOT.glob('*/evidence_*.json.gz'))

@pytest.mark.parametrize('evidence',CASES,ids=lambda p:p.parent.name+'_'+p.name)
def test_archived_native_scores_and_genomic_coordinates(evidence,tmp_path):
    expected=read_json(evidence.with_name(evidence.name.replace('evidence_','expected_').replace('.json.gz','.json')))
    main(['--models',str(evidence.parent/'models.json.gz'),'--evidence',str(evidence),'--output',str(tmp_path),'--no-bam'])
    actual=read_json(tmp_path/'window_00001.json.gz')
    index={(r['dataset'],r['unit_id'],tuple(r['interval'])):r for r in actual['rows']}
    for row in expected['rows']:
        found=index[row['dataset'],row['unit_id'],tuple(row['interval'])]
        assert found['genomic_interval']==row['genomic_interval']
        scores={s['family']:s for s in found['scores']}
        for previous in row['scores']:
            score=scores[previous['family']]
            assert score['status']==previous['status']
            assert score['compatible']==previous['compatible']
            assert score.get('simulations',0)==previous.get('simulations',0)
            if previous.get('tail_interval') is not None:
                assert score['predictive_tail_interval']==pytest.approx(previous['tail_interval'],abs=1e-12)
