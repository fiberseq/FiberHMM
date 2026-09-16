from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.artifacts import read_json,write_json
from fiberhmm.inference.consensus.cli import main as consensus
from fiberhmm.inference.consensus.transfer_cli import main
from fiberhmm.inference.consensus.transfer import load_bundle,score_payload


def test_public_fit_freeze_transfer_no_refitting(tmp_path,monkeypatch):
    from test_staged_families import payload,parameters
    from fiberhmm.inference.consensus.harmonized_families import workflow
    data=payload()
    for i,u in enumerate(data['strata'][0]['units']):u['read_name']=f'train/{i}/ccs'
    data['pooling']={'windows':[dict(chrom='chr1',start=0,end=81,strand='+',name='source')]}
    write_json(tmp_path/'evidence.json.gz',data);write_json(tmp_path/'params.json',parameters(physical_radius_bp=5))
    consensus(['--evidence',str(tmp_path/'evidence.json.gz'),'--parameters',str(tmp_path/'params.json'),'--output',str(tmp_path/'fit')])
    def forbidden(*a,**k):raise AssertionError('target refitting forbidden')
    monkeypatch.setattr(workflow,'fit_source',forbidden)
    main(['--freeze-run',str(tmp_path/'fit'),'--output',str(tmp_path/'models')])
    model=tmp_path/'models/frozen_models.json.gz';bundle=load_bundle(model)
    excluded=score_payload(bundle,data)
    assert len(excluded['exclusions'])==6 and not excluded['rows']
    target=deepcopy(data)
    for i,u in enumerate(target['strata'][0]['units']):u['read_name']=f'target/{i}/ccs'
    write_json(tmp_path/'target.json.gz',target)
    main(['--models',str(model),'--evidence',str(tmp_path/'target.json.gz'),'--no-bam','--output',str(tmp_path/'scored')])
    result=read_json(tmp_path/'scored/window_00001.json.gz')
    assert not result['model_refitted'] and len(result['rows'])==6
    assert any(r['compatible'] for r in result['rows'])
    assert (tmp_path/'scored/families.tsv').is_file()
    broken=read_json(model);broken['region'][1]+=1;write_json(tmp_path/'broken.json',broken)
    with pytest.raises(ValueError,match='digest'):load_bundle(tmp_path/'broken.json')


def test_bam_bed_transfer_nonzero_origin_reverse_and_chip(tmp_path):
    from test_consensus_bed_cli import make_bam
    from fiberhmm.inference.consensus.transfer import load_bundle
    import pysam
    model=__import__('pathlib').Path(__file__).parent/'fixtures/paper_families/gata_tal1/models.json.gz'
    bam=tmp_path/'reads.bam';make_bam(bam)
    bed=tmp_path/'windows.bed';bed.write_text('chr1\t0\t300\tplus\t0\t+\nchr1\t0\t300\tminus\t0\t-\n')
    chip=tmp_path/'chip.bed';chip.write_text('chr1\t150\t170\tpeak\nchr1\t180\t190\tpeak\n')
    main(['--models',str(model),'--bam',str(bam),'--bed',str(bed),'--chip-bed',str(chip),'--output',str(tmp_path/'out')])
    first=read_json(tmp_path/'out/window_00001.json.gz');second=read_json(tmp_path/'out/window_00002.json.gz')
    assert first['model_sha256']==load_bundle(model)['content_sha256']
    assert second['model_refitted'] is False
    assert (tmp_path/'out/chip_evaluation.json').exists()
    assert (tmp_path/'out/families.svg').exists()
    exported=read_json(tmp_path/'out/bams/bam_exports.json')['outputs']
    with pysam.AlignmentFile(exported[0]['bam'],'rb') as handle:assert len(list(handle))==1
