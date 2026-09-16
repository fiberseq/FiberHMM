from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.harmonized_families.evidence import intern,expand,REF


def test_storage_roundtrip_deduplicates_without_changing_evidence():
    shared={'scores':[dict(family='family_'+str(i),loss=i/10,status='scored') for i in range(100)]}
    original={'original':shared,'parents':[shared,shared],'status':'multi_compatible'}
    before=deepcopy(original);pool={};packed=intern(original,pool);size=len(pool)
    assert expand(packed,pool)==original==before
    assert intern(original,pool)==packed and len(pool)==size
    assert REF in packed['original']['scores']


def test_missing_tampered_or_excessive_expansion_fails_closed():
    pool={};packed=intern({'scores':list(range(300))},pool)
    with pytest.raises(ValueError,match='Missing'):expand(packed,{})
    key=packed['scores'][REF];corrupted=deepcopy(pool);corrupted[key]={'changed':True}
    with pytest.raises(ValueError,match='hash mismatch'):expand(packed,corrupted)
    with pytest.raises(ValueError,match='budget'):expand(packed,pool,maximum_nodes=20)
    with pytest.raises(ValueError,match='Malformed'):expand({REF:'id','extra':True},pool)
    with pytest.raises(ValueError,match='Reserved'):intern({REF:'already_encoded'},pool)
