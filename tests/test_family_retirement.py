from copy import deepcopy
import pytest
from fiberhmm.inference.consensus.harmonized_families.reference.family_retirement import retention_checks


def fixture():
    rows=[];calls=[];scores={}
    for i in range(8):
        uid=f'u{i}';old=i<4;new=i>=2
        row=dict(unit_id=uid,interval=[10,40],display_hypotheses=['short'] if old else [],
            parent_evaluations=[dict(hypothesis='short',compatible=old)])
        rows.append(row);calls.append(dict(unit_id=uid,start=10,end=40,evidence_group_id=uid))
        scores[(uid,10,40)]=[dict(hypothesis='parent',compatible=new)]
    return dict(calls=calls),dict(records=rows),{'short':['parent']},scores


def test_bidirectional_recurrence_retains_without_mutating_inputs():
    data=fixture();before=deepcopy(data)
    check=retention_checks(*data)['short']
    assert check['retain']
    assert check['alternative_only_groups']==['u0','u1']
    assert check['replacement_only_groups']['parent']==['u4','u5','u6','u7']
    assert data==before


@pytest.mark.parametrize('kind',['missing','unassessed','warning','compatible'])
def test_missing_or_ambiguous_is_not_discriminating(kind):
    data=fixture()
    for uid in ['u0','u1']:
        e=data[3][(uid,10,40)][0]
        if kind=='missing':data[3][(uid,10,40)]=[]
        elif kind=='unassessed':e['compatible']=None
        elif kind=='warning':e['fit_warning']=True
        else:e['compatible']=True
    assert not retention_checks(*data)['short']['retain']


def test_duplicates_do_not_create_recurrence():
    data=fixture();data[0]['calls'][1]['evidence_group_id']='u0'
    assert not retention_checks(*data)['short']['retain']


def test_no_reverse_witnesses_no_two_class_claim():
    data=fixture()
    for row in data[1]['records'][4:]:row['parent_evaluations'][0]['compatible']=None
    assert not retention_checks(*data)['short']['retain']


def test_a_directly_compatible_replacement_prevents_redundant_retention():
    data=fixture();data[2]['short'].append('other')
    for scores in data[3].values():scores.append(dict(hypothesis='other',compatible=True))
    assert not retention_checks(*data)['short']['retain']


def test_legacy_and_stricter_settings():
    data=fixture()
    assert not retention_checks(*data,minimum_groups=0)['short']['retain']
    assert not retention_checks(*data,minimum_groups=3)['short']['retain']


def test_same_physical_molecule_cannot_witness_both_directions():
    data=fixture();data[0]['calls'][4]['evidence_group_id']='u0'
    check=retention_checks(*data)['short']
    assert check['conflicting_physical_groups']==['u0']
    assert not check['retain']


def test_retained_alternatives_do_not_renominate_the_main_catalog():
    from fiberhmm.inference.consensus.harmonized_families.reference.refine_consensus_parents import nominate_refinements
    from fiberhmm.inference.consensus.harmonized_families.reference.native_support_nomination import nominate_supported_unions
    # Missing geometry is deliberate: retained alternatives must be excluded
    # before nomination inspects fitted edges or loads any cached model.
    annotation=dict(radius=5,hypotheses=[dict(id='kept',display=True,kind='bounded_parent',
        fit_warning=False,status='retained_informative_alternative')])
    case=dict(models=[],ledger=[])
    assert nominate_refinements(case,annotation)==[]
    def forbidden(_):raise AssertionError('Retained alternative entered cached nomination')
    assert nominate_supported_unions(case,annotation,forbidden,physical_support=True)==[]
