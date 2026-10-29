import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_audit import reconstruct


def ledger():
    a=torch.tensor([True,True]);b=torch.tensor([True,False]);z=torch.tensor([False,False])
    return {'initial_value':torch.tensor([1.,2.]),'cutoff':torch.tensor([2,1]),'rows':[
        {'reward':torch.tensor([.1,.2]),'current':torch.tensor([1.,2.]),'next_value':torch.tensor([1.1,0.]),
         'alive_before':a,'done':torch.tensor([False,True]),'timeout':z,'roulette_active':a,'fixed_active':a},
        {'reward':torch.tensor([.3,0.]),'current':torch.tensor([1.1,0.]),'next_value':torch.tensor([1.2,0.]),
         'alive_before':b,'done':z,'timeout':z,'roulette_active':b,'fixed_active':b}]}


def test_reconstructs_terminal_and_survivor_targets():
    out=reconstruct(ledger(),.99,.99,horizon=2)
    assert out['roulette_target'].tolist()==pytest.approx([1.577,.2],abs=1e-6)
    assert out['fixed_reward'].tolist()==pytest.approx([.397,.2],abs=1e-6)
    assert out['fixed_bootstrap'].tolist()==pytest.approx([.99**2*1.2,0.],abs=1e-6)
    assert out['roulette_failed'].tolist()==[False,True]


def test_rejects_bad_masks_and_missing_tail():
    d=ledger();d['rows'][1]['roulette_active']=torch.tensor([False,False])
    with pytest.raises(ValueError,match='active mask'):reconstruct(d,.99,.99,2)
    d=ledger();d['rows']=d['rows'][:1]
    with pytest.raises(ValueError,match='incomplete'):reconstruct(d,.99,.99,2)
