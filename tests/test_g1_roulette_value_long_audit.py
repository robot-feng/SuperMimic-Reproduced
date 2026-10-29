import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_long_audit import reconstruct
from test_g1_roulette_value_audit import ledger


def test_clip_end_is_terminal_and_nonclip_timeout_is_rejected():
    d=ledger();d['rows'][0]['done']=torch.tensor([False,False]);d['rows'][0]['timeout']=torch.tensor([False,True])
    d['rows'][0]['clip_end']=torch.tensor([False,True]);d['rows'][1]['clip_end']=torch.tensor([False,False])
    out=reconstruct(d,.99,.99,horizon=2)
    assert out['roulette_target'].tolist()==pytest.approx([1.577,.2],abs=1e-6)
    assert out['roulette_clip_end'].tolist()==[False,True]
    assert out['roulette_failed'].tolist()==[False,False]
    d['rows'][0]['clip_end']=torch.tensor([False,False])
    with pytest.raises(ValueError,match='non-clip timeout'):reconstruct(d,.99,.99,horizon=2)
