import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_bias_response_summary import summarize


def test_matched_summary_and_audit():
    modes=('original','raw','raw_canonical','canonical','augment')
    model={m:{'response_mse':1.,'forward_nse':2.} for m in modes}
    record={'valid':True,'initial_nse':0.,'context_error':0.,'observation_error':0.,'frame_mismatch':False,'models':model}
    groups=[{'epsilon':eps,'records':[record],'valid_count':1,'summary':model} for eps in (.05,.2)]
    d={'seed':1,'fit_seed':7,'checkpoint_sha256':'c','source_models':{},'script_sha256':'s','scale_probe_sha256':'p',
       'counted_simulator_transitions':10,'groups':groups}
    d2=copy.deepcopy(d);d2['seed']=2
    assert summarize([d,d2])['results']['0.05']['valid_count']==2
    d2['groups'][0]['records'][0]['initial_nse']=1.
    with pytest.raises(ValueError,match='invalid pair'):summarize([d,d2])
