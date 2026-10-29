import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_bias_actor_summary import summarize


def test_pair_comparison_and_reject_bad_baseline():
    clean={'reward_plus':2.,'reward_delta_vs_base':1.,'failed_plus':False}
    candidate={'reward_plus':2.5,'reward_delta_vs_base':1.5,'failed_plus':False}
    row={'index':0,'frame':9,'clip':'b','base_reward':1.,'base_control_reward':1.,'base_failed':False,
         'control_failed':False,'clean_gradient':clean,'noise_gradient':candidate}
    section={'valid_pairs':1,'excluded_timeouts':0,'base_control_max_abs_reward_difference':0.,
             'base_control_max_abs_bootstrap_difference':0.,'base_control_failure_disagreements':0,'pairs':[row]}
    d={'seed':1,'fit_seed':7,'checkpoint_sha256':'c','source_models':{},'script_sha256':'s','scale_probe_sha256':'p',
       'horizon':8,'eval_horizon':80,'eps':.005,'gradient_bias_rms':.01,'gradient_input_observation_error':0.,
       'gradient_input_goal_error':0.,'gradient_eligible_indices':[0,1],'counted_simulator_transitions':1650,
       'gradient_clips':['a'],'evaluation_clips':['b'],'evaluation_frames':[9],'gradient_cosine':.9,
       'direction_mapping':{'clean_gradient':'raw_refit_canonical_input','noise_gradient':'canonical_refit_canonical_input'},
       'results':{'sigma':.1,'eligible_pairs':1,'action_checks':{'calls':80,'max_clean_error':0,'max_executed_error':0},
                  'short':section,'long':copy.deepcopy(section)}}
    d2=copy.deepcopy(d);d2['seed']=2
    assert summarize([d,d2])['results']['long']['mean_reward_difference']==pytest.approx(.5)
    d2['results']['long']['pairs'][0]['base_control_reward']=3.
    with pytest.raises(ValueError,match='base mismatch'):summarize([d,d2])
