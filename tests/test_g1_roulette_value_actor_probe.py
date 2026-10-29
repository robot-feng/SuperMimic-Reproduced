import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_actor_probe import AffineCritic


def test_affine_value_and_survival_gradient_identity():
    class Critic(torch.nn.Module):
        def forward(self,x):return x.square()
    x=torch.tensor(.3,dtype=torch.float64,requires_grad=True);a=1.2;b=-.5
    v=Critic()(x);c=torch.sigmoid(x);g=.99**8
    old=torch.autograd.grad(g*c*v,x,retain_graph=True)[0]
    new=torch.autograd.grad(g*c*AffineCritic(Critic(),a,b)(x),x,retain_graph=True)[0]
    gcv=torch.autograd.grad(c*v,x,retain_graph=True)[0]
    gc=torch.autograd.grad(c,x)[0]
    assert float(new-old)==pytest.approx(float(g*((a-1)*gcv+b*gc)),abs=1e-12)
