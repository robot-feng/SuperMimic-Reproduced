"""Streaming independent-pilot full-vector gradient blending diagnostics."""
from __future__ import annotations
import torch
from g1_score_vector_moments import VectorMoments


class BlendMoments:
    def __init__(self):self.moments=VectorMoments()

    def add(self,real,model):
        if real.shape!=model.shape:raise ValueError('gradient shape mismatch')
        self.moments.add(real-model,model)

    def fit(self):
        m=self.moments
        if m.n<2:raise ValueError('need at least two pilot samples')
        var=m.m2_r/(m.n-1);cov=m.cross/(m.n-1)
        # Double dot avoids precision loss when bias² nearly cancels noise/n.
        d=m.mean_r.double();bias_raw=float(torch.dot(d,d))-var/m.n
        bias=max(0.,bias_raw);denom=bias+var
        alpha=max(0.,min(1.,(bias-cov)/denom)) if denom>1e-24 else 0.
        return {'count':m.n,'bias_squared_raw':bias_raw,'bias_squared_clipped':bias,
                'difference_variance_trace':var,'difference_model_covariance_trace':cov,
                'model_variance_trace':m.m2_z/(m.n-1),'coefficient':alpha}


def paired_risk_difference(difference,model,target,alpha):
    """Compare squared distances to independent target without catastrophic subtraction."""
    if difference.ndim!=1 or model.shape!=difference.shape or target.shape!=difference.shape or not 0<=alpha<=1:
        raise ValueError('invalid risk input')
    d=difference.double();residual=model.double()-target.double()
    dot=float(torch.dot(d,residual));norm=float(torch.dot(d,d))
    return {'blend_minus_model':2*alpha*dot+alpha*alpha*norm,
            'score_minus_model':2*dot+norm}
