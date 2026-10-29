"""Stream Euclidean and fixed-Jacobian covariance of paired gradient vectors."""
import math

import torch


class WeightedVectorMoments:
    def __init__(self, jacobian, chunk=262144):
        if jacobian.ndim != 1 or chunk < 1 or not torch.isfinite(jacobian).all():
            raise ValueError('invalid fixed Jacobian')
        self.jacobian = jacobian.detach()
        self.chunk = chunk
        self.n = 0
        self.mean_r = self.mean_z = None
        self.euclidean = [0., 0., 0.]
        self.weighted = [0., 0., 0.]

    def add(self, real, zero):
        if real.ndim != 1 or real.shape != zero.shape or real.shape != self.jacobian.shape:
            raise ValueError('invalid vector shape')
        if not torch.isfinite(real).all() or not torch.isfinite(zero).all():
            raise ValueError('nonfinite gradient')
        if self.n == 0:
            self.mean_r, self.mean_z = torch.zeros_like(real), torch.zeros_like(zero)
        self.n += 1
        factor = (self.n - 1)/self.n
        for start in range(0, len(real), self.chunk):
            sl = slice(start, start+self.chunk)
            dr = (real[sl] - self.mean_r[sl]).double()
            dz = (zero[sl] - self.mean_z[sl]).double()
            j = self.jacobian[sl].double()
            jr, jz = j*dr, j*dz
            for target, a, b in ((self.euclidean, dr, dz), (self.weighted, jr, jz)):
                target[0] += factor*float(a@a)
                target[1] += factor*float(b@b)
                target[2] += factor*float(a@b)
            self.mean_r[sl].add_(dr.to(real.dtype), alpha=1/self.n)
            self.mean_z[sl].add_(dz.to(zero.dtype), alpha=1/self.n)

    def summary(self):
        if self.n < 2:
            raise ValueError('at least two samples required')
        result = {}
        for name, values in [('euclidean', self.euclidean), ('weighted', self.weighted)]:
            vr, vz, cov = (x/(self.n-1) for x in values)
            if not all(math.isfinite(v) for v in (vr, vz, cov)) or vr <= 0 or vz <= 0:
                raise ValueError('degenerate covariance')
            result[name] = {'real_variance': vr, 'zero_variance': vz, 'cross_covariance': cov,
                            'fitted_coefficient': -cov/vz}
        return {'count': self.n, **result}


def validate(fit, test):
    """Apply independently fitted coefficients to held-out covariance."""
    candidates = {k: max(-1., min(1., fit[k]['fitted_coefficient'])) for k in ('euclidean', 'weighted')}
    output = {}
    for metric in ('euclidean', 'weighted'):
        m = test[metric]
        variance = lambda c: m['real_variance'] + 2*c*m['cross_covariance'] + c*c*m['zero_variance']
        output[metric] = {'base_variance': m['real_variance'],
                          'euclidean_coefficient_variance_ratio': variance(candidates['euclidean'])/m['real_variance'],
                          'weighted_coefficient_variance_ratio': variance(candidates['weighted'])/m['real_variance'],
                          'unit_coefficient_variance_ratio': variance(1.)/m['real_variance']}
    return {'pilot_coefficients': candidates, 'heldout': output}
