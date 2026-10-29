"""Re-execute the project's objective functions with small NumPy tensors.

This checks the project formulas without importing PyTorch or starting a simulator.
It does not test PyTorch autograd or the live training pipeline.
"""
import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Tensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def clone(self):
        return self.copy().view(Tensor)


def tensor(data):
    return np.asarray(data, dtype=np.float64).view(Tensor)


class NumpyTorch:
    Tensor = Tensor

    @staticmethod
    def arange(n, device, dtype):
        return np.arange(n, dtype=dtype)

    @staticmethod
    def empty_like(x):
        return np.empty_like(x).view(Tensor)

    @staticmethod
    def no_grad():
        return lambda f: f


def project_formulas():
    path = Path(__file__).resolve().parents[1] / 'src/supermimic/imagination/objective.py'
    module = ast.parse(path.read_text(encoding='utf8'))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('policy_objective', 'gae_targets')]
    if len(functions) != 2:
        raise ValueError('project objective functions missing')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    extracted = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    scope = {'torch': NumpyTorch(), 'ImagRollout': object}
    exec(compile(extracted, str(path), 'exec'), scope)
    return scope['policy_objective'], scope['gae_targets']


class ProjectValueNormalizationTests(unittest.TestCase):
    def test_project_lambda_and_policy_objective(self):
        objective, gae = project_formulas()
        rng = np.random.default_rng(24)
        for horizon in (1, 8, 32):
            r = tensor(rng.normal(size=(horizon, 3)))
            d = tensor(rng.uniform(size=(horizon, 3)))
            d[0, 0], d[-1, 1] = 1., 0.
            survival = tensor(np.concatenate([np.ones((1, 3)), np.cumprod(1.-d, axis=0)], axis=0))
            v = tensor(rng.normal(size=(horizon+1, 3)))
            gamma, mu, sigma = .99, 2.1, 1.7
            rn = tensor((r-mu*(1.-gamma*(1.-d)))/sigma)
            vn = tensor((v-mu)/sigma)
            raw = objective(SimpleNamespace(rewards=r, survival=survival), v[-1], gamma)
            normalized = objective(SimpleNamespace(rewards=rn, survival=survival), vn[-1], gamma)
            self.assertAlmostEqual(float(normalized), float((raw-mu)/sigma), places=12)
            for lam in (0., .95, 1.):
                np.testing.assert_allclose(gae(rn,d,vn,gamma,lam), (gae(r,d,v,gamma,lam)-mu)/sigma, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
