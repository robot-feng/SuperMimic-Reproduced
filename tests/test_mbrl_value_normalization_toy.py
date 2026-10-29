import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'analysis'))
from mbrl_value_normalization_toy import gae, grad, objective, scalar, transform


class ValueNormalizationTests(unittest.TestCase):
    def test_discounted_survival_and_gae_identities(self):
        rng = random.Random(913)
        for horizon in (1, 2, 8, 32):
            for lam in (0., .4, 1.):
                rewards = [rng.uniform(-3, 3) for _ in range(horizon)]
                deaths = [rng.choice([0., 1., rng.random()]) for _ in range(horizon)]
                values = [rng.uniform(-4, 4) for _ in range(horizon+1)]
                gamma, mu, sigma = .97, -1.2, 2.3
                rn, vn = transform(rewards, deaths, values, gamma, mu, sigma)
                self.assertAlmostEqual(objective(rn, deaths, vn[-1], gamma),
                    (objective(rewards, deaths, values[-1], gamma)-mu)/sigma, places=11)
                for actual, expected in zip(gae(rn,deaths,vn,gamma,lam), gae(rewards,deaths,values,gamma,lam), strict=True):
                    self.assertAlmostEqual(actual, (expected-mu)/sigma, places=11)

    def test_consistent_actor_gradient_and_naive_difference(self):
        for theta in (-.3, 0., .5):
            self.assertAlmostEqual(scalar(theta,'raw',.99,2.,2.5),scalar(theta,'consistent',.99,2.,2.5), places=12)
            self.assertAlmostEqual(grad(theta,'raw',.99,2.,2.5),grad(theta,'consistent',.99,2.,2.5), places=8)
        self.assertGreater(abs(grad(.2,'raw',.99,2.,2.5)-grad(.2,'terminal_only',.99,2.,2.5)), .01)

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            transform([1.], [0.], [1., 2.], .99, 0., 0.)


if __name__ == '__main__':
    unittest.main()
