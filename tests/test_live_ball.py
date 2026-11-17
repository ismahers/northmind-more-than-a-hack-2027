"""Constructed binary fixtures only: no files, models or datasets."""

from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np

from possession.live_ball import decode_live_probability, fit_binary_prior


class LiveBallTests(unittest.TestCase):
    @staticmethod
    def view(alive, time=None, run=None, **extra):
        n = len(alive)
        return SimpleNamespace(
            alive=np.array(alive),
            time=np.arange(n) / 10 if time is None else np.array(time),
            run=np.zeros(n, dtype=int) if run is None else np.array(run),
            **extra,
        )

    @staticmethod
    def prior(matrix=((.98, .02), (.02, .98))):
        return {"transition_probabilities": np.array(matrix, dtype=float)}

    def decode(self, p, prior=None, time=None, run=None, **kwargs):
        n = len(p)
        return decode_live_probability(
            p, np.arange(n) / 10 if time is None else time,
            np.zeros(n, dtype=int) if run is None else run,
            self.prior() if prior is None else prior, **kwargs,
        )

    def test_prior_counts_unknown_gap_run_and_view_boundaries(self):
        a = self.view([0, 1, -1, 0, 0, 1, 1, 0],
                      time=[0, .1, .2, .3, .4, .8, 0, .1],
                      run=[0, 0, 0, 0, 0, 0, 1, 1])
        fitted = fit_binary_prior([a, self.view([1, 1])])
        np.testing.assert_array_equal(fitted["transition_counts"], [[2, 2], [2, 2]])
        self.assertEqual(fitted["observed_transition_count"], 4)
        self.assertEqual(fitted["state_order"], ("D", "L"))
        np.testing.assert_array_equal(fitted["transition_probabilities"], np.full((2, 2), .5))

    def test_prior_ignores_home_away_swaps(self):
        rng = np.random.default_rng(307)
        alive = rng.integers(-1, 2, 100)
        home_away = rng.integers(0, 2, 100)
        a = fit_binary_prior([self.view(alive, labels=home_away)])
        b = fit_binary_prior([self.view(alive, labels=1 - home_away)])
        np.testing.assert_array_equal(a["transition_counts"], b["transition_counts"])
        np.testing.assert_array_equal(a["transition_probabilities"], b["transition_probabilities"])

    def test_prior_requires_observed_pair(self):
        for views in ([], [self.view([])], [self.view([0])],
                      [self.view([0, -1, 1])], [self.view([0]), self.view([1])]):
            with self.subTest(views=views), self.assertRaises(ValueError):
                fit_binary_prior(views)

    def test_exact_score_matches_brute_force(self):
        rng = np.random.default_rng(19)
        for n in range(1, 8):
            for _ in range(8):
                matrix = rng.uniform(.01, 1, (2, 2))
                matrix /= matrix.sum(axis=1, keepdims=True)
                p = rng.uniform(.001, .999, n)
                log_emit = np.column_stack((np.log1p(-p), np.log(p)))

                def score(path):
                    value = -np.log(2) + sum(log_emit[i, s] for i, s in enumerate(path))
                    return value + .5 * sum(np.log(matrix[path[i - 1], path[i]])
                                             for i in range(1, n))

                out = self.decode(p, self.prior(matrix)).astype(int)
                optimal = max(score(path) for path in product((0, 1), repeat=n))
                self.assertAlmostEqual(score(out), optimal, places=12)

    def test_evidence_persists_but_changes_for_sustained_observations(self):
        p = np.r_[np.full(5, .95), .45, np.full(5, .95), np.full(8, .01)]
        out = self.decode(p)
        self.assertTrue(np.all(out[:11]))
        self.assertFalse(np.any(out[11:]))

    def test_gap_and_run_reset_equal_independent_decodes(self):
        a, b = [.99, .99, .99], [.49, .49]
        expected = np.r_[self.decode(a), self.decode(b)]
        self.assertTrue(np.all(self.decode(a + b)))
        for time, run in (([0, .1, .2, 1, 1.1], [0] * 5),
                          ([0, .1, .2, 0, .1], [0, 0, 0, 1, 1])):
            np.testing.assert_array_equal(self.decode(a + b, time=time, run=run), expected)

    def test_ties_and_extreme_probabilities(self):
        np.testing.assert_array_equal(self.decode([.5] * 4), [False] * 4)
        uniform = self.prior(((.5, .5), (.5, .5)))
        # At the second frame both predecessors tie: prefer staying live.
        np.testing.assert_array_equal(self.decode([.5, .9], uniform), [True, True])
        np.testing.assert_array_equal(self.decode([0, 1], uniform), [False, True])
        self.assertEqual(self.decode([]).dtype, np.bool_)

    def test_inputs_are_not_mutated(self):
        view = self.view([0, 0, 1, 1])
        copies = [x.copy() for x in (view.alive, view.time, view.run)]
        prior = fit_binary_prior([view])
        matrix = prior["transition_probabilities"].copy()
        p = np.array([0, .3, .7, 1.])
        original = p.copy()
        decode_live_probability(p, view.time, view.run, prior)
        for before, after in zip(copies, (view.alive, view.time, view.run)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(matrix, prior["transition_probabilities"])
        np.testing.assert_array_equal(original, p)

    def test_invalid_prior_probabilities_clocks_and_configuration(self):
        invalid_priors = ({}, self.prior([[1, 0], [.2, .8]]),
                          self.prior([[.8, .8], [.2, .8]]),
                          self.prior([[np.nan, .2], [.2, .8]]),
                          self.prior([.5, .5]),
                          {**self.prior(), "state_order": ("L", "D")})
        for prior in invalid_priors:
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                self.decode([.5, .5], prior)
        for p in ([np.nan], [np.inf], [-.01], [1.01], [[.5]]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.decode(p)
        for time, run in (([0, 0], [0, 0]), ([1, 0], [0, 0]),
                          ([0, np.nan], [0, 0]), ([0], [0, 0]),
                          ([0, .1], [0, .2]), ([0, .1], [0, np.inf])):
            with self.subTest(time=time, run=run), self.assertRaises(ValueError):
                self.decode([.5, .5], time=time, run=run)
        for fps in (0, -1, np.nan, np.inf, True):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.decode([.5], fps=fps)
        for strength in (0, 1, np.nan, True):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                self.decode([.5], strength=strength)
        for alive in ([2], [np.nan], [[0, 1]]):
            with self.subTest(alive=alive), self.assertRaises(ValueError):
                fit_binary_prior([self.view(alive)])


if __name__ == "__main__":
    unittest.main()
