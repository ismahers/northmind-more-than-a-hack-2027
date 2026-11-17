"""Synthetic physical fixtures only: no files, datasets, models, or labels."""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from possession.control_memory import ControlMemoryConfig, temporal_control_memory


class ControlMemoryTests(unittest.TestCase):
    def fixture(self, fps=25, seconds=5):
        t = np.arange(int(seconds * fps) + 1) / fps
        return (
            np.tile([0.8, 3.0], (len(t), 1)),
            np.column_stack([t, np.zeros(len(t))]),
            t, np.zeros(len(t), dtype=int), np.ones(len(t), dtype=bool),
        )

    def extract(self, fixture, fps=25):
        return temporal_control_memory(*fixture, ControlMemoryConfig(fps=fps))

    def test_control_flight_reception_and_no_initial_memory(self):
        f = self.fixture()
        d, xy, t, _, _ = f
        xy[:, 0] = np.where(t <= 1, t, np.where(t < 2, 1 + 10 * (t - 1), 11 + t - 2))
        d[(t > 1) & (t < 2)] = [4, 4]
        d[t >= 2] = [3, 0.8]
        out = self.extract(f)
        self.assertFalse(out.supported[0])
        self.assertTrue(np.all(out.memory_signal[(t >= .4) & (t <= 1)] == 1))
        self.assertGreater(out.memory_signal[np.searchsorted(t, 1.8)], 0)
        self.assertEqual(out.memory_signal[np.searchsorted(t, 2.6)], -1)
        self.assertTrue(np.any(out.speed_m_s > 5.5))

    def test_rival_brief_touch_cannot_transfer_but_sustained_can(self):
        f = self.fixture()
        d, _, t, _, _ = f
        d[(t >= 1) & (t <= 1.08)] = [3, .8]
        d[(t > 1.08) & (t < 1.5)] = [4, 4]
        d[t >= 1.5] = [3, .8]
        out = self.extract(f)
        self.assertTrue(np.all(out.memory_signal[(t >= 1) & (t < 1.7)] > 0))
        self.assertEqual(out.memory_signal[np.searchsorted(t, 1.8)], -1)

    def test_missing_observations_only_decay_and_expire(self):
        f = self.fixture()
        d, xy, t, _, _ = f
        d[t >= 1] = np.inf
        xy[t >= 1] = np.nan
        out = self.extract(f)
        self.assertTrue(np.all(np.isnan(out.speed_m_s[t >= 1])))
        at = np.searchsorted(t, 2)
        self.assertAlmostEqual(out.memory_signal[at], 1 - (t[at] - .96) / 2)
        self.assertFalse(np.any(out.supported[t >= 3]))
        self.assertFalse(np.any(out.confirmed_control[t >= 1]))

    def test_no_live_span_resets_without_leakage(self):
        f = self.fixture()
        d, _, t, _, live = f
        live[(t >= 1) & (t < 1.2)] = False
        d[t >= 1] = [4, 4]
        out = self.extract(f)
        self.assertFalse(np.any(out.supported[t >= 1]))
        self.assertTrue(np.all(np.isnan(out.speed_m_s[~live])))

    def test_run_and_clock_gap_reset(self):
        for kind in ("run", "gap"):
            with self.subTest(kind=kind):
                f = self.fixture()
                d, _, t, run, _ = f
                boundary = np.searchsorted(t, 1)
                d[boundary:] = [4, 4]
                if kind == "run":
                    run[boundary:] = 1
                    t[boundary:] -= 1
                else:
                    t[boundary:] += .2
                out = self.extract(f)
                self.assertFalse(np.any(out.supported[boundary:]))
                self.assertTrue(out.segment_start[boundary])
                self.assertTrue(np.isnan(out.speed_m_s[boundary]))

    def test_swap_teams_and_reflect_coordinates(self):
        f = self.fixture()
        d, xy, t, run, live = f
        d[(t > 1) & (t < 2)] = [4, 4]
        d[t >= 2] = [.9 + 2, .9]
        original = self.extract(f)
        swapped = self.extract((d[:, ::-1], -xy, t, run, live))
        np.testing.assert_allclose(swapped.memory_signal, -original.memory_signal)
        np.testing.assert_allclose(swapped.speed_m_s, original.speed_m_s, equal_nan=True)
        np.testing.assert_array_equal(swapped.supported, original.supported)

    def test_fps_uses_elapsed_seconds_and_exact_speed_lag(self):
        confirmations = []
        for fps in (10, 25):
            f = self.fixture(fps=fps)
            d, xy, t, _, _ = f
            xy[:, 0] = 2 * t
            d[t >= 1] = [4, 4]
            out = self.extract(f, fps=fps)
            first = np.flatnonzero(out.confirmed_control)[0]
            confirmations.append(t[first])
            self.assertGreaterEqual(t[first], .32 - 1e-9)
            self.assertLessEqual(t[first], .32 + 1 / fps + 1e-9)
            np.testing.assert_allclose(out.speed_m_s[np.isfinite(out.speed_m_s)], 2)
            last_control = t[np.flatnonzero(out.confirmed_control)[-1]]
            at = np.searchsorted(t, 2)
            self.assertAlmostEqual(out.memory_signal[at], 1 - (t[at] - last_control) / 2)
        self.assertLessEqual(abs(confirmations[0] - confirmations[1]), .1)

    def test_causal_prefix_and_input_arrays_unchanged(self):
        f = self.fixture()
        copies = [a.copy() for a in f]
        full = self.extract(f)
        prefix = self.extract(tuple(a[:40] for a in f))
        np.testing.assert_array_equal(full.memory_signal[:40], prefix.memory_signal)
        for original, copied in zip(f, copies):
            np.testing.assert_array_equal(original, copied)

    def test_physical_candidate_gates_do_not_invent_control(self):
        for kind in ("distance", "margin", "speed", "missing_rival", "tie"):
            f = self.fixture()
            d, xy, t, _, _ = f
            if kind == "distance":
                d[:] = [1.81, 4]
            elif kind == "margin":
                d[:] = [.8, 1.54]
            elif kind == "speed":
                xy[:, 0] = 5.6 * t
            elif kind == "missing_rival":
                d[:, 1] = np.inf
            else:
                d[:] = [.8, .8]
            with self.subTest(kind=kind):
                out = self.extract(f)
                self.assertFalse(np.any(out.supported))
                self.assertFalse(np.any(out.candidate_team))

    def test_missing_coordinates_require_fresh_speed_history(self):
        f = self.fixture()
        _, xy, t, _, _ = f
        xy[25] = np.nan
        out = self.extract(f)
        self.assertTrue(np.all(np.isnan(out.speed_m_s[25:29])))
        self.assertTrue(np.all(out.confirmed_control[25:29] == 0))
        self.assertGreater(out.memory_signal[28], 0)
        self.assertTrue(np.isfinite(out.speed_m_s[29]))

    def test_invalid_config_and_frozen_config(self):
        for field in ControlMemoryConfig.__dataclass_fields__:
            for bad in (0, -1, np.inf, np.nan, True, "bad"):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        ControlMemoryConfig(**{field: bad})
        with self.assertRaises(FrozenInstanceError):
            ControlMemoryConfig().fps = 10

    def test_input_validation_and_empty(self):
        for index, replacement in (
            (0, np.ones((5, 2))), (1, np.ones(5)), (2, np.array([np.nan])),
            (3, np.zeros(2)), (4, np.ones(126, dtype=int)),
        ):
            f = list(self.fixture())
            f[index] = replacement
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.extract(f)
        for index, value in ((2, np.nan), (3, np.inf), (3, .5), (0, -1)):
            f = list(self.fixture())
            f[index] = f[index].astype(float)
            f[index][5] = value
            with self.assertRaises(ValueError):
                self.extract(f)
        f = self.fixture()
        f[2][5] = f[2][4]
        with self.assertRaises(ValueError):
            self.extract(f)
        empty = self.extract(tuple(a[:0] for a in self.fixture()))
        self.assertEqual(empty.memory_signal.shape, (0,))


if __name__ == "__main__":
    unittest.main()
