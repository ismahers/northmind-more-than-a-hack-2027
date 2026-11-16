"""Synthetic serialization/contract tests; no target input, model or GT access."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.final_submission import _generate, serialize_submission


class SubmissionSerializationTests(unittest.TestCase):
    def test_exact_utf8_lf_bytes_and_zero_based_frames(self):
        expected = b"frame,label\n0,H\n1,A\n2,D\n3,X\n"
        self.assertEqual(serialize_submission(["H", "A", "D", "X"], 4), expected)

    def test_regeneration_is_byte_identical(self):
        labels = ["X", "H", "D", "A"] * 7
        self.assertEqual(serialize_submission(labels, len(labels)),
                         serialize_submission(list(labels), len(labels)))

    def test_rejects_wrong_counts(self):
        for labels in ([], ["H"], ["H", "A", "D"]):
            with self.subTest(count=len(labels)), self.assertRaises(ValueError):
                serialize_submission(labels, 2)

    def test_rejects_invalid_labels_without_normalizing(self):
        for label in ("h", " H", "H ", "", None, 0, "HOME", "H\nA"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                serialize_submission([label], 1)

    def test_generation_requires_cold_geometry_and_neural_provenance(self):
        import numpy as np

        labels = np.array(["X", "H", "D", "A"])
        loader = Mock(return_value=object())
        checks = {"uses_equivariant_st_tcn": True, "neural_cache_fingerprint": "synthetic",
                  "control_only_preserves_dead_mask": True}
        decoder = Mock(return_value=({"northmind_sequence": labels.copy()}, checks))
        modules = {
            "possession.inference": SimpleNamespace(predict=Mock(return_value=({"northmind_geometry": labels}, {}))),
            "possession.sequence_decoding": SimpleNamespace(predict_sequence=decoder),
            "possession.tactical_tagging": SimpleNamespace(load_team_prior=Mock(return_value=(np.zeros(4), {}))),
            "possession.tracking_features": SimpleNamespace(load_evidence=loader),
        }
        with patch.dict("sys.modules", modules), patch("scripts.final_submission.serialize_submission", return_value=b"synthetic"):
            result, _ = _generate(object(), object(), Mock())
            self.assertEqual(set(result), {"Northmind.csv", "Northmind-fallback.csv"})
            loader.assert_called_once_with(use_cache=False)
            checks["uses_equivariant_st_tcn"] = False
            with self.assertRaises(ValueError):
                _generate(object(), object(), Mock())


if __name__ == "__main__":
    unittest.main()
