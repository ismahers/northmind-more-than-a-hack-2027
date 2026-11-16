"""Synthetic portability fixtures only; never open real models, data or results."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from possession import st_tcn_target as target
from scripts import submission_assets as assets

FIXTURES = Path(__file__).resolve().parents[1] / "evaluation/reports/final-delivery/test-fixtures"


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=FIXTURES)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        if os.name == "nt":
            # The authorized fixture directory plus replicated repository paths
            # exceeds legacy MAX_PATH on Windows; no system temp directory.
            self.root = Path("\\\\?\\" + str(self.root.resolve()))
            import shutil
            self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def make_registration(self, style):
        external = self.root / "evaluation/reports/external"
        fusion = external / "scale-selective-v1"
        teachers = external / "temporal-teachers-v1"
        probe = external / "temporal-probe-v1"
        def key(relative):
            if style == "windows":
                return "C:\\historical\\repository\\" + relative.replace("/", "\\")
            if style == "posix":
                return "/different/historical/repository/" + relative
            return relative.replace("/", "\\")
        source = self.write("possession/phase7/example.py", b"sealed source\n")
        registration = {"protocol": {"pilots": [0, 6]}, "probe_protocol": {},
                        "sources": {key(source.relative_to(self.root).as_posix()): assets._digest(source)},
                        "references": {}}
        for fold in (0, 6):
            relatives = [f"temporal-teachers-v1/training/fold-{fold}/seed-{seed}/binary/last.pt"
                         for seed in (2027, 2028)]
            relatives += [f"temporal-probe-v1/training/fold-{fold}/{arm}/last.pt"
                          for arm in ("local_classification", "multiscale_classification")]
            for relative in relatives:
                path = self.write("evaluation/reports/external/" + relative, b"synthetic checkpoint")
                registration["references"][key(path.relative_to(self.root).as_posix())] = assets._digest(path)
        reg_path = self.write("evaluation/reports/external/scale-selective-v1/registration.json",
                              json.dumps(registration).encode())
        digest = assets._digest(reg_path)
        seal = {"test_used": False, "registration_sha256": digest, "selections": {}}
        for fold in (0, 6):
            selection = {"test_used": False, "registration_sha256": digest,
                         "arm_gates": {"multiseed_scale_selective": {"passed": True}},
                         "teacher": {"seed": 2027}, "alt_teacher": {"seed": 2028},
                         "selected_params": {"multiseed_scale_selective": {}}}
            path = self.write(f"evaluation/reports/external/scale-selective-v1/selections/fold-{fold}.json",
                              json.dumps(selection).encode())
            seal["selections"][str(fold)] = assets._digest(path)
        self.write("evaluation/reports/external/scale-selective-v1/selection-seal.json", json.dumps(seal).encode())
        return registration, reg_path, fusion, teachers, probe

    def spec(self, fixture):
        _, _, fusion, teachers, probe = fixture
        with patch.multiple(target, ROOT=self.root, FUSION_DIR=fusion, TEACHERS_DIR=teachers, PROBE_DIR=probe):
            return target._transfer_spec()

    def test_windows_posix_and_relative_names_relocate_without_rewriting_seals(self):
        for style in ("windows", "posix", "relative"):
            with self.subTest(style=style):
                fixture = self.make_registration(style)
                before = fixture[1].read_bytes()
                _, _, members, dependencies = self.spec(fixture)
                self.assertEqual(len(members), 2)
                self.assertTrue(all(str(member["teacher"].resolve()) in dependencies for member in members))
                self.assertEqual(fixture[1].read_bytes(), before)

    def test_changed_checkpoint_is_rejected(self):
        fixture = self.make_registration("windows")
        (fixture[3] / "training/fold-0/seed-2027/binary/last.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            self.spec(fixture)

    def test_traversal_unknown_paths_and_ambiguous_markers_are_rejected(self):
        invalid = ["evaluation/reports/external/temporal-teachers-v1/../secret.pt",
                   "evaluation/reports/external/unknown/last.pt", "data/target.pt",
                   "C:/evaluation/reports/old/evaluation/reports/external/temporal-teachers-v1/x.pt"]
        with patch.object(target, "ROOT", self.root):
            for name in invalid:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    target._registered_path(name)

    def test_colliding_reference_names_rejected_even_with_identical_hashes(self):
        fixture = self.make_registration("windows")
        registration, reg_path, fusion, _, _ = fixture
        original, digest = next(iter(registration["references"].items()))
        registration["references"][original.replace("\\", "/")] = digest
        reg_path.write_text(json.dumps(registration))
        # Reseal synthetic fixture only, to reach collision validation.
        seal = json.loads((fusion / "selection-seal.json").read_text())
        seal["registration_sha256"] = assets._digest(reg_path)
        (fusion / "selection-seal.json").write_text(json.dumps(seal))
        with self.assertRaisesRegex(ValueError, "Colliding"):
            self.spec(fixture)

    def archive(self, rows=None):
        content = b"synthetic public model"
        name = "evaluation/reports/external/temporal-teachers-v1/last.pt"
        rows = rows or [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]
        archive = self.root / "public.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for row in rows:
                bundle.writestr(row["path"], content)
            bundle.writestr("ARTIFACT-MANIFEST.json", json.dumps({"version": "northmind-public-handoff-v1", "files": rows}))
        return archive, dict(expected_sha256=assets._digest(archive), expected_bytes=archive.stat().st_size,
                             expected_count=len(rows)), name

    def test_public_archive_restore_preserves_existing_and_is_idempotent(self):
        archive, expected, name = self.archive()
        destination = self.root / "clone"
        result = assets.restore(archive, destination, **expected)
        self.assertEqual(result["restored"], 1)
        self.assertEqual(assets.restore(archive, destination, **expected)["restored"], 0)
        (destination / name).write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "differs"):
            assets.restore(archive, destination, **expected)
        self.assertEqual((destination / name).read_bytes(), b"different")

    def test_archive_bad_digest_and_traversal_rejected_before_write(self):
        archive, expected, _ = self.archive()
        expected["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            assets.restore(archive, self.root / "clone", **expected)
        bad = [{"path": "evaluation/reports/external/temporal-teachers-v1/../../escape.pt",
                "bytes": 22, "sha256": "0" * 64}]
        archive, expected, _ = self.archive(bad)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            assets.restore(archive, self.root / "clone", **expected)
        self.assertFalse((self.root / "clone").exists())

    def test_archive_colliding_member_names_rejected_before_write(self):
        name = "evaluation/reports/external/temporal-teachers-v1/last.pt"
        rows = [{"path": path, "bytes": 22, "sha256": "0" * 64}
                for path in (name, name.upper())]
        archive, expected, _ = self.archive(rows)
        with self.assertRaisesRegex(ValueError, "Colliding"):
            assets.restore(archive, self.root / "clone", **expected)
        self.assertFalse((self.root / "clone").exists())

    def test_long_destination_root_restores_with_plain_windows_path(self):
        archive, expected, name = self.archive()
        destination = self.root / ("long-fixture-" + "x" * 90)
        plain_destination = Path(str(destination).removeprefix("\\\\?\\")) if os.name == "nt" else destination
        result = assets.restore(archive, plain_destination, **expected)
        self.assertEqual(result["restored"], 1)
        self.assertEqual((destination / name).read_bytes(), b"synthetic public model")
        self.assertEqual(assets.restore(archive, plain_destination, **expected)["restored"], 0)


if __name__ == "__main__":
    unittest.main()
