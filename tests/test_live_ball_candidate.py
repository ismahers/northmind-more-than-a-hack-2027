"""Synthetic target candidate contracts; no assets, target GT, models or logs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession.live_ball_target import decode_target_candidate,recover_cached_live
from possession.tracking_features import Evidence
from possession.inference import Config
from possession import sequence_decoding as sequence
from scripts import live_ball_candidate as runner


def fixture(n=151):
    t = np.arange(n)/25
    evidence = Evidence(np.zeros(n,int),np.tile([.8,3.],(n,1)),
        np.column_stack((50+t,np.full(n,34.))))
    return dict(evidence=evidence,reference_labels=np.full(n,"H",dtype="<U1"),
        team_prior=np.zeros(n),quick_restart_mask=np.zeros(n,bool),restart_box_prior=np.zeros(n),
        neural_cache={"team_signal":np.zeros(n),"alive_prob":np.ones(n,np.float32),"fingerprint":"synthetic"},
        learned_live=np.ones(n,bool),blocks=[(1,0,n-1)])


def anchor_fixture():
    from possession.external_data.features import nearest_indices
    from possession.st_tcn_target import _native_signals
    native_time = np.arange(26)/25
    time10 = np.arange(11)/10
    match = SimpleNamespace(frame=np.arange(2,28),time=native_time,period=np.ones(26,int),
        metadata={"native_fps":25.})
    view = SimpleNamespace(time=time10,run=np.zeros(11,int),
        native_index=nearest_indices(native_time,time10))
    probability = np.linspace(.1,.9,11)
    alive = _native_signals(match,view,np.zeros(11),probability,30)[1]
    return match,view,{"alive_prob":alive}


def external_and_recipe():
    p0 = np.array([[.9,.1],[.05,.95]])
    p6 = np.array([[.8,.2],[.1,.9]])
    prior = lambda matrix:dict(state_order=["D","L"],transition_probabilities=matrix.tolist())
    external = dict(registration_sha256="a"*64,passes_both=False,automatic_promotion=False,
        transition_priors={"0":prior(p0),"6":prior(p6)})
    recipe = dict(version="live-ball-prior-v1",**prior(.5*p0+.5*p6),source_registration_sha256="a"*64,
        policy="equal_average_of_fold_0_and_6_transition_probabilities",
        selected_for_submission=False,validation_gates_passed=False)
    return external,recipe


def status_fixture():
    status = {key:False for key in runner._FLAGS}
    status["phase"] = "setup"
    return status


class LiveBallCandidateTests(unittest.TestCase):
    def test_baseline_legacy_parity_with_all_target_priors(self):
        f = fixture()
        n = len(f["reference_labels"])
        f["team_prior"][:] = np.linspace(-.8,.8,n)
        f["restart_box_prior"][:] = np.cos(np.arange(n)/30)*.5
        f["neural_cache"]["team_signal"][:] = np.sin(np.arange(n)/35)*.7
        old,_ = decode_target_candidate(**f)
        with patch.dict(sequence.GAMES,{sequence.GAME:{"n_video_frames":n,"pitch":(105.,68.)}}),patch(
                "possession.sequence_decoding.half_video_blocks",return_value=f["blocks"]),patch(
                "possession.sequence_decoding.get_quick_restart_mask",return_value=f["quick_restart_mask"]),patch(
                "possession.sequence_decoding.load_restart_and_box_prior",return_value=f["restart_box_prior"]),patch(
                "possession.sequence_decoding.load_cached_st_tcn_target",return_value=f["neural_cache"]):
            expected,_ = sequence.predict_sequence(f["evidence"],f["reference_labels"],f["team_prior"],
                Config(),sequence.SequenceConfig())
        np.testing.assert_array_equal(old,expected["northmind_sequence"])

    def test_baseline_legacy_parity_missing_dead_and_half_separators(self):
        f = fixture(303)
        f["blocks"] = [(1,1,150),(2,152,301)]
        f["reference_labels"][[0,151,302]] = "X"
        f["reference_labels"][70:90] = "D"
        f["evidence"].team_distance_m[70:90] = np.inf
        f["evidence"].ball_xy_m[70:90] = np.nan
        f["neural_cache"]["team_signal"][152:] = -.7
        old,new = decode_target_candidate(**f)
        with patch.dict(sequence.GAMES,{sequence.GAME:{"n_video_frames":303,"pitch":(105.,68.)}}),patch(
                "possession.sequence_decoding.half_video_blocks",return_value=f["blocks"]),patch(
                "possession.sequence_decoding.get_quick_restart_mask",return_value=f["quick_restart_mask"]),patch(
                "possession.sequence_decoding.load_restart_and_box_prior",return_value=f["restart_box_prior"]),patch(
                "possession.sequence_decoding.load_cached_st_tcn_target",return_value=f["neural_cache"]):
            expected,_ = sequence.predict_sequence(f["evidence"],f["reference_labels"],f["team_prior"])
        np.testing.assert_array_equal(old,expected["northmind_sequence"])
        self.assertTrue((new[[0,151,302]]=="X").all())

    def test_same_dead_mask_gives_identical_baseline_and_candidate(self):
        f = fixture()
        old,new = decode_target_candidate(**f)
        np.testing.assert_array_equal(old,new)

    def test_missing_dead_can_become_live_from_neural_probability(self):
        f = fixture()
        f["evidence"].team_distance_m[:] = np.inf
        f["evidence"].ball_xy_m[:] = np.nan
        f["reference_labels"][:] = "D"
        f["neural_cache"]["team_signal"][:] = .8
        old,new = decode_target_candidate(**f)
        self.assertTrue((old=="D").all())
        self.assertTrue((new=="H").all())

    def test_new_live_sign_fallback_and_tie_home(self):
        for signal,expected in ((-.8,"A"),(0.,"H")):
            f = fixture()
            f["evidence"].team_distance_m[:] = np.inf
            f["evidence"].ball_xy_m[:] = np.nan
            f["reference_labels"][:] = "D"
            f["neural_cache"]["team_signal"][:] = signal
            _,new = decode_target_candidate(**f)
            self.assertTrue((new==expected).all())

    def test_neural_dead_can_be_added_to_good_tracking(self):
        f = fixture()
        f["learned_live"][60:90] = False
        old,new = decode_target_candidate(**f)
        self.assertFalse((old=="D").any())
        np.testing.assert_array_equal(new=="D",~f["learned_live"])

    def test_geometric_dead_stays_protected(self):
        f = fixture()
        f["evidence"].ball_xy_m[:,0] = 999.
        f["reference_labels"][:] = "D"
        old,new = decode_target_candidate(**f)
        self.assertTrue((old=="D").all())
        np.testing.assert_array_equal(old,new)

    def test_quick_restart_suppresses_outside_only_as_current_recipe(self):
        f = fixture()
        f["evidence"].ball_xy_m[:,0] = 999.
        f["quick_restart_mask"][:] = True
        old,new = decode_target_candidate(**f)
        self.assertTrue((old=="H").all())
        np.testing.assert_array_equal(old,new)

    def test_pure_decoder_does_not_call_any_loader(self):
        f = fixture()
        with patch("possession.tracking_features.load_evidence",side_effect=AssertionError("loader")),patch(
                "possession.tactical_tagging.load_team_prior",side_effect=AssertionError("loader")),patch(
                "possession.st_tcn_target.load_cached_st_tcn_target",side_effect=AssertionError("loader")):
            decode_target_candidate(**f)

    def test_arrays_not_mutated(self):
        f = fixture()
        reference,prior,ball = f["reference_labels"].copy(),f["team_prior"].copy(),f["evidence"].ball_xy_m.copy()
        decode_target_candidate(**f)
        np.testing.assert_array_equal(reference,f["reference_labels"])
        np.testing.assert_array_equal(prior,f["team_prior"])
        np.testing.assert_array_equal(ball,f["evidence"].ball_xy_m)

    def test_invalid_masks_or_neural_values_fail(self):
        for key,value in (("learned_live",np.ones(151)),("quick_restart_mask",np.ones(151))):
            f = fixture()
            f[key] = value
            with self.assertRaises(ValueError):
                decode_target_candidate(**f)
        f = fixture()
        f["neural_cache"]["team_signal"][0] = np.nan
        with self.assertRaises(ValueError):
            decode_target_candidate(**f)

    def test_bad_half_boundaries_and_x_inside_half_rejected(self):
        for blocks in ([(1,0,151)],[(1,0,100),(2,100,150)]):
            f = fixture()
            f["blocks"] = blocks
            with self.assertRaises(ValueError):
                decode_target_candidate(**f)
        f = fixture()
        f["reference_labels"][50] = "X"
        with self.assertRaises(ValueError):
            decode_target_candidate(**f)

    def test_cached_anchor_roundtrip_is_exact_and_native_mask_boolean(self):
        match,view,cache = anchor_fixture()
        with patch("possession.live_ball.decode_live_probability",side_effect=lambda p,*a,**kw:p>.5):
            mask = recover_cached_live(match,view,cache,{},30)
        self.assertEqual(mask.dtype,np.bool_)
        self.assertEqual(mask.shape,(30,))
        self.assertFalse(mask[:2].any())
        self.assertFalse(mask[28:].any())

    def test_cached_anchor_roundtrip_rejects_nonreconstructable_native_value(self):
        match,view,cache = anchor_fixture()
        anchor_frames = set(match.frame[view.native_index])
        index = next(i for i in match.frame if i not in anchor_frames)
        cache["alive_prob"][index] = np.float32(.123456)
        with self.assertRaises(ValueError):
            recover_cached_live(match,view,cache,{},30)

    def test_anchor_reconstruction_rejects_wrong_precision_and_nan(self):
        match,view,cache = anchor_fixture()
        cache["alive_prob"] = cache["alive_prob"].astype(np.float64)
        with self.assertRaises(ValueError):
            recover_cached_live(match,view,cache,{},30)
        cache["alive_prob"] = cache["alive_prob"].astype(np.float32)
        cache["alive_prob"][0] = np.nan
        with self.assertRaises(ValueError):
            recover_cached_live(match,view,cache,{},30)

    def test_prior_must_equal_original_fixed_external_average(self):
        external,recipe = external_and_recipe()
        prior = runner.validate_prior_recipe(recipe,external)
        self.assertEqual(prior["state_order"],["D","L"])
        recipe["transition_probabilities"] = [[.7,.3],[.2,.8]]
        with self.assertRaises(ValueError):
            runner.validate_prior_recipe(recipe,external)

    def test_prior_does_not_allow_automatic_selection_or_false_pass_claim(self):
        for key in ("selected_for_submission","validation_gates_passed"):
            external,recipe = external_and_recipe()
            recipe[key] = True
            with self.assertRaises(ValueError):
                runner.validate_prior_recipe(recipe,external)

    def test_prior_requires_matching_external_registration(self):
        external,recipe = external_and_recipe()
        external["registration_sha256"] = "b"*64
        with self.assertRaises(ValueError):
            runner.validate_prior_recipe(recipe,external)

    def test_candidate_write_refuses_differing_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidate.csv"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                runner.atomic_candidate_write(path,b"different")
            self.assertEqual(path.read_bytes(),b"existing")

    def test_candidate_atomic_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidate.csv"
            runner.atomic_candidate_write(path,b"same")
            runner.atomic_candidate_write(path,b"same")
            self.assertEqual(path.read_bytes(),b"same")
            self.assertEqual([p.name for p in Path(directory).iterdir()],["candidate.csv"])

    def test_atomic_creation_race_preserves_other_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidate.csv"
            def occupied(source,destination):
                Path(destination).write_bytes(b"other-writer")
                raise FileExistsError()
            operation = "rename" if runner.os.name=="nt" else "link"
            with patch.object(runner.os,operation,side_effect=occupied):
                with self.assertRaises(FileExistsError):
                    runner.atomic_candidate_write(path,b"our-candidate")
            self.assertEqual(path.read_bytes(),b"other-writer")
            self.assertEqual([p.name for p in Path(directory).iterdir()],["candidate.csv"])

    def test_public_status_has_no_paths_labels_hashes_counts_or_messages(self):
        status = status_fixture()
        status.update(path="secret",sha256="secret",labels=["H"],count=10)
        safe = runner.safe_status(status,ValueError("private_input_data"))
        self.assertEqual(set(safe),{"phase","exception_class",*runner._FLAGS})
        self.assertNotIn("secret",json.dumps(safe))
        self.assertNotIn("private_input_data",json.dumps(safe))
        self.assertEqual(safe["exception_class"],"ValueError")

    def test_public_status_rejects_selection_and_unknown_phases(self):
        status = status_fixture()
        status["selected_for_submission"] = True
        with self.assertRaises(ValueError):
            runner.safe_status(status)
        status = status_fixture()
        status["phase"] = "private_input_path"
        with self.assertRaises(ValueError):
            runner.safe_status(status)

    def test_public_status_sanitizes_unknown_exception_classes(self):
        unknown = type("PrivateIdentity",(Exception,),{})
        self.assertEqual(runner.safe_status(status_fixture(),unknown("private"))["exception_class"],"Exception")

    def test_csv_serialization_is_original_official_format(self):
        from scripts.final_submission import serialize_submission
        self.assertEqual(serialize_submission(["H","D","A","X"],4),
            b"frame,label\n0,H\n1,D\n2,A\n3,X\n")


if __name__=="__main__":
    unittest.main()
