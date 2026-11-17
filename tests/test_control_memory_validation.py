"""Pure synthetic barriers and validation/export contracts; never load datasets."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession import control_memory_validation as validation
from possession.control_memory import ControlMemoryConfig
from dataclasses import asdict


def protocol():
    return dict(version="control-memory-validation-v1",part="validation",train=False,
        test_access=False,target_access=False,parameter_search=False,folds=[0,6],
        scenarios=list(validation.SCENARIOS),weights=[.5,.25,.25],
        views=[[False,False],[False,True],[True,False],[True,True]],
        cpu_threads=4,max_seconds=3600,memory_weight=.15,weak_strength_threshold=.25,
        neutral_tagging=True,guardrails=validation.GUARDS.copy(),fps=25.,
        memory=asdict(ControlMemoryConfig()),memory_smoothing="within_predicted_live_spans")


def metrics(value=.5):
    return {s: dict(spellF1=value,macroF1=.6,short_recall=.4,long_recall=.7,
        false_transitions_per_minute=1.) for s in validation.SCENARIOS}


def summary():
    folds = {}
    for fold in ("0","6"):
        base, candidate = metrics(), metrics(.51)
        folds[fold] = dict(metrics=dict(baseline=base,memory=candidate),
            gates=validation.gates(candidate,base),
            identity_parity={s:True for s in validation.SCENARIOS})
    return dict(folds=folds,registration_sha256="a"*64,seconds=1.,
        test_accessed=False,target_accessed=False,calibration_accessed=False,
        automatic_promotion=False)


def match_fixture(two_segments=False):
    n = 51
    t = np.arange(n)/25
    if two_segments:
        t = np.r_[t,t]
    count = len(t)
    return SimpleNamespace(metadata=dict(native_fps=25.,pitch_m=[105.,68.]),
        time=t,frame=np.arange(count),period=np.r_[np.ones(n,int),np.full(n,2,int)]
        if two_segments else np.ones(n,int),
        players=np.broadcast_to(np.array([[[.8,0.]],[[3.,0.]]]),(count,2,1,2)).copy(),
        player_available=np.ones((count,2,1),bool),
        ball=np.zeros((count,3)),ball_available=np.ones(count,bool))


def model_view(two_segments=False):
    t = np.arange(21)/10
    return SimpleNamespace(time=np.r_[t,t] if two_segments else t,
        run=np.r_[np.zeros(21,int),np.ones(21,int)] if two_segments else np.zeros(21,int))


class ValidationTests(unittest.TestCase):
    def test_fixed_protocol(self):
        validation.validate_protocol(protocol())

    def test_access_flags_fail_closed(self):
        for key in ("train","test_access","target_access","parameter_search"):
            p = protocol()
            p[key] = True
            with self.subTest(key=key),self.assertRaises(ValueError):
                validation.validate_protocol(p)

    def test_integer_false_is_not_access_authorization(self):
        p = protocol()
        p["target_access"] = 0
        with self.assertRaises(ValueError):
            validation.validate_protocol(p)

    def test_no_parameter_sweep_or_changed_memory(self):
        for key,value in (("memory_weight",.2),("max_seconds",3601),("folds",[0])):
            p = protocol()
            p[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validation.validate_protocol(p)
        p = protocol()
        p["memory"]["memory_horizon_s"] = 3
        with self.assertRaises(ValueError):
            validation.validate_protocol(p)

    def test_geometry_never_needs_labels(self):
        # This object has no alive, team, provenance or label attributes.
        native = validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21,bool))
        np.testing.assert_allclose(native.distances,np.tile([.8,3.],(51,1)))
        np.testing.assert_allclose(native.ball,np.tile([52.5,34.],(51,1)))
        self.assertEqual(len(native.time),51)

    def test_corruption_matches_exact_nearest_anchor_mask(self):
        view = model_view()
        hidden = np.zeros(21,bool)
        hidden[5:10] = True
        native = validation.build_native_geometry(match_fixture(),view,hidden)
        expected = hidden[native.anchor_index]
        np.testing.assert_array_equal(~np.isfinite(native.ball).all(1),expected)
        np.testing.assert_array_equal(~np.isfinite(native.distances).any(1),expected)

    def test_segment_mapping_does_not_cross_clock_restart(self):
        hidden = np.r_[np.zeros(21,bool),np.ones(21,bool)]
        native = validation.build_native_geometry(match_fixture(True),model_view(True),hidden)
        self.assertTrue(np.isfinite(native.ball[native.run==0]).all())
        self.assertFalse(np.isfinite(native.ball[native.run==1]).any())
        self.assertTrue((native.anchor_index[native.run==1]>=21).all())
        self.assertEqual(native.time[51],0.)

    def test_adapter_rejects_nonboolean_mask(self):
        with self.assertRaises(ValueError):
            validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21))

    def test_adapter_rejects_segment_mismatch(self):
        view = model_view()
        view.run[:] = 1
        with self.assertRaises(ValueError):
            validation.build_native_geometry(match_fixture(),view,np.zeros(21,bool))

    def test_adapter_inputs_are_unchanged(self):
        match,view = match_fixture(),model_view()
        before = match.ball.copy()
        hidden = np.ones(21,bool)
        validation.build_native_geometry(match,view,hidden)
        np.testing.assert_array_equal(match.ball,before)
        self.assertTrue(match.ball_available.all())

    def test_dead_mask_is_geometry_only_and_preserved(self):
        native = validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21,bool))
        # Entire outside-pitch run is D independently of opposite neural teams.
        ball = native.ball.copy()
        ball[:,0] = 110.
        outside = validation.NativeGeometry(native.time,native.run,native.native_index,
            native.anchor_index,ball,native.distances)
        for sign in (-1.,1.):
            old,new = validation.decode_pair(outside,np.full(51,sign),[105.,68.])
            self.assertTrue((old=="D").all())
            np.testing.assert_array_equal(old,new)

    def test_zero_memory_is_identical_ablation(self):
        native = validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21,bool))
        with patch("possession.control_memory.temporal_control_memory",return_value=
                SimpleNamespace(memory_signal=np.zeros(51),supported=np.zeros(51,bool))):
            old,new = validation.decode_pair(native,np.zeros(51),[105.,68.])
        np.testing.assert_array_equal(old,new)

    def test_strong_base_cannot_receive_memory(self):
        native = validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21,bool))
        with patch("possession.control_memory.temporal_control_memory",return_value=
                SimpleNamespace(memory_signal=-np.ones(51),supported=np.ones(51,bool))):
            old,new = validation.decode_pair(native,np.ones(51),[105.,68.])
        np.testing.assert_array_equal(old,new)

    def test_weak_unsmoothed_gate_adds_fixed_weight_only(self):
        native = validation.build_native_geometry(match_fixture(),model_view(),np.zeros(21,bool))
        weak = validation.NativeGeometry(native.time,native.run,native.native_index,
            native.anchor_index,native.ball,np.full((51,2),5.))
        captured = []
        def smoother(values,window):
            captured.append(np.array(values))
            self.assertEqual(window,5)
            return np.asarray(values)
        with patch("possession.control_memory.temporal_control_memory",return_value=
                SimpleNamespace(memory_signal=np.ones(51),supported=np.ones(51,bool))), patch(
                "possession.sequence_decoding._smooth_strength",side_effect=smoother):
            validation.decode_pair(weak,np.full(51,.2),[105.,68.])
        self.assertEqual(len(captured),3)  # geometry, baseline, candidate
        np.testing.assert_allclose(captured[2],.15)

    def test_smoothed_memory_cannot_cross_short_dead_barrier(self):
        reference = np.array(list("HHHDHHH"))
        first = validation.smooth_live_delta(np.array([.15,.15,.15,0,0,0,0]),reference,5)
        second = validation.smooth_live_delta(np.array([-.15,-.15,-.15,0,0,0,0]),reference,5)
        np.testing.assert_array_equal(first[3:],second[3:])
        self.assertTrue((first[3:]==0).all())

    def test_score_expansion_keeps_segment_separators_unknown(self):
        context = SimpleNamespace(gt=np.zeros(8,int),parts=[
            (np.array([0,1]),np.array([0,0,1]),slice(0,3)),
            (np.array([2]),np.array([0]),slice(7,8))])
        hard = validation.expand_hard(context,np.array(["H","D","A"]))
        np.testing.assert_array_equal(hard,[0,0,2,3,3,3,3,1])

    def test_gate_robust_requires_half_point_gain_per_fold(self):
        self.assertFalse(validation.gates(metrics(.504),metrics())["passed"])
        self.assertTrue(validation.gates(metrics(.506),metrics())["passed"])

    def test_each_clean_guard_can_block_promotion(self):
        for key in ("spellF1","short_recall","long_recall","macroF1",
                    "false_transitions_per_minute"):
            old,new = metrics(),metrics(.53)
            new["clean"][key] = old["clean"][key]+(.1 if key=="false_transitions_per_minute" else -.1)
            with self.subTest(key=key):
                self.assertFalse(validation.gates(new,old)["passed"])

    def test_public_export_whitelists_paths_and_free_strings(self):
        raw = summary()
        raw["input_path"] = "private_input_do_not_export"
        raw["exception"] = "untrusted_exception_do_not_export"
        safe = validation.safe_public_summary(raw)
        encoded = json.dumps(safe)
        self.assertNotIn("do_not_export",encoded)
        self.assertFalse(safe["full_target_recipe"])
        self.assertFalse(safe["automatic_promotion"])
        self.assertTrue(safe["passes_both"])

    def test_public_export_cannot_hide_a_failed_fold(self):
        raw = summary()
        row = raw["folds"]["6"]
        row["metrics"]["memory"] = metrics(.499)
        row["gates"] = validation.gates(row["metrics"]["memory"],row["metrics"]["baseline"])
        self.assertFalse(validation.safe_public_summary(raw)["passes_both"])

    def test_public_export_rejects_nonfinite_strings_and_bool_metrics(self):
        for value in (float("nan"),float("inf"),"private text",True):
            raw = summary()
            raw["folds"]["0"]["metrics"]["memory"]["clean"]["macroF1"] = value
            with self.subTest(value=type(value).__name__),self.assertRaises(ValueError):
                validation.safe_public_summary(raw)

    def test_public_export_rejects_target_access_incomplete_or_false_parity(self):
        raw = summary()
        raw["target_accessed"] = True
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)
        raw = summary()
        del raw["folds"]["6"]
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)
        raw = summary()
        raw["folds"]["0"]["identity_parity"]["clean"] = False
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_public_export_rejects_unexpected_metrics(self):
        raw = summary()
        raw["folds"]["0"]["metrics"]["memory"]["clean"]["raw_labels"] = "sensitive"
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_public_export_recomputes_gates(self):
        raw = summary()
        raw["folds"]["0"]["gates"]["passed"] = False
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_status_cannot_export_exception_messages(self):
        with patch.object(validation,"STATUS") as destination, patch.object(
                validation,"_PHASE","adapt"):
            validation._status("failed",ValueError("private_input_path_never_export"))
        record = json.loads(destination.write_text.call_args.args[0])
        self.assertEqual(record,{"state":"failed","phase":"adapt","exception_class":"ValueError"})

    def test_status_unknown_exception_class_is_sanitized(self):
        unsafe = type("UntrustedInputIdentity",(Exception,),{})
        with patch.object(validation,"STATUS") as destination, patch.object(
                validation,"_PHASE","parity"):
            validation._status("failed",unsafe("private message"))
        self.assertEqual(json.loads(destination.write_text.call_args.args[0])["exception_class"],"Exception")

    def test_test_receipt_fails_before_any_dependency_access(self):
        with self.assertRaises(ValueError):
            validation._validate_test_receipt(dict(synthetic_only=True,success=False,tests=50))


if __name__ == "__main__":
    unittest.main()
