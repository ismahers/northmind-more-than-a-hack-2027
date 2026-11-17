"""Synthetic live-ball harness checks; no datasets, checkpoints or old reports."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession import live_ball_validation as validation
from possession.control_memory_validation import NativeGeometry


def protocol():
    return dict(version="live-ball-validation-v1",part="validation",folds=[0,6],
        scenarios=list(validation.SCENARIOS),weights=[.5,.25,.25],
        views=[[False,False],[False,True],[True,False],[True,True]],fps=25.,model_fps=10.,
        cpu_threads=4,max_seconds=3600,train_models=False,fit_transition_prior_on_train=True,
        test_access=False,target_access=False,parameter_search=False,binary_hmm_strength=.5,
        preserve_geometric_dead=True,neutral_tagging=True,memory_enabled=False,guardrails=validation.GUARDS.copy(),
        target_prior_policy_if_promoted="equal_average_of_fold_0_and_6_transition_probabilities")


def native_fixture(n=151,missing=False):
    t = np.arange(n)/25.
    ball = np.column_stack((52.5+t,np.full(n,34.)))
    distances = np.tile([.8,3.],(n,1))
    if missing:
        ball[:] = np.nan
        distances[:] = np.inf
    return NativeGeometry(t,np.zeros(n,int),np.arange(n),np.zeros(n,int),ball,distances)


def metric_rows(value=.5):
    return {s:dict(spellF1=value,macroF1=.6,short_recall=.4,long_recall=.7,
        false_transitions_per_minute=1.) for s in validation.SCENARIOS}


def summary():
    folds = {}
    for fold in ("0","6"):
        old,new = metric_rows(),metric_rows(.51)
        folds[fold] = dict(metrics={"baseline":old,"live_ball":new},gates=validation.gates(new,old),
            identity_parity={s:True for s in validation.SCENARIOS},baseline_parity=True)
    return dict(folds=folds,registration_sha256="a"*64,test_accessed=False,target_accessed=False,
        calibration_accessed=False,automatic_promotion=False,
        transition_priors={fold:dict(state_order=["D","L"],
            transition_probabilities=[[.9,.1],[.05,.95]]) for fold in ("0","6")})


def split_fixture():
    return dict(train=["idsse:a","idsse:b","idsse:c","idsse:d"],
        calibration="idsse:e",validation="idsse:f",test="idsse:g")


class LiveBallValidationTests(unittest.TestCase):
    def test_protocol_accepts_one_fixed_hypothesis(self):
        validation.validate_protocol(protocol())

    def test_protocol_rejects_target_test_training_and_parameter_search(self):
        for key in ("train_models","test_access","target_access","parameter_search","memory_enabled"):
            p = protocol()
            p[key] = True
            with self.subTest(key=key),self.assertRaises(ValueError):
                validation.validate_protocol(p)

    def test_protocol_boolean_barriers_are_strict(self):
        p = protocol()
        p["target_access"] = 0
        with self.assertRaises(ValueError):
            validation.validate_protocol(p)
        p = protocol()
        p["preserve_geometric_dead"] = 1
        with self.assertRaises(ValueError):
            validation.validate_protocol(p)

    def test_protocol_rejects_new_strength_or_changed_guard(self):
        for key,value in (("binary_hmm_strength",.6),("model_fps",25.),("max_seconds",3601)):
            p = protocol()
            p[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validation.validate_protocol(p)

    def test_prior_requests_only_own_fold_training_views(self):
        names = []
        store = SimpleNamespace(harmonized=lambda key:names.append(key) or object())
        with patch("possession.live_ball.fit_binary_prior",return_value={"fixture":True}) as fit:
            result = validation.fit_fold_prior(store,split_fixture())
        self.assertEqual(names,split_fixture()["train"])
        self.assertEqual(result,{"fixture":True})
        self.assertEqual(fit.call_args.kwargs,{"fps":10.})

    def test_prior_rejects_leaking_roles_before_store_access(self):
        roles = split_fixture()
        roles["train"][0] = roles["validation"]
        store = SimpleNamespace(harmonized=lambda key:self.fail("No view may be loaded"))
        with self.assertRaises(ValueError):
            validation.fit_fold_prior(store,roles)

    def test_new_live_removes_missing_observation_dead_mask(self):
        native = native_fixture(missing=True)
        old,new = validation.decode_pair(native,np.full(151,.8),np.ones(151,bool),[105.,68.])
        self.assertTrue((old=="D").all())
        self.assertTrue((new=="H").all())

    def test_new_live_fallback_can_be_away(self):
        native = native_fixture(missing=True)
        _,new = validation.decode_pair(native,np.full(151,-.8),np.ones(151,bool),[105.,68.])
        self.assertTrue((new=="A").all())

    def test_new_live_exact_zero_tie_falls_back_home(self):
        native = native_fixture(missing=True)
        _,new = validation.decode_pair(native,np.zeros(151),np.ones(151,bool),[105.,68.])
        self.assertTrue((new=="H").all())

    def test_learned_dead_can_add_dead_with_good_tracking(self):
        native = native_fixture()
        live = np.ones(151,bool)
        live[65:85] = False
        old,new = validation.decode_pair(native,np.zeros(151),live,[105.,68.])
        self.assertTrue((old=="H").all())
        np.testing.assert_array_equal(new=="D",~live)

    def test_geometric_dead_is_preserved_despite_learned_live(self):
        native = native_fixture()
        native.ball[:,0] = 110.
        old,new = validation.decode_pair(native,np.ones(151),np.ones(151,bool),[105.,68.])
        self.assertTrue((old=="D").all())
        np.testing.assert_array_equal(old,new)

    def test_no_dead_change_reproduces_baseline_without_memory(self):
        native = native_fixture()
        old,new = validation.decode_pair(native,np.zeros(151),np.ones(151,bool),[105.,68.])
        np.testing.assert_array_equal(old,new)

    def test_baseline_exactly_reproduces_previous_pure_decoder_branch(self):
        from possession.control_memory_validation import decode_pair as old_decode
        native = native_fixture()
        neural = np.sin(native.time)
        expected,_ = old_decode(native,neural,[105.,68.])
        actual,_ = validation.decode_pair(native,neural,np.ones(151,bool),[105.,68.])
        np.testing.assert_array_equal(actual,expected)

    def test_baseline_previous_decoder_identity_under_missingness(self):
        from possession.control_memory_validation import decode_pair as old_decode
        native = native_fixture(n=351)
        native.ball[75:275] = np.nan
        native.distances[75:275] = np.inf
        signal = np.cos(native.time)
        expected,_ = old_decode(native,signal,[105.,68.])
        actual,_ = validation.decode_pair(native,signal,np.ones(351,bool),[105.,68.])
        np.testing.assert_array_equal(actual,expected)

    def test_segment_restart_never_joins_missing_live_inference(self):
        one = native_fixture(n=101,missing=True)
        native = NativeGeometry(np.r_[one.time,one.time],np.r_[np.zeros(101,int),np.ones(101,int)],
            np.arange(202),np.zeros(202,int),np.r_[one.ball,one.ball],np.r_[one.distances,one.distances])
        _,new = validation.decode_pair(native,np.r_[np.ones(101),-np.ones(101)],np.ones(202,bool),[105.,68.])
        self.assertTrue((new[:101]=="H").all())
        self.assertTrue((new[101:]=="A").all())

    def test_inference_needs_no_labels_or_model_files(self):
        # NativeGeometry has only coordinate arrays and clocks, no supervision.
        native = native_fixture()
        self.assertFalse(hasattr(native,"alive"))
        self.assertFalse(hasattr(native,"team"))
        validation.decode_pair(native,np.zeros(151),np.ones(151,bool),[105.,68.])

    def test_neural_and_learned_mask_contracts_fail_closed(self):
        native = native_fixture()
        for signal,live in ((np.full(151,np.nan),np.ones(151,bool)),
                            (np.zeros(151),np.ones(151)),
                            (np.ones(151)*1.1,np.ones(151,bool))):
            with self.assertRaises(ValueError):
                validation.decode_pair(native,signal,live,[105.,68.])

    def test_public_baseline_parity_tolerance(self):
        expected = metric_rows()["clean"]
        actual = expected.copy()
        actual["spellF1"] += 1e-11
        validation.assert_baseline(actual,expected)
        actual["spellF1"] += 1e-8
        with self.assertRaises(ValueError):
            validation.assert_baseline(actual,expected)

    def test_public_export_has_closed_scope_and_no_free_input_text(self):
        raw = summary()
        raw["input_path"] = "must_never_export"
        safe = validation.safe_public_summary(raw)
        self.assertNotIn("must_never_export",json.dumps(safe))
        self.assertFalse(safe["memory_used"])
        self.assertFalse(safe["automatic_promotion"])
        self.assertFalse(safe["full_target_recipe"])
        self.assertEqual(set(safe["folds"]["0"]["metrics"]),{"baseline","live_ball"})

    def test_public_export_requires_historical_baseline_parity(self):
        raw = summary()
        raw["folds"]["6"]["baseline_parity"] = False
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_public_export_prior_matrices_are_portable_without_raw_counts(self):
        safe = validation.safe_public_summary(summary())
        self.assertEqual(set(safe["transition_priors"]),{"0","6"})
        for prior in safe["transition_priors"].values():
            self.assertEqual(set(prior),{"state_order","transition_probabilities"})
            self.assertEqual(prior["state_order"],["D","L"])
            np.testing.assert_allclose(np.sum(prior["transition_probabilities"],axis=1),1.)

    def test_public_prior_schema_rejects_extra_keys_free_strings_or_invalid_rows(self):
        for value in (dict(state_order=["D","L"],transition_probabilities=[[.9,.1],[.05,.95]],
                           input_path="private"),
                      dict(state_order=["L","D"],transition_probabilities=[[.9,.1],[.05,.95]]),
                      dict(state_order=["D","L"],transition_probabilities=[[".9",.1],[.05,.95]]),
                      dict(state_order=["D","L"],transition_probabilities=[[.9,.2],[.05,.95]]),
                      dict(state_order=["D","L"],transition_probabilities=[[float("nan"),.1],[.05,.95]])):
            with self.assertRaises(ValueError):
                validation.checked_prior(value)

    def test_public_export_rejects_nonfinite_metrics(self):
        raw = summary()
        raw["folds"]["0"]["metrics"]["live_ball"]["clean"]["spellF1"] = float("nan")
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_public_export_requires_both_folds_and_recomputes_gates(self):
        raw = summary()
        raw["folds"]["6"]["gates"]["passed"] = False
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)
        raw = summary()
        del raw["folds"]["6"]
        with self.assertRaises(ValueError):
            validation.safe_public_summary(raw)

    def test_status_never_exports_messages_or_untrusted_class_name(self):
        unsafe = type("UntrustedIdentity",(Exception,),{})
        with patch.object(validation,"STATUS") as destination,patch.object(validation,"_PHASE","prior"):
            validation._status("failed",unsafe("private_input_path"))
        record = json.loads(destination.write_text.call_args.args[0])
        self.assertEqual(record,dict(state="failed",phase="prior",exception_class="Exception"))

    def test_receipt_rejected_before_any_file_access(self):
        with self.assertRaises(ValueError):
            validation._validate_receipt(dict(success=False,tests=30))

    def test_static_source_closure_excludes_unrelated_future_modules(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root/"main.py").write_text("from helpers import used\n",encoding="utf-8")
            (root/"helpers.py").write_text("def used(): return 1\n",encoding="utf-8")
            (root/"future.py").write_text("raise RuntimeError('never execute')\n",encoding="utf-8")
            with patch.object(validation,"ROOT",root):
                actual = validation._source_closure(["main.py"])
            self.assertEqual({p.name for p in actual},{"main.py","helpers.py"})

    def test_static_source_closure_handles_relative_imports(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            package = root/"package"
            package.mkdir()
            (package/"__init__.py").write_text("",encoding="utf-8")
            (package/"main.py").write_text("from .support import helper\n",encoding="utf-8")
            (package/"support.py").write_text("helper = 1\n",encoding="utf-8")
            with patch.object(validation,"ROOT",root):
                actual = validation._source_closure(["package/main.py"])
            self.assertEqual({p.name for p in actual},{"main.py","support.py","__init__.py"})

    def test_neural_four_view_team_and_alive_means_match_target(self):
        view = SimpleNamespace(time=np.arange(3)/10,run=np.zeros(3,int))
        reference = np.zeros((3,3))
        fused = [(np.full(3,float(z)),np.full(3,alive))
            for z,alive in ((1,.2),(2,.4),(3,.6),(4,.8))]
        with patch("possession.st_tcn_target._transform_view",return_value=view),patch(
                "possession.phase8.evaluation.predict",return_value=(reference,None,np.zeros((3,2)))),patch(
                "possession.temporal_probe.windows.predict",return_value=(np.zeros(3),None,None)),patch(
                "possession.st_tcn_target._fuse_member",side_effect=fused),patch(
                "possession.temporal_probe.windows.emissions",return_value=reference):
            team,alive = validation._neural(view,view,{"params":{}},[object(),object()],
                [object(),object()],dict(core_frames=16,batch_size=1),reference,lambda:None)
        np.testing.assert_allclose(team,np.tanh(-1/2.2))
        np.testing.assert_allclose(alive,.5)

    def test_neural_alive_roundoff_is_clipped_exactly_as_target(self):
        view = SimpleNamespace(time=np.arange(3)/10,run=np.zeros(3,int))
        reference = np.zeros((3,3))
        with patch("possession.st_tcn_target._transform_view",return_value=view),patch(
                "possession.phase8.evaluation.predict",return_value=(reference,None,np.zeros((3,2)))),patch(
                "possession.temporal_probe.windows.predict",return_value=(np.zeros(3),None,None)),patch(
                "possession.st_tcn_target._fuse_member",return_value=(np.zeros(3),np.full(3,1+1e-8))),patch(
                "possession.temporal_probe.windows.emissions",return_value=reference):
            _,alive = validation._neural(view,view,{"params":{}},[object(),object()],
                [object(),object()],dict(core_frames=16,batch_size=1),reference,lambda:None)
        np.testing.assert_array_equal(alive,np.ones(3))


if __name__=="__main__":
    unittest.main()
