"""Run reviewed tasks with all child output captured locally; print only OK/FAILED."""

import subprocess
import os
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "evaluation" / "reports" / "local-logs"
TASKS = {
    "clean-submission-install": ("from scripts.clean_submission_install import main; raise SystemExit(main())", ()),
    "refresh-submission-source": ("from scripts.build_submission_repo import refresh_review_copy; raise SystemExit(refresh_review_copy())", ()),
    "compare-submission-replay": ("from scripts.compare_submission_replay import main; raise SystemExit(main())", ()),
    "check-submission-install": ("from scripts.check_submission_install import main; raise SystemExit(main())", ()),
    "audit-submission-environment": ("from scripts.submission_environment import main; raise SystemExit(main())", ()),
    "prepare-final-submission": ("from scripts.final_submission import main; raise SystemExit(main())", ()),
    "test-final-submission": ("from scripts.final_submission import test; raise SystemExit(test())", ()),
    "restore-submission-assets": ("from scripts.submission_assets import main; raise SystemExit(main())", ()),
    "test-submission-portability": ("from scripts.submission_assets import tests; raise SystemExit(tests())", ()),
    "build-submission-source": ("from scripts.build_submission_repo import main; raise SystemExit(main())", ()),
    "test-control-transfer-geometry": ("from scripts.control_transfer_geometry import tests; raise SystemExit(tests())", ()),
    "run-control-transfer-geometry": ("from scripts.control_transfer_geometry import run; raise SystemExit(run())", ()),
    "verify-control-transfer-geometry": ("from scripts.control_transfer_geometry import verify; raise SystemExit(verify())", ()),
    "test-control-transfer": ("from scripts.control_transfer import main; raise SystemExit(main())", ()),
    "test-actor-continuity-view": ("from scripts.actor_continuity_view import tests; raise SystemExit(tests())", ()),
    "build-actor-continuity-view": ("from scripts.actor_continuity_view import build; raise SystemExit(build())", ()),
    "verify-actor-continuity-view": ("from scripts.actor_continuity_view import verify; raise SystemExit(verify())", ()),
    "replay-actor-continuity-view": ("from scripts.actor_continuity_view import replay; raise SystemExit(replay())", ()),
    "export-actor-continuity-view": ("from scripts.actor_continuity_view import export; raise SystemExit(export())", ()),
    "test-pass-coordination-contrast": ("from scripts.pass_coordination_contrast import tests; raise SystemExit(tests())", ()),
    "build-pass-coordination-contrast": ("from scripts.pass_coordination_contrast import build; raise SystemExit(build())", ()),
    "verify-pass-coordination-contrast": ("from scripts.pass_coordination_contrast import verify; raise SystemExit(verify())", ()),
    "replay-pass-coordination-contrast": ("from scripts.pass_coordination_contrast import replay; raise SystemExit(replay())", ()),
    "export-pass-coordination-contrast": ("from scripts.pass_coordination_contrast import export; raise SystemExit(export())", ()),
    "test-pass-coordination-view": ("from scripts.pass_coordination_view import tests; raise SystemExit(tests())", ()),
    "build-pass-coordination-view": ("from scripts.pass_coordination_view import build; raise SystemExit(build())", ()),
    "verify-pass-coordination-view": ("from scripts.pass_coordination_view import verify; raise SystemExit(verify())", ()),
    "replay-pass-coordination-view": ("from scripts.pass_coordination_view import replay; raise SystemExit(replay())", ()),
    "export-pass-coordination-view": ("from scripts.pass_coordination_view import export; raise SystemExit(export())", ()),
    "test-pass-coordination": ("from scripts.pass_coordination import tests; raise SystemExit(tests())", ()),
    "run-pass-coordination": ("from scripts.pass_coordination import run; raise SystemExit(run())", ()),
    "verify-pass-coordination": ("from scripts.pass_coordination import verify; raise SystemExit(verify())", ()),
    "replay-pass-coordination": ("from scripts.pass_coordination import replay; raise SystemExit(replay())", ()),
    "export-pass-coordination": ("from scripts.pass_coordination import export; raise SystemExit(export())", ()),
    "audit-video-availability": ("from scripts.video_availability import main; raise SystemExit(main())", ()),
    "run-oof-case-review": ("from scripts.oof_case_review import run; raise SystemExit(run())", ()),
    "verify-oof-case-review": ("from scripts.oof_case_review import verify; raise SystemExit(verify())", ()),
    "replay-oof-case-review": ("from scripts.oof_case_review import replay; raise SystemExit(replay())", ()),
    "export-oof-case-review": ("from scripts.oof_case_review import export; raise SystemExit(export())", ()),
    "test-oof-case-review": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_oof_case_review.py", "-v")),
    "run-conditional-fusion": ("from scripts.conditional_fusion import run; raise SystemExit(run())", ()),
    "verify-conditional-fusion": ("from scripts.conditional_fusion import verify; raise SystemExit(verify())", ()),
    "replay-conditional-fusion": ("from scripts.conditional_fusion import replay; raise SystemExit(replay())", ()),
    "export-conditional-fusion": ("from scripts.conditional_fusion import export; raise SystemExit(export())", ()),
    "test-conditional-fusion": ("from scripts.conditional_fusion import tests; raise SystemExit(tests())", ()),
    "budget-conditional-fusion": ("from possession.conditional_fusion_budget import run; raise SystemExit(run())", ()),
    "test-conditional-fusion-budget": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_conditional_fusion_budget.py", "-v")),
    "run-football-mechanism": ("from scripts.football_mechanism import run; raise SystemExit(run())", ()),
    "verify-football-mechanism": ("from scripts.football_mechanism import verify; raise SystemExit(verify())", ()),
    "replay-football-mechanism": ("from scripts.football_mechanism import replay; raise SystemExit(replay())", ()),
    "export-football-mechanism": ("from scripts.football_mechanism import export; raise SystemExit(export())", ()),
    "test-football-mechanism": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_football_mechanism.py", "-v")),
    "run-football-predictive": ("from scripts.football_predictive import run; raise SystemExit(run())", ()),
    "verify-football-predictive": ("from scripts.football_predictive import verify; raise SystemExit(verify())", ()),
    "replay-football-predictive": ("from scripts.football_predictive import replay; raise SystemExit(replay())", ()),
    "export-football-predictive": ("from scripts.football_predictive import export; raise SystemExit(export())", ()),
    "test-football-predictive": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_football_predictive.py", "-v")),
    "test-football-control": ("from scripts.football_control import tests; raise SystemExit(tests())", ()),
    "run-football-control": ("from scripts.football_control import run; raise SystemExit(run())", ()),
    "verify-football-control": ("from scripts.football_control import verify; raise SystemExit(verify())", ()),
    "export-football-control": ("from scripts.football_control import export; raise SystemExit(export())", ()),
    "run-sequence-risk": ("from scripts.sequence_risk import run; raise SystemExit(run())", ()),
    "verify-sequence-risk": ("from scripts.sequence_risk import verify; raise SystemExit(verify())", ()),
    "replay-sequence-risk": ("from scripts.sequence_risk import replay; raise SystemExit(replay())", ()),
    "export-sequence-risk": ("from scripts.sequence_risk import export; raise SystemExit(export())", ()),
    "test-sequence-risk": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_sequence_risk.py", "-v")),
    "audit-skillcorner-phases": ("from scripts.skillcorner_phase_audit import run; raise SystemExit(run())", ()),
    "export-skillcorner-audit": ("from scripts.skillcorner_phase_audit import export; raise SystemExit(export())", ()),
    "test-skillcorner-audit": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_skillcorner_phase_audit.py", "-v")),
    "audit-skillcorner-data": ("from scripts.skillcorner_audit import run; raise SystemExit(run())", ()),
    "run-trajectory-adapter": ("from scripts.trajectory_adapter import run; raise SystemExit(run())", ()),
    "verify-trajectory-adapter": ("from scripts.trajectory_adapter import verify; raise SystemExit(verify())", ()),
    "replay-trajectory-adapter": ("from scripts.trajectory_adapter import replay; raise SystemExit(replay())", ()),
    "export-trajectory-adapter": ("from scripts.trajectory_adapter import export; raise SystemExit(export())", ()),
    "test-trajectory-adapter": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_trajectory_adapter.py", "-v")),
    "run-hsmm-pilot": ("from scripts.hsmm_pilot import main; raise SystemExit(main())", ()),
    "verify-hsmm-pilot": ("from scripts.hsmm_pilot import verify_main; raise SystemExit(verify_main())", ()),
    "replay-hsmm-pilot": ("from scripts.hsmm_pilot import replay_main; raise SystemExit(replay_main())", ()),
    "test-hsmm-pilot": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_hsmm_pilot.py", "-v")),
    "audit-publish-long-diagnosis": ("from scripts.audit_long_diagnosis import main; raise SystemExit(main())", ()),
    "test-long-diagnosis-audit": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_long_diagnosis_audit.py", "-v")),
    "freeze-long-diagnosis": ("from scripts.long_diagnosis import freeze_main; raise SystemExit(freeze_main())", ()),
    "run-long-diagnosis": ("from scripts.long_diagnosis import main; raise SystemExit(main())", ()),
    "verify-long-diagnosis": ("from scripts.long_diagnosis import verify_main; raise SystemExit(verify_main())", ()),
    "test-long-diagnosis": ("from scripts.long_diagnosis import test_main; raise SystemExit(test_main())", ()),
    "validate-pulse-consensus": ("from scripts.pulse_consensus_validation import main; raise SystemExit(main())", ()),
    "verify-pulse-consensus": ("from scripts.pulse_consensus_validation import verify_main; raise SystemExit(verify_main())", ()),
    "test-pulse-consensus": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_pulse_consensus.py", "-v")),
    "validate-evidence-first": ("from scripts.evidence_first_validation import main; raise SystemExit(main())", ()),
    "verify-evidence-first": ("from scripts.evidence_first_validation import verify_main; raise SystemExit(verify_main())", ()),
    "test-evidence-first": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_evidence_first.py", "-v")),
    "validate-transfer-symmetry": ("from scripts.transfer_symmetry_validation import main; raise SystemExit(main())", ()),
    "audit-target-transfer-public": ("from scripts.target_transfer_audit import main; raise SystemExit(main())", ()),
    "test-target-st-tcn": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_st_tcn_target.py", "-v")),
    "compute-target-st-tcn": ("from scripts.compute_target_st_tcn import main; raise SystemExit(main())", ()),
    "run-scale-selective": ("from scripts.scale_selective import main; raise SystemExit(main())", ()),
    "test-scale-selective": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_scale_selective.py", "-v")),
    "analyse-ball-attention-legacy-cohorts": ("from scripts.ball_attention_legacy_cohorts import main; raise SystemExit(main())", ()),
    "report-ball-attention": ("from scripts.ball_attention_report import main; raise SystemExit(main())", ()),
    "run-ball-attention": ("from scripts.ball_attention import main; raise SystemExit(main())", ()),
    "test-ball-attention": ("from scripts.ball_attention_checks import main; raise SystemExit(main())", ()),
    "analyse-ha-pilot-mechanism": ("from scripts.ha_pilot_mechanism import main; raise SystemExit(main())", ()),
    "plot-ha-discrimination": ("from scripts.plot_ha_discrimination import main; raise SystemExit(main())", ()),
    "diagnose-ha-discrimination": ("from scripts.ha_discrimination import main; raise SystemExit(main())", ()),
    "test-ha-discrimination": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_ha_discrimination.py", "-v")),
    "run-local-head-budget": ("from scripts.local_head_budget import main; raise SystemExit(main())", ()),
    "test-local-head-budget": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_local_head_budget.py", "-v")),
    "benchmark-mps": ("from scripts.mps_benchmark import main; raise SystemExit(main())", ()),
    "test-mps-benchmark": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_mps_benchmark.py", "-v")),
    "analyse-duration-balanced": ("from scripts.duration_balanced_analysis import main; raise SystemExit(main())", ()),
    "run-duration-balanced": ("from scripts.duration_balanced import main; raise SystemExit(main())", ()),
    "test-duration-balanced": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_duration_balanced.py", "-v")),
    "continue-temporal-teachers": ("from scripts.continue_temporal_teachers import main; raise SystemExit(main())", ()),
    "plot-temporal-probe": ("from scripts.plot_temporal_probe import main; raise SystemExit(main())", ()),
    "status-temporal-probe": ("from scripts.temporal_probe_status import main; raise SystemExit(main())", ()),
    "run-temporal-probe": ("from scripts.temporal_probe import main; raise SystemExit(main())", ()),
    "test-temporal-probe": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_temporal_probe.py", "-v")),
    "setup-temporal-probe": ("from scripts.temporal_probe_environment import main; raise SystemExit(main())", ()),
    "check-temporal-probe": ("from scripts.temporal_probe_readiness import main; raise SystemExit(main())", ()),
    "report-phase8": ("from scripts.phase8_report import main; raise SystemExit(main())", ()),
    "verify-phase8": ("from scripts.verify_phase8 import main; raise SystemExit(main())", ()),
    "analyse-phase8-variability": ("from scripts.phase8_variability import main; raise SystemExit(main())", ()),
    "probe-phase8-intervals": ("from scripts.phase8_intervals import main; raise SystemExit(main())", ()),
    "test-interval-probe": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_interval_probe.py", "-v")),
    "test-phase8-diagnosis": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase8_diagnosis.py", "-v")),
    "diagnose-phase8": ("from scripts.phase8_diagnosis import main; raise SystemExit(main())", ()),
    "run-phase8": ("from scripts.phase8_experiments import main; raise SystemExit(main())", ()),
    "research-phase8-details": ("from scripts.phase8_source_details import main; raise SystemExit(main())", ()),
    "research-phase8-sources": ("from scripts.phase8_sources import main; raise SystemExit(main())", ()),
    "test-phase8": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase8.py", "-v")),
    "train-phase8": ("from scripts.phase8_train import main; raise SystemExit(main())", ()),
    "profile-phase8": ("from scripts.phase8_profile import main; raise SystemExit(main())", ()),
    "verify-phase7-capacity": ("from scripts.verify_phase7_capacity import main; raise SystemExit(main())", ()),
    "capacity-phase7": ("from scripts.phase7_capacity import main; raise SystemExit(main())", ()),
    "test-phase7-capacity": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase7_capacity.py", "-v")),
    "audit-phase7-grid": ("from scripts.phase7_distribution import main; raise SystemExit(main())", ()),
    "profile-phase7": ("from scripts.phase7_profile import main; raise SystemExit(main())", ()),
    "report-phase7": ("from scripts.phase7_report import main; raise SystemExit(main())", ()),
    "freeze-phase7": ("from scripts.phase7_freeze import main; raise SystemExit(main())", ()),
    "verify-phase7": ("from scripts.verify_phase7 import main; raise SystemExit(main())", ()),
    "run-phase7": ("from scripts.phase7_experiments import main; raise SystemExit(main())", ()),
    "audit-phase7": ("from scripts.phase7_audit import main; raise SystemExit(main())", ()),
    "test-phase7": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase7.py", "-v")),
    "robust-phase6-flags": ("from scripts.phase6_flags_robustness import main; raise SystemExit(main())", ()),
    "ablate-phase6-flags": ("from scripts.phase6_flag_ablation import main; raise SystemExit(main())", ()),
    "diagnose-phase6": ("from scripts.phase6_diagnosis import main; raise SystemExit(main())", ()),
    "verify-phase6": ("from scripts.verify_phase6 import main; raise SystemExit(main())", ()),
    "report-phase6": ("from scripts.phase6_report import main; raise SystemExit(main())", ()),
    "transfer-phase6": ("from scripts.phase6_transfer import main; raise SystemExit(main())", ()),
    "audit-phase6-target": ("from scripts.phase6_target import main; raise SystemExit(main())", ()),
    "run-phase6": ("from scripts.phase6_experiments import main; raise SystemExit(main())", ()),
    "test-phase6": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase6.py", "-v")),
    "check-phase6-assets": ("from scripts.phase6_availability import main; raise SystemExit(main())", ()),
    "analyse-phase5": ("from scripts.phase5_error_analysis import main; raise SystemExit(main())", ()),
    "verify-phase5": ("from scripts.verify_phase5 import main; raise SystemExit(main())", ()),
    "report-phase5": ("from scripts.phase5_report import main; raise SystemExit(main())", ()),
    "run-phase5": ("from scripts.phase5_experiments import main; raise SystemExit(main())", ()),
    "test-phase5": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase5.py", "-v")),
    "run-night": ("from scripts.night_experiments import main; raise SystemExit(main())", ()),
    "test-night": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_night.py", "-v")),
    "audit-external-identities": ("from scripts.audit_external_identities import main; raise SystemExit(main())", ()),
    "verify-phase3": ("from scripts.verify_phase3 import main; raise SystemExit(main())", ()),
    "run-phase3": ("from scripts.phase3 import main; raise SystemExit(main())", ()),
    "test-phase3": ("import unittest; unittest.main(module=None)", ("discover", "-s", "tests", "-p", "test_phase3.py", "-v")),
    "verify-colab-bundle": ("from scripts.build_colab_phase2 import verify_main; raise SystemExit(verify_main())", ()),
    "verify-external-reproduction": ("from scripts.external_reproduction import main; raise SystemExit(main())", ()),
    "build-colab-phase2": ("from scripts.build_colab_phase2 import main; raise SystemExit(main())", ()),
    "verify-external-readiness": ("from scripts.external_readiness import main; raise SystemExit(main())", ()),
    "summarize-external-audits": ("from scripts.external_benchmark import audit_summary_main; raise SystemExit(audit_summary_main())", ()),
    "summarize-external-benchmark": ("from scripts.external_benchmark import benchmark_summary_main; raise SystemExit(benchmark_summary_main())", ()),
    "check-external-environment": ("from scripts.external_environment import main; raise SystemExit(main())", ()),
    "download-external-data": ("from scripts.external_data import download_main; raise SystemExit(download_main())", ()),
    "prepare-external-data": ("from scripts.external_data import prepare_main; raise SystemExit(prepare_main())", ()),
    "make-external-folds": ("from scripts.external_data import folds_main; raise SystemExit(folds_main())", ()),
    "train-external-e0": ("from scripts.train_e0 import train_main; raise SystemExit(train_main())", ()),
    "evaluate-external-e0": ("from scripts.train_e0 import evaluate_main; raise SystemExit(evaluate_main())", ()),
    "smoke-external-e0": ("from scripts.smoke_e0 import main; raise SystemExit(main())", ()),
    "test-external-phase1": (
        "import unittest; unittest.main(module=None)",
        ("discover", "-s", "tests", "-p", "test_external_*.py", "-v")),
    "evaluate-public-baselines": (
        "import sys; sys.path.insert(0, 'evaluation'); from run_evaluation import main; raise SystemExit(main())",
        ("--gt", "evaluation/gt/BAR-ATH_possession_gt_public.csv",
         "--results", ".venv/public-baselines", "--out", "evaluation/reports/public-baselines",
         "--bootstrap", "1000", "--no-detail"),
    ),
    "reproduce-baselines": ("from scripts.reproduce_baselines import main; raise SystemExit(main())", ()),
    "evaluate-tracking": ("from scripts.run_experiments import main; raise SystemExit(main())", ()),
    "evaluate-tracking-with-tagging": (
        "from scripts.run_experiments import main; raise SystemExit(main())", ("--with-tagging",)),
    "verify-outputs": ("from scripts.verify_outputs import main; raise SystemExit(main())", ()),
    "evaluate-sequence-model": (
        "from scripts.run_experiments import main; raise SystemExit(main())", ("--sequence",)),
    "verify-sequence-outputs": (
        "from scripts.verify_outputs import main; raise SystemExit(main())", ("--sequence",)),
    "test-sequence-decoder": (
        "import unittest; unittest.main(module=None)",
        ("discover", "-s", "tests", "-p", "test_sequence_decoding.py", "-v")),
}


def run_task(task, extra_args=()):
    """Never read captured output, including on failure. Return a generic status.

    The child imports processing code only AFTER stdout and stderr have been
    redirected at process creation. Warnings, parser errors, tracebacks and
    inherited evaluator output all go to the same ignored local log.
    """
    try:
        code, default_args = TASKS[task]
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / (task + "-" + uuid.uuid4().hex + ".log")
        with log_path.open("xb") as log:
            process = subprocess.run(
                [sys.executable, "-c", code, *default_args, *extra_args],
                cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=2400 if task == "prepare-final-submission" else 600 if task in {"run-pass-coordination", "run-control-transfer-geometry", "verify-control-transfer-geometry"} else 30 if task in {"build-actor-continuity-view", "test-control-transfer", "test-control-transfer-geometry"} else 120 if task in {"build-pass-coordination-view", "build-pass-coordination-contrast"} else None,
                env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                     "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"} if task in {"run-pass-coordination", "run-control-transfer-geometry", "test-control-transfer-geometry", "verify-control-transfer-geometry"} else None,
                check=False,
            )
        success = process.returncode == 0
    except BaseException:
        # Even launcher errors and interrupts must not expose exception text.
        success = False
    print("OK" if success else "FAILED", flush=True)
    return 0 if success else 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("FAILED", flush=True)
        return 1
    # Only the newly reviewed external-data CLIs accept extra arguments. Legacy tasks stay fixed.
    accepts_args = {"analyse-duration-balanced", "run-duration-balanced", "run-temporal-probe", "run-phase8", "run-phase7", "download-external-data", "prepare-external-data", "make-external-folds",
                    "train-external-e0", "evaluate-external-e0", "summarize-external-benchmark", "run-phase3", "run-night", "run-phase5", "run-phase6", "report-phase6"}
    accepts_args.add("benchmark-mps")
    accepts_args.add("run-local-head-budget")
    accepts_args.add("diagnose-ha-discrimination")
    accepts_args.add("run-ball-attention")
    accepts_args.add("run-scale-selective")
    if len(sys.argv) > 2 and sys.argv[1] not in accepts_args:
        print("FAILED", flush=True)
        return 1
    return run_task(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
