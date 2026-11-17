"""Prepare a separate, UNSELECTED target candidate through scripts.run_local.

No target GT, evaluation, video or parameter search. Public status contains
closed phases, booleans and sanitized exception classes only. Hashes, arrays,
predictions and provenance remain in the ignored local candidate directory.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/live-ball-candidate-v1"
CANDIDATE = DEST/"Northmind-live-ball-v1.csv"
ORIGINAL = ROOT/"evaluation/reports/final-delivery/Northmind.csv"
PRIVATE_RECEIPT = DEST/"provenance.json"
PRIOR = ROOT/"configs/live_ball_prior_v1.json"
EXTERNAL = ROOT/"docs/public-results/2026-11-17/live-ball-v1.json"
TEST_RECEIPT = ROOT/"docs/public-results/2026-11-17/live-ball-candidate-tests.json"
PUBLIC_STATUS = ROOT/"docs/public-results/2026-11-17/live-ball-candidate-status.json"
TEST_SOURCES = ("possession/live_ball_target.py","scripts/live_ball_candidate.py",
    "tests/test_live_ball_candidate.py","configs/live_ball_prior_v1.json")
EXTRA_SOURCES = TEST_SOURCES+("possession/live_ball.py","possession/external_data/features.py",
    "possession/external_data/audit.py","possession/external_data/schema.py",
    "possession/phase7/features.py","possession/live_ball_validation.py",
    "possession/control_memory_validation.py","evaluation/possession_metrics.py","scripts/run_local.py")
_PHASES = {"setup","neural","provenance","geometry","anchors","decode","validate","write","complete"}
_EXCEPTIONS = {"ValueError","TypeError","KeyError","IndexError","OSError","FileNotFoundError",
    "FileExistsError","PermissionError","TimeoutError","RuntimeError","AssertionError",
    "ImportError","ModuleNotFoundError","MemoryError"}
_FLAGS = ("validated","reproducible","candidate_exists","original_unchanged",
    "anchor_roundtrip","legacy_baseline_parity","selected_for_submission","validation_gates_passed")


def tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_live_ball_candidate.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    hashes = {name:hashlib.sha256((ROOT/name).read_text(encoding="utf-8-sig").replace(
        "\r\n","\n").encode("utf-8")).hexdigest() for name in TEST_SOURCES}
    record = dict(synthetic_only=True,tests=result.testsRun,success=result.wasSuccessful() and result.testsRun>=20,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),sources_sha256_lf=hashes)
    record["failed_test_ids"] = [case.id() for case,_ in result.failures+result.errors
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*",case.id())]
    _write_json(TEST_RECEIPT,record)
    return 0 if record["success"] else 1


def _require_tests():
    record = _json(TEST_RECEIPT)
    if (record.get("synthetic_only") is not True or record.get("success") is not True
            or record.get("tests",0)<20 or record.get("errors")!=0 or record.get("failures")!=0
            or record.get("skipped")!=0 or set(record.get("sources_sha256_lf",{}))!=set(TEST_SOURCES)):
        raise ValueError("Current synthetic receipt required")
    for name in TEST_SOURCES:
        digest = hashlib.sha256((ROOT/name).read_text(encoding="utf-8-sig").replace(
            "\r\n","\n").encode("utf-8")).hexdigest()
        if digest!=record["sources_sha256_lf"][name]:
            raise ValueError("Candidate source changed after synthetic tests")


def validate_prior_recipe(recipe,external):
    import numpy as np
    from possession.live_ball_validation import checked_prior
    if (recipe.get("version")!="live-ball-prior-v1" or recipe.get("selected_for_submission") is not False
            or recipe.get("validation_gates_passed") is not False
            or recipe.get("policy")!="equal_average_of_fold_0_and_6_transition_probabilities"
            or not isinstance(recipe.get("source_registration_sha256"),str)
            or re.fullmatch(r"[0-9a-f]{64}",recipe["source_registration_sha256"]) is None
            or external.get("registration_sha256")!=recipe["source_registration_sha256"]
            or external.get("passes_both") is not False or external.get("automatic_promotion") is not False):
        raise ValueError("Candidate must remain experimental and unselected")
    prior = checked_prior({k:recipe[k] for k in ("state_order","transition_probabilities")})
    means = []
    for fold in ("0","6"):
        value = checked_prior(external["transition_priors"][fold])
        means.append(np.asarray(value["transition_probabilities"],float))
    if not np.array_equal(np.asarray(prior["transition_probabilities"]),.5*means[0]+.5*means[1]):
        raise ValueError("Binary prior must equal the fixed external fold average")
    return prior


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_status(status,error=None):
    if status.get("phase") not in _PHASES or any(type(status.get(key)) is not bool for key in _FLAGS):
        raise ValueError("Invalid candidate status schema")
    if status["selected_for_submission"] or status["validation_gates_passed"]:
        raise ValueError("Candidate cannot be selected automatically")
    result = {key:status[key] for key in ("phase",)+_FLAGS}
    if error is not None:
        name = type(error).__name__
        result["exception_class"] = name if name in _EXCEPTIONS else "Exception"
    return result


def _publish(status,error=None):
    _write_json(PUBLIC_STATUS,safe_status(status,error))


def _guard_existing(path,content):
    if path.exists() and path.read_bytes()!=content:
        raise FileExistsError("Existing candidate differs; no replacement permitted")


def atomic_candidate_write(path,content):
    """Atomically add a new artifact; never overwrite even in a creation race."""
    _guard_existing(path,content)
    if path.exists():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Windows rename is atomic on one volume and refuses an
                # existing destination. Hard links may be unavailable in a
                # managed/synced filesystem; no overwrite fallback is used.
                if temporary.parent.resolve()!=path.parent.resolve():
                    raise ValueError("Atomic candidate paths must share a directory")
                os.rename(temporary,path)
            else:
                os.link(temporary,path)
        except FileExistsError:
            _guard_existing(path,content)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot():
    from scripts.final_submission import _provenance,sha256_file
    from possession.st_tcn_target import CACHE_PATH
    inputs,sources = _provenance()
    sources.update({name:sha256_file(ROOT/name) for name in EXTRA_SOURCES})
    return dict(inputs_sha256=inputs,sources_sha256=sources,cache_sha256=sha256_file(CACHE_PATH),
        prior_sha256=sha256_file(PRIOR),external_artifact_sha256=sha256_file(EXTERNAL),
        synthetic_receipt_sha256=sha256_file(TEST_RECEIPT),original_sha256=sha256_file(ORIGINAL))


def _generate(prior,config,sequence_config,set_phase,budget):
    import numpy as np
    from games import GAME,half_video_blocks
    from possession.inference import predict
    from possession.sequence_decoding import predict_sequence
    from possession.tracking_features import load_evidence
    from possession.tactical_tagging import load_team_prior,get_quick_restart_mask,load_restart_and_box_prior
    from possession.st_tcn_target import load_cached_st_tcn_target
    from possession.target_domain import load_target
    from possession.external_data.features import build_features
    from possession.phase7.features import harmonize
    from possession.live_ball_target import recover_cached_live,decode_target_candidate
    from scripts.final_submission import serialize_submission,EXPECTED_ROWS
    budget()
    set_phase("geometry")
    evidence = load_evidence(use_cache=False)
    team_prior,_ = load_team_prior()
    tagged,_ = predict(evidence,config,team_prior=team_prior)
    reference = tagged["northmind_geometry"]
    quick = get_quick_restart_mask(window_frames=50)
    restart = load_restart_and_box_prior(n_frames=EXPECTED_ROWS,post_restart_frames=50)
    cache = load_cached_st_tcn_target(EXPECTED_ROWS)
    if cache is None or not cache.get("fingerprint"):
        raise ValueError("Registered target neural cache required")
    set_phase("anchors")
    match = load_target()  # Only this loader; never run_target or any target scorer.
    if match is None:
        raise ValueError("Target geometry unavailable")
    view = harmonize(build_features(match))
    learned_live = recover_cached_live(match,view,cache,prior,EXPECTED_ROWS)
    budget()
    set_phase("decode")
    baseline,candidate = decode_target_candidate(evidence,reference,team_prior,quick,restart,cache,
        learned_live,list(half_video_blocks(GAME)),config,sequence_config)
    old,checks = predict_sequence(evidence,reference,team_prior,config,sequence_config)
    if (not np.array_equal(baseline,old["northmind_sequence"])
            or checks.get("uses_equivariant_st_tcn") is not True
            or checks.get("neural_cache_fingerprint")!=cache["fingerprint"]
            or checks.get("control_only_preserves_dead_mask") is not True):
        raise ValueError("Current target baseline parity failed")
    budget()
    return (serialize_submission(baseline),serialize_submission(candidate),cache["fingerprint"])


def _validate_bytes(content):
    from evaluation import possession_metrics as pm
    from scripts.final_submission import EXPECTED_ROWS
    path = DEST/("validation-"+uuid.uuid4().hex+".csv")
    try:
        path.write_bytes(content)
        if pm.validate_submission(pm.read_label_csv(path),EXPECTED_ROWS):
            raise ValueError("Invalid candidate submission format")
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _exclusive():
    DEST.mkdir(parents=True,exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x",encoding="utf-8") as stream:
        stream.write("Unselected live-ball candidate\n")
    try:
        yield
    finally:
        lock.unlink()


def prepare():
    """Two cold geometry generations, without selecting or replacing delivery."""
    status = {key:False for key in _FLAGS}
    status["phase"] = "setup"
    start = time.perf_counter()
    def budget():
        if time.perf_counter()-start>=2400:
            raise TimeoutError("Candidate preparation budget exhausted")
    def phase(name):
        status["phase"] = name
        _publish(status)
        budget()
    try:
        with _exclusive():
            phase("setup")
            _require_tests()
            recipe = _json(PRIOR)
            prior = validate_prior_recipe(recipe,_json(EXTERNAL))
            from games import GAME,GAMES
            from possession.inference import Config
            from possession.sequence_decoding import SequenceConfig
            from possession.st_tcn_target import compute_and_cache_st_tcn_target,load_cached_st_tcn_target
            from scripts.final_submission import EXPECTED_ROWS
            import torch
            if GAME!="BAR-ATH" or GAMES[GAME]["n_video_frames"]!=EXPECTED_ROWS:
                raise ValueError("Unexpected target contract")
            config,seq = Config(),SequenceConfig()
            torch.set_num_threads(4)
            phase("neural")
            if compute_and_cache_st_tcn_target()!=0 or load_cached_st_tcn_target(EXPECTED_ROWS) is None:
                raise ValueError("Required neural cache unavailable")
            torch.set_num_threads(4)
            phase("provenance")
            before = _snapshot()
            original = ORIGINAL.read_bytes()
            first = _generate(prior,config,seq,phase,budget)
            second = _generate(prior,config,seq,phase,budget)
            phase("validate")
            after = _snapshot()
            if before!=after or first!=second or first[0]!=original or ORIGINAL.read_bytes()!=original:
                raise ValueError("Candidate or original failed exact reproducibility")
            _validate_bytes(first[0])
            _validate_bytes(first[1])
            status.update(validated=True,reproducible=True,anchor_roundtrip=True,
                legacy_baseline_parity=True,original_unchanged=True)
            record = dict(version="live-ball-candidate-v1",selected_for_submission=False,
                validation_gates_passed=False,uses_gt=False,evaluated=False,provenance=after,
                recipe=recipe,config=asdict(config),sequence_config=asdict(seq),
                candidate_sha256=hashlib.sha256(first[1]).hexdigest(),
                baseline_sha256=hashlib.sha256(first[0]).hexdigest(),neural_cache_fingerprint=first[2],
                independent_generations=2,anchor_roundtrip=True,legacy_baseline_parity=True,
                precision_note="target cached alive float32; external validation averaged alive float64")
            encoded = (json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
            _guard_existing(CANDIDATE,first[1])
            _guard_existing(PRIVATE_RECEIPT,encoded)
            phase("write")
            atomic_candidate_write(CANDIDATE,first[1])
            atomic_candidate_write(PRIVATE_RECEIPT,encoded)
            if _snapshot()!=after or ORIGINAL.read_bytes()!=original:
                raise ValueError("Inputs or original changed during candidate commit")
            status.update(phase="complete",candidate_exists=True)
            _publish(status)
            return 0
    except Exception as error:
        status.update(validated=False,reproducible=False,original_unchanged=False)
        status["candidate_exists"] = CANDIDATE.is_file()
        _publish(status,error)
        raise  # Captured locally from process creation by scripts.run_local.


def verify():
    _require_tests()
    recipe = _json(PRIOR)
    validate_prior_recipe(recipe,_json(EXTERNAL))
    record = _json(PRIVATE_RECEIPT)
    if (record.get("version")!="live-ball-candidate-v1" or record.get("selected_for_submission") is not False
            or record.get("validation_gates_passed") is not False or record.get("uses_gt") is not False
            or record.get("evaluated") is not False or record.get("independent_generations")!=2
            or record.get("anchor_roundtrip") is not True or record.get("legacy_baseline_parity") is not True
            or record.get("recipe")!=recipe or record.get("provenance")!=_snapshot()):
        raise ValueError("Invalid experimental candidate provenance")
    candidate,original = CANDIDATE.read_bytes(),ORIGINAL.read_bytes()
    if (hashlib.sha256(candidate).hexdigest()!=record["candidate_sha256"]
            or hashlib.sha256(original).hexdigest()!=record["baseline_sha256"]):
        raise ValueError("Candidate or original changed")
    _validate_bytes(candidate)
    return 0
