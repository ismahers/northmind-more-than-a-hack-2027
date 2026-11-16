"""Fixed three-arm spatial experiment; both validation seals precede test."""
import time
import numpy as np
import torch
from possession.external_data.schema import file_hash
from possession.learning.data import Windows
from possession.learning.training import forward
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors
from possession.phase5.experiment import scenarios
from possession.phase8.packing import compact_collate
from possession.night.common import atomic_json
from possession.duration_balanced.selection import promotion, objective
from possession.local_head_budget.metrics import measured
from possession.ha_discrimination.diagnosis import analyze
from possession.ha_discrimination.features import geometry
from possession.temporal_probe.common import Store, Budget, read_json
from .common import ROOT, DEST, register, verify
from .training import train, load_model, equivalence, folder, equal


def select(trials,spec):
    original = trials["original"]; gates = {}
    for arm in ("ball","capacity"):
        gate = promotion(trials[arm],original,original,spec)
        gate["checks"]["direct_HA_false_transitions"] = all(trials[arm][name]["false_transitions_per_minute"] <=
            original[name]["false_transitions_per_minute"]+spec["promotion"]["false_transitions_per_minute_increase_max"]
            for name in spec["scenarios"])
        if arm == "ball":
            gate["checks"]["beats_capacity"] = objective(trials[arm],spec) >= objective(trials["capacity"],spec)+spec["capacity_margin"]
        gate["passed"] = all(gate["checks"].values());gates[arm] = gate
    allowed = ["original"]+[a for a,g in gates.items() if g["passed"]]
    return {"winner":max(allowed,key=lambda a:objective(trials[a],spec)),"gates":gates,
            "objectives":{a:objective(m,spec) for a,m in trials.items()},"test_used":False}


def predict(model,view,budget):
    ds = Windows({"public":view},256,15)
    p = np.zeros((len(view.time),3),np.float32); logit = np.zeros(len(view.time),np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ds),8):
            budget.check();items = [ds[i] for i in range(start,min(start+8,len(ds)))]
            out = forward(model,compact_collate(items))
            for j,item in enumerate(items):
                a,b,left = item["a"],item["b"],item["left"]
                p[a:b] = out["probabilities"][j,a-left:b-left].numpy()
                logit[a:b] = out["team_logit"][j,a-left:b-left].numpy()
    return p,logit


def summarize_cases(result):
    return {k:v for k,v in result.items() if k!="per_possession"}


def cohorts(original,candidate):
    a,b = original["per_possession"],candidate["per_possession"]
    if len(a)!=len(b) or any(x["duration_s"]!=y["duration_s"] or x["boundary_type"]!=y["boundary_type"] for x,y in zip(a,b)):
        raise ValueError("Cohort order mismatch")
    result = {}
    for name,predicate in (("no_correct_frame",lambda r:r["B_no_frame"]),
                           ("insufficient",lambda r:r["B"]),("decoder_lost",lambda r:r["C"]),
                           ("physical_conflict",lambda r:r["physical_conflict"])):
        ix = [i for i,row in enumerate(a) if predicate(row)]
        result[name] = {"n":len(ix),"recovered":sum(b[i]["hmm_recovered"] for i in ix),
            "raw_recovered":sum(b[i]["raw_recovered"] for i in ix),
            "any_correct_frame":sum(b[i]["correct_fraction"]>0 for i in ix),
            "majority_correct":sum(b[i]["correct_fraction"]>=.5 for i in ix),
            "p_correct_change":float(np.mean([b[i]["p_correct"]-a[i]["p_correct"] for i in ix])) if ix else None}
    return result


def assert_sealed(record,digest):
    seal = read_json(DEST/"selection-seal.json")
    if seal["registration_sha256"] != digest or seal["test_used"]:
        raise ValueError("Invalid seal")
    for fold in record["protocol"]["pilots"]:
        selection = read_json(DEST/f"selections/fold-{fold}.json")
        if seal["selections"][str(fold)] != file_hash(DEST/f"selections/fold-{fold}.json"):
            raise ValueError("Selection changed")
        for arm,sha in selection["checkpoint_hashes"].items():
            if sha != file_hash(folder(fold,arm,"binary")/"last.pt"):
                raise ValueError("Checkpoint changed since seal")


def score_part(store,record,fold,part,budget,digest):
    if part == "test":
        assert_sealed(record,digest)
    spec = record["protocol"]; teacher = record["teachers"][str(fold)]
    key = teacher["split"][part];view = store.harmonized(key);match = store.match(key)
    context = ScoreContext(match,view)
    prior = fit_priors([store.harmonized(k) for k in teacher["split"]["train"]])
    scenario_views = scenarios(view,key,record["phase8_recipe"])
    diagnostic_spec = read_json(ROOT/"configs/ha_discrimination_v1.json")
    trials = {}; diagnoses = {}; raw_cases = {}; query_stats = {}
    for arm in spec["arms"]:
        model = load_model(fold,arm);trials[arm] = {};diagnoses[arm] = {};raw_cases[arm] = {};query_stats[arm] = {}
        for name,(degraded,missing) in scenario_views.items():
            budget.check();start = time.perf_counter();p,logit = predict(model,degraded,budget)
            path = DEST/f"predictions/fold-{fold}/{arm}/{part}/{name}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream,probabilities=p,team_logit=logit,time=view.time,run=view.run,native_index=view.native_index)
            path.with_suffix(".tmp").replace(path)
            metric = measured(context,p,prior);metric["actual_hidden_fraction"] = float(missing.mean())
            metric["inference_seconds"] = time.perf_counter()-start
            trials[arm][name] = metric
            _,physical = geometry(degraded,match.metadata["pitch_m"])
            result = analyze(context,p,logit,physical,prior,diagnostic_spec)
            raw_cases[arm][name] = result;diagnoses[arm][name] = summarize_cases(result)
            atomic_json(DEST/f"cases/{part}/fold-{fold}/{arm}/{name}.json",result)
            with torch.no_grad():
                g = torch.from_numpy(degraded.global_features)
                q = model.encoder.queries(g); available = g[:,3]>.5
                residual = (q-model.encoder.seeds).norm(dim=(1,2))
                query_stats[arm][name] = {"observed_query_residual_norm":float(residual[available].mean()) if available.any() else None,
                    "unavailable_query_residual_max":float(residual[~available].max()) if (~available).any() else None,
                    "query_variation":float(q.std(0).mean())}
    comparisons = {arm:{name:cohorts(raw_cases["original"][name],raw_cases[arm][name]) for name in spec["scenarios"]}
                   for arm in ("ball","capacity")}
    value = {"fold":fold,"part":part,"metrics":trials,"diagnosis":diagnoses,"frozen_original_cohorts":comparisons,"queries":query_stats}
    atomic_json(DEST/f"scores/{part}/fold-{fold}.json",value)
    return value


def finish(record,digest):
    assert_sealed(record,digest);checks = verify(record);spec = record["protocol"]
    checks["equivalence_passed"] = read_json(DEST/"equivalence.json")["passed"]
    training_seconds = 0.;parameters = {};updates = {};costs = {}
    for fold in spec["pilots"]:
        for stage,epochs in (("base",12),("binary",8)):
            states = []
            for arm in spec["arms"]:
                path = folder(fold,arm,stage)
                state = torch.load(path/"last.pt",map_location="cpu",weights_only=False);states.append(state)
                done = read_json(path/"complete.json");costs[f"{fold}/{arm}/{stage}"] = done
                training_seconds += sum(r["training_seconds"] for r in done["history"])
                parameters[arm] = done["parameters"]
                updates[f"{fold}/{arm}/{stage}"] = sum(r["train"].get("updates",r["train"].get("optimizer_steps",0)) for r in state["history"])
                checks[f"{fold}/{arm}/{stage}/epochs"] = len(state["history"]) == epochs
                checks[f"{fold}/{arm}/{stage}/registration"] = state["registration_sha256"] == digest
            for label in ("torch_rng","loader_rng","initial_rng","numpy_rng","python_rng"):
                checks[f"{fold}/{stage}/same_{label}"] = all(equal(states[0][label],s[label]) for s in states[1:])
            checks[f"{fold}/{stage}/same_windows_updates"] = all(
                [(r["windows"],r["train"].get("updates",r["train"].get("optimizer_steps"))) for r in s["history"]] ==
                [(r["windows"],r["train"].get("updates",r["train"].get("optimizer_steps"))) for r in states[0]["history"]] for s in states[1:])
    files = list((DEST/"predictions").glob("**/*.npz"));testfiles = list((DEST/"predictions").glob("**/test/*.npz"))
    checks["36_prediction_files"] = len(files)==36
    checks["both_selections_before_test"] = bool(testfiles) and (DEST/"selection-seal.json").stat().st_mtime_ns < min(p.stat().st_mtime_ns for p in testfiles)
    checks["capacity_exact_parameter_match"] = parameters["ball"] == parameters["capacity"]
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            p = z["probabilities"]
            checks["normalized:"+str(path.relative_to(DEST))] = bool(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6))
    selections = {str(i):read_json(DEST/f"selections/fold-{i}.json") for i in spec["pilots"]}
    results = {part:{str(i):read_json(DEST/f"scores/{part}/fold-{i}.json") for i in spec["pilots"]} for part in ("validation","test")}
    for part,folds in results.items():
        for fold,value in folds.items():
            for scenario in spec["scenarios"]:
                hidden = [value["metrics"][a][scenario]["actual_hidden_fraction"] for a in spec["arms"]]
                checks[f"{part}/{fold}/{scenario}/same_masks"] = len(set(hidden))==1
                if scenario!="clean":
                    checks[f"{part}/{fold}/{scenario}/learned_fallback_exact"] = value["queries"]["ball"][scenario]["unavailable_query_residual_max"]==0
    atomic_json(DEST/"verification.json",{"passed":all(checks.values()),"count":len(checks),"checks":checks})
    success = all(s["gates"]["ball"]["passed"] for s in selections.values())
    atomic_json(DEST/"summary.json",{"selections":selections,"results":results,
        "ball_passes_both_validations":success,"no_automatic_expansion":True,"verification_passed":all(checks.values()),
        "verification_count":len(checks),"parameters":parameters,"training_seconds":training_seconds,
        "updates":updates,"costs":costs,"compute":read_json(DEST/"compute.json"),
        "decision":"Propose independent controlled expansion" if success else "Close this fixed ball-query architecture; retain reference"})
    return 0 if all(checks.values()) else 1


def run(stage):
    record = register();spec = record["protocol"];torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True);digest = file_hash(DEST/"registration.json")
    if stage == "report":
        return finish(record,digest)
    budget = Budget(spec["max_compute_seconds"],DEST);status = "failed"
    lock = DEST/"running.lock"
    with lock.open("x") as stream:
        stream.write("One bounded ball attention pilot only.\n")
    try:
        store = Store(record["records"])
        if not (DEST/"equivalence.json").exists():
            equivalence(store,record)
        if not read_json(DEST/"equivalence.json")["passed"]:
            raise ValueError("Control equivalence required")
        if stage == "equivalence":
            status = "complete";return 0
        for fold in spec["pilots"]:
            path = DEST/f"selections/fold-{fold}.json"
            if path.exists():
                continue
            for arm in spec["arms"]:
                train(store,record,fold,arm,budget,digest)
            scored = score_part(store,record,fold,"validation",budget,digest)
            choice = select(scored["metrics"],spec)
            choice["checkpoint_hashes"] = {a:file_hash(folder(fold,a,"binary")/"last.pt") for a in spec["arms"]}
            atomic_json(path,choice)
        seal = {"registration_sha256":digest,"test_used":False,
                "selections":{str(i):file_hash(DEST/f"selections/fold-{i}.json") for i in spec["pilots"]}}
        path = DEST/"selection-seal.json"
        if path.exists() and read_json(path)!=seal:
            raise ValueError("Seal changed")
        if not path.exists():
            atomic_json(path,seal)
        for fold in spec["pilots"]:
            if not (DEST/f"scores/test/fold-{fold}.json").exists():
                score_part(store,record,fold,"test",budget,digest)
        status = "complete"
    finally:
        budget.close(stage,status);lock.unlink()
    return finish(record,digest)
