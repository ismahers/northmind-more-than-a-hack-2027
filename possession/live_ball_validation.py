"""One frozen external learned-live ablation, independent of control memory.

Only a fold's training labels fit its binary transition prior. Validation
labels enter the unchanged scorer after prediction. No target or test loader
is called. All data-dependent execution requires the reviewed local launcher.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np

from possession.control_memory_validation import (build_native_geometry,
    geometric_reference, geometric_dead, expand_hard, gates, METRICS, SCENARIOS,
    GUARDS, safe_public_summary as _checked_public)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/live_ball_validation_v1.json"
DEST = ROOT/"evaluation/reports/external/live-ball-validation-v1"
PUBLIC = ROOT/"docs/public-results/2026-11-17/live-ball-v1.json"
BASELINE = ROOT/"docs/public-results/2026-11-17/control-memory-v1.json"
RECEIPT = ROOT/"docs/public-results/2026-11-17/live-ball-tests.json"
STATUS = ROOT/"docs/public-results/2026-11-17/live-ball-status.json"
_PHASE = "setup"
REVIEWED = ("possession/live_ball.py", "possession/live_ball_validation.py",
    "tests/test_live_ball.py", "tests/test_live_ball_validation.py",
    "scripts/live_ball.py", "configs/live_ball_validation_v1.json")


def validate_protocol(p):
    expected = dict(version="live-ball-validation-v1",part="validation",folds=[0,6],
        scenarios=list(SCENARIOS),weights=[.5,.25,.25],
        views=[[False,False],[False,True],[True,False],[True,True]],
        fps=25.,model_fps=10.,cpu_threads=4,max_seconds=3600,
        train_models=False,fit_transition_prior_on_train=True,test_access=False,
        target_access=False,parameter_search=False,binary_hmm_strength=.5,
        preserve_geometric_dead=True,neutral_tagging=True,memory_enabled=False,guardrails=GUARDS,
        target_prior_policy_if_promoted="equal_average_of_fold_0_and_6_transition_probabilities")
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError("Unexpected fixed live-ball protocol")
    for key in ("train_models","test_access","target_access","parameter_search","memory_enabled"):
        if p[key] is not False:
            raise ValueError("Invalid access barrier")
    for key in ("fit_transition_prior_on_train","preserve_geometric_dead","neutral_tagging"):
        if p[key] is not True:
            raise ValueError("Invalid fixed ablation contract")


def _source_closure(entries):
    """Static local import closure; no discovery of unrelated future modules.

    AST inspection reads Python source only. Imports are never executed here.
    A source path must stay inside ROOT and end in .py, including any package
    __init__ files. Third-party packages are represented by environment versions.
    """
    root = ROOT.resolve()
    pending = [root/entry for entry in entries]
    found = set()
    def add_module(name):
        if not name or not all(part.isidentifier() for part in name.split(".")):
            return
        parts = name.split(".")
        candidate = root.joinpath(*parts).with_suffix(".py")
        package = root.joinpath(*parts,"__init__.py")
        if candidate.is_file():
            pending.append(candidate)
        elif package.is_file():
            pending.append(package)
        for index in range(1,len(parts)):
            init = root.joinpath(*parts[:index],"__init__.py")
            if init.is_file():
                pending.append(init)
    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(root) or path.suffix!=".py":
            raise ValueError("Invalid source closure path")
        if path in found:
            continue
        found.add(path)
        module = path.relative_to(root).with_suffix("").parts
        package_parts = list(module[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    if node.level>len(package_parts)+1:
                        raise ValueError("Invalid relative import")
                    prefix = package_parts[:len(package_parts)-node.level+1]
                    base = ".".join(prefix+([node.module] if node.module else []))
                else:
                    base = node.module or ""
                add_module(base)
                for alias in node.names:
                    if alias.name!="*":
                        add_module(".".join(filter(None,(base,alias.name))))
    return sorted(found)


def _validate_receipt(record):
    if (record.get("synthetic_only") is not True or record.get("success") is not True
            or record.get("tests",0)<20 or record.get("failures")!=0
            or record.get("errors")!=0 or record.get("skipped")!=0
            or set(record.get("sources_sha256_lf",{}))!=set(REVIEWED)):
        raise ValueError("Current synthetic receipt required")
    for name in REVIEWED:
        data = (ROOT/name).read_text(encoding="utf-8-sig").replace("\r\n","\n").encode("utf-8")
        if record["sources_sha256_lf"][name]!=hashlib.sha256(data).hexdigest():
            raise ValueError("Reviewed source changed")


def setup():
    from possession import st_tcn_target as transfer
    from possession.evidence_first_validation import reference_path
    from possession.external_data.schema import file_hash
    from possession.temporal_probe.common import splits_valid, public_path
    import torch
    p = transfer._json(PROTOCOL)
    validate_protocol(p)
    _validate_receipt(transfer._json(RECEIPT))
    config, heads, members, dependencies = transfer._transfer_spec()
    if [m["fold"] for m in members]!=[0,6]:
        raise ValueError("Wrong folds")
    manifest = ROOT/"evaluation/reports/external/folds/idsse-only.json"
    records = transfer._json(manifest)["artifacts"]
    used = {}
    for member in members:
        split = transfer._json(transfer.FUSION_DIR/f"selections/fold-{member['fold']}.json")["teacher"]["split"]
        if not splits_valid(split):
            raise ValueError("Invalid fold roles")
        used[str(member["fold"])] = {}
        for key in split["train"]+[split["validation"]]:
            digest = file_hash(public_path(records[key]["path"]))
            if digest!=records[key]["sha256"]:
                raise ValueError("Changed public input")
            used[str(member["fold"])][key] = digest
        for scenario in SCENARIOS:
            path = reference_path(member["fold"],scenario)
            dependencies[str(path)] = file_hash(path)
    sources = {str(path.relative_to(ROOT)):file_hash(path) for path in _source_closure(
        [name for name in REVIEWED if name.endswith(".py")]+["scripts/run_local.py"])}
    sources[str(PROTOCOL.relative_to(ROOT))] = file_hash(PROTOCOL)
    for path in (manifest,BASELINE,RECEIPT):
        dependencies[str(path)] = file_hash(path)
    stamp = dict(protocol=p,sources=sources,dependencies=dependencies,
        fold_train_validation_sha256=used,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
    return p,config,heads,members,records,stamp


def fit_fold_prior(store,split):
    """Enforce roles before even requesting any training feature view."""
    from possession.temporal_probe.common import splits_valid
    from possession.live_ball import fit_binary_prior
    if not splits_valid(split):
        raise ValueError("Invalid prior roles")
    return fit_binary_prior([store.harmonized(key) for key in split["train"]],fps=10.)


def _neural(view,degraded,member,models,heads,config,reference,check_budget):
    """Recompute BOTH neural outputs from four sealed public-model views."""
    from possession import st_tcn_target as transfer
    from possession.phase8.evaluation import predict as predict_teacher
    from possession.temporal_probe.windows import predict as predict_head, emissions
    z_mean,alive_mean = np.zeros(len(view.time)),np.zeros(len(view.time))
    for swap,reflect in ((False,False),(False,True),(True,False),(True,True)):
        check_budget()
        transformed = transfer._transform_view(degraded,swap,reflect)
        p,_,hidden = predict_teacher(models[0],transformed,embeddings=True)
        alt,_,_ = predict_teacher(models[1],transformed)
        inputs = dict(hidden=hidden,time=view.time,run=view.run)
        local,_,_ = predict_head(heads[0],inputs,config["core_frames"],config["batch_size"])
        multi,_,_ = predict_head(heads[1],inputs,config["core_frames"],config["batch_size"])
        z,alive = transfer._fuse_member(p,alt,local,multi,transformed,member["params"],config)
        if not swap and not reflect:
            actual = emissions(z,np.column_stack((alive,np.zeros(len(alive)),1-alive)))
            if actual.shape!=reference.shape or not np.allclose(actual,reference,atol=3e-6,rtol=3e-5):
                raise ValueError("Frozen identity parity failed")
        z_mean += .25*(-1 if swap else 1)*z
        alive_mean += .25*alive
    if not np.isfinite(alive_mean).all() or np.any((alive_mean < -1e-6)|(alive_mean > 1+1e-6)):
        raise ValueError("Invalid neural live probability")
    return np.tanh(z_mean/2.2),np.clip(alive_mean,0,1)


def _decode_reference(values,raw_geom,base,seq):
    from possession.sequence_decoding import (runs,_smooth_strength,_adaptive_switch_costs,
        decode_control,_filter_oscillations,_align_transitions_to_geometry)
    geom_strength = _smooth_strength(raw_geom,seq.strength_smooth_frames)
    smoothed = _smooth_strength(base,seq.strength_smooth_frames)
    alignment,labels = values.copy(),values.copy()
    for a,b in runs(np.isin(values,("H","A"))):
        alignment[a:b] = decode_control(geom_strength[a:b],
            _adaptive_switch_costs(geom_strength[a:b],seq),fallback=values[a])
        labels[a:b] = decode_control(smoothed[a:b],
            _adaptive_switch_costs(smoothed[a:b],seq),fallback=values[a])
    alignment = _filter_oscillations(alignment)
    return _align_transitions_to_geometry(_filter_oscillations(labels),alignment,20)


def decode_pair(native,neural,learned_live,pitch):
    """Same H/A evidence; replace live/dead mask, protecting geometric dead cues.

    Geometric outside/still evidence remains protected. Newly live samples
    formerly marked D use sign of unsmoothed fused strength (exact tie H) for
    DP fallback. Smoothing stays segment-wide and identical to the baseline.
    No memory signal or learned-live probability enters H/A strength.
    """
    from possession.inference import Config,team_strength
    from possession.sequence_decoding import SequenceConfig,_ball_kinematics_weights
    signal,live = np.asarray(neural,float),np.asarray(learned_live)
    if (signal.shape!=native.time.shape or not np.isfinite(signal).all() or np.any(abs(signal)>1)
            or live.shape!=signal.shape or live.dtype!=np.bool_):
        raise ValueError("Invalid external decoder inputs")
    cfg,seq = Config(),SequenceConfig()
    old,new = np.full(len(signal),"X",dtype="<U1"),np.full(len(signal),"X",dtype="<U1")
    for identity in np.unique(native.run):
        ix = np.flatnonzero(native.run==identity)
        dist,ball = native.distances[ix],native.ball[ix]
        reference = geometric_reference(dist,ball,pitch,cfg)
        protected = geometric_dead(dist,ball,pitch,cfg)
        raw = np.clip(team_strength(dist,cfg,None)*_ball_kinematics_weights(
            ball,dist,pass_speed_threshold_ms=seq.pass_speed_threshold_ms),-1,1)
        weight = np.where(np.isfinite(dist.min(1)),seq.neural_weight_observed,seq.neural_weight_occluded)
        base = np.clip((1-.5*weight)*raw+weight*signal[ix],-1,1)
        candidate = reference.copy()
        new_dead = protected | ~live[ix]
        newly_live = ~new_dead & ~np.isin(reference,("H","A"))
        candidate[newly_live] = np.where(base[newly_live]>=0,"H","A")
        candidate[new_dead] = "D"
        old[ix] = _decode_reference(reference,raw,base,seq)
        new[ix] = _decode_reference(candidate,raw,base,seq)
        if np.any(old[ix][protected]!="D") or np.any(new[ix][protected]!="D"):
            raise ValueError("Geometric dead protection failed")
        if not np.array_equal(old[ix]=="D",reference=="D") or not np.array_equal(new[ix]=="D",new_dead):
            raise ValueError("Decoder changed requested dead mask")
    return old,new


def assert_baseline(actual,expected):
    if set(actual)!=set(METRICS) or set(expected)!=set(METRICS):
        raise ValueError("Baseline metric schema changed")
    for name in METRICS:
        if not np.isclose(actual[name],expected[name],atol=1e-10,rtol=0):
            raise ValueError("Historical baseline parity failed")


def checked_prior(prior):
    """Whitelist the tiny own-training artifact needed for portable inference."""
    if set(prior)!={"state_order","transition_probabilities"} or prior["state_order"]!=["D","L"]:
        raise ValueError("Invalid public binary prior schema")
    rows = prior["transition_probabilities"]
    if not isinstance(rows,list) or len(rows)!=2 or any(not isinstance(row,list) or len(row)!=2 for row in rows):
        raise ValueError("Invalid public binary prior matrix")
    for row in rows:
        for value in row:
            if (isinstance(value,bool) or not isinstance(value,(int,float))
                    or not np.isfinite(value) or value<=0 or value>=1):
                raise ValueError("Invalid public binary transition probability")
    matrix = np.asarray(rows,float)
    if not np.allclose(matrix.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError("Invalid public transition row sum")
    return dict(state_order=["D","L"],transition_probabilities=matrix.tolist())


def safe_public_summary(raw):
    # Reuse the reviewed pure metric whitelist with a temporary variant name;
    # no old setup/verification or mutable artifact is called.
    rewritten = {key:raw[key] for key in ("registration_sha256","test_accessed",
        "target_accessed","calibration_accessed","automatic_promotion")}
    rewritten["folds"] = {}
    if set(raw["folds"])!={"0","6"}:
        raise ValueError("Incomplete external validation")
    for fold,row in raw["folds"].items():
        if (row["baseline_parity"] is not True or set(row["metrics"])!={"baseline","live_ball"}):
            raise ValueError("Missing historical baseline parity")
        rewritten["folds"][fold] = dict(metrics={"baseline":row["metrics"]["baseline"],
            "memory":row["metrics"]["live_ball"]},gates=row["gates"],identity_parity=row["identity_parity"])
    public = _checked_public(rewritten)
    public.update(scope="external_seen_validation_learned_live_neutral_tag_ablation",
        memory_used=False,transition_prior_fit_on_fold_train_only=True)
    if set(raw["transition_priors"])!={"0","6"}:
        raise ValueError("Incomplete public binary priors")
    public["transition_priors"] = {fold:checked_prior(raw["transition_priors"][fold]) for fold in ("0","6")}
    for row in public["folds"].values():
        row["metrics"]["live_ball"] = row["metrics"].pop("memory")
        row["baseline_parity"] = True
    return public


def _write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")


def _status(state,error=None):
    classes = {"ValueError","TypeError","KeyError","IndexError","OSError","FileNotFoundError",
        "FileExistsError","PermissionError","TimeoutError","RuntimeError","AssertionError",
        "ImportError","ModuleNotFoundError","MemoryError"}
    if state not in {"running","failed","complete"} or _PHASE not in {
            "setup","prior","parity","adapt","score","seal","complete"}:
        raise ValueError("Invalid diagnostic enum")
    name = None if error is None else type(error).__name__
    output = dict(state=state,phase=_PHASE,exception_class=name if name in classes else
        ("Exception" if name else None))
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps(output,sort_keys=True)+"\n",encoding="utf-8")


@contextmanager
def _exclusive():
    DEST.mkdir(parents=True,exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x",encoding="utf-8") as stream:
        stream.write("live-ball-validation-v1\n")
    try:
        yield
    finally:
        lock.unlink()


def run():
    global _PHASE
    _PHASE = "setup"
    _status("running")
    try:
        result = _run()
    except Exception as error:
        _status("failed",error)
        raise
    _PHASE = "complete"
    _status("complete")
    return result


def _run():
    global _PHASE
    from possession import st_tcn_target as transfer
    from possession.evidence_first_validation import FoldStore,load_models,reference_path
    from possession.external_data.schema import file_hash
    from possession.learning.study import ScoreContext
    from possession.live_ball import decode_live_probability
    from possession.phase5.experiment import scenarios
    from possession.hsmm_pilot import score
    import torch
    start = time.perf_counter()
    with _exclusive():
        protocol,config,head_config,members,records,stamp = setup()
        marker = DEST/"registration.json"
        if marker.exists():
            if transfer._json(marker)!=stamp:
                raise ValueError("Immutable live-ball registration changed")
        else:
            _write_new(marker,stamp)  # Before any fit, validation GT or metric computation.
        if (DEST/"result-seal.json").exists():
            return verify()
        if (DEST/"summary.json").exists():
            raise ValueError("Unsealed result requires review")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        def budget():
            if time.perf_counter()-start>=3600:
                raise TimeoutError("Fixed external validation budget exhausted")
        parent = transfer._json(transfer.FUSION_DIR/"registration.json")
        historical = transfer._json(BASELINE)["folds"]
        summaries,priors = {},{}
        for member in members:
            budget()
            fold = member["fold"]
            split = transfer._json(transfer.FUSION_DIR/f"selections/fold-{fold}.json")["teacher"]["split"]
            store = FoldStore(records,split)
            _PHASE = "prior"
            prior = fit_fold_prior(store,split)
            priors[str(fold)] = checked_prior(dict(state_order=list(prior["state_order"]),
                transition_probabilities=np.asarray(prior["transition_probabilities"]).tolist()))
            match = store.match(split["validation"])
            view = store.harmonized(split["validation"])
            models,heads = load_models(member,split,head_config)
            rows,parity = {"baseline":{},"live_ball":{}},{}
            for scenario,(degraded,hidden) in scenarios(view,split["validation"],parent["teacher_protocol"]).items():
                if scenario not in SCENARIOS:
                    raise ValueError("Unexpected validation scenario")
                with np.load(reference_path(fold,scenario),allow_pickle=False) as z:
                    reference = z["probabilities"].copy()
                if reference.shape!=(len(view.time),3) or not np.isfinite(reference).all():
                    raise ValueError("Invalid frozen reference")
                _PHASE = "parity"
                team,alive = _neural(view,degraded,member,models,heads,config,reference,budget)
                parity[scenario] = True
                _PHASE = "adapt"
                live = decode_live_probability(alive,view.time,view.run,prior,fps=10.,strength=.5)
                native = build_native_geometry(match,view,hidden)
                baseline,candidate = decode_pair(native,team[native.anchor_index],live[native.anchor_index],
                    match.metadata["pitch_m"])
                _PHASE = "score"
                context = ScoreContext(match,SimpleNamespace(time=native.time,run=native.run,
                    native_index=native.native_index))
                measured = score(context,expand_hard(context,baseline))
                rows["baseline"][scenario] = {k:float(measured[k]) for k in METRICS}
                assert_baseline(rows["baseline"][scenario],historical[str(fold)]["metrics"]["baseline"][scenario])
                measured = score(context,expand_hard(context,candidate))
                rows["live_ball"][scenario] = {k:float(measured[k]) for k in METRICS}
                budget()
            summaries[str(fold)] = dict(metrics=rows,gates=gates(rows["live_ball"],rows["baseline"]),
                identity_parity=parity,baseline_parity=True)
        _PHASE = "seal"
        if setup()[-1]!=stamp:
            raise ValueError("Dependencies changed during external validation")
        budget()
        result = dict(folds=summaries,registration_sha256=file_hash(marker),seconds=time.perf_counter()-start,
            test_accessed=False,target_accessed=False,calibration_accessed=False,automatic_promotion=False,
            transition_priors=priors)
        safe_public_summary(result)
        _write_new(DEST/"summary.json",result)
        _write_new(DEST/"result-seal.json",dict(registration_sha256=file_hash(marker),
            outputs={"summary.json":file_hash(DEST/"summary.json")}))
    return verify()


def verify():
    from possession import st_tcn_target as transfer
    from possession.external_data.schema import file_hash
    stamp = setup()[-1]
    marker = DEST/"registration.json"
    if transfer._json(marker)!=stamp:
        raise ValueError("Changed live-ball registration")
    seal = transfer._json(DEST/"result-seal.json")
    if (seal["registration_sha256"]!=file_hash(marker) or set(seal["outputs"])!={"summary.json"}
            or seal["outputs"]["summary.json"]!=file_hash(DEST/"summary.json")):
        raise ValueError("Invalid live-ball seal")
    raw = transfer._json(DEST/"summary.json")
    if raw["registration_sha256"]!=file_hash(marker):
        raise ValueError("Invalid result provenance")
    safe_public_summary(raw)
    historical = transfer._json(BASELINE)["folds"]
    for fold,row in raw["folds"].items():
        for scenario in SCENARIOS:
            assert_baseline(row["metrics"]["baseline"][scenario],historical[fold]["metrics"]["baseline"][scenario])
    return 0


def export():
    from possession import st_tcn_target as transfer
    verify()
    public = safe_public_summary(transfer._json(DEST/"summary.json"))
    if PUBLIC.exists():
        if transfer._json(PUBLIC)!=public:
            raise ValueError("Immutable public live-ball export changed")
    else:
        _write_new(PUBLIC,public)
    return 0
