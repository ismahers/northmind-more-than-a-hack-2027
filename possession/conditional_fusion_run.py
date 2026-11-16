"""Strict two-way crossfit -> frozen gate -> own exploratory validation.

All data, models and caches are explicit public IDSSE artifacts. Run only via
the output-capturing launcher. Original reference objects remain read-only.
"""
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.conditional_fusion import fit_gate, blend
from possession.conditional_fusion_budget import units, eligible
from possession.football_predictive import features_for, fit as fit_geometry, predict as geometry_predict, combine
from possession.football_predictive_run import SPLITS
from possession.external_data.features import build_features
from possession.external_data.schema import Match, file_hash
from possession.phase7.features import harmonize
from possession.phase7.model import SpatialE0
from possession.learning.data import Windows
from possession.learning.training import seed_everything, epoch_loss, atomic_checkpoint
from possession.phase6.experiment import corrupt, epoch_pass, restore, save_state
from possession.phase6.model import transition_targets, TransitionWindows
from possession.phase8.packing import compact_collate, compact_transition_collate
from possession.phase8.evaluation import predict as neural_predict
from possession.phase5.experiment import scenarios
from possession.temporal_probe.model import TemporalHead
from possession.temporal_probe.windows import EmbeddingWindows, predict as head_predict
from possession.temporal_probe.training import train_epoch as head_epoch
from possession.st_tcn_target import _transform_view, _fuse_member
from possession.temporal_probe.windows import emissions
from possession.hsmm_pilot import load_context, score
from possession.learning.temporal import TemporalDecoder
from possession.phase5.metrics import diagnostics
from possession.transfer_validation import gates

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/conditional-fusion-v1"
PARENT = ROOT/"evaluation/reports/external/football-predictive-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-29"
CONFIG = ROOT/"configs/conditional_fusion_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def sources():
    paths = sorted((ROOT/"possession").rglob("*.py")) + [ROOT/n for n in (
        "scripts/run_local.py", "scripts/conditional_fusion.py", "tests/test_conditional_fusion.py",
        "configs/conditional_fusion_v1.json", "configs/phase8_protocol.json", "configs/temporal_probe_protocol.json",
        "evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for p in paths}


class Budget:
    def __init__(self, limit):
        self.old = read(DEST/"compute.json") if (DEST/"compute.json").exists() else {"seconds": 0., "sessions": []}
        self.started = time.perf_counter(); self.limit = limit
    def used(self):
        return self.old["seconds"]+time.perf_counter()-self.started
    def check(self):
        if self.used() >= self.limit-60:
            raise TimeoutError("Three-hour cumulative budget reached; completed checkpoints preserved")
    def close(self, status):
        elapsed = time.perf_counter()-self.started
        self.old["seconds"] += elapsed
        self.old["sessions"].append({"status": status, "seconds": elapsed})
        write(DEST/"compute.json", self.old)


class CheckedAdamW(torch.optim.AdamW):
    def __init__(self, params, check, **kwargs):
        super().__init__(params, **kwargs); self.check = check
    def step(self, closure=None):
        self.check()
        return super().step(closure)


def progress(budget, stage, **kwargs):
    value = {"stage": stage, "used_seconds": budget.used(), "limit_seconds": budget.limit, **kwargs}
    write(DEST/"public-progress.json", value)
    # Only fixed enums/counters and public model indexes; no input data or errors.
    write(PUBLIC/"conditional-fusion-progress.json", value)


def register():
    p = read(CONFIG)
    if p["max_compute_seconds"] != 10800 or any(p[k] for k in ("test_access", "target_access", "parameter_search", "automatic_promotion")):
        raise ValueError("Fixed three-hour public pilot only")
    parent = read(PARENT/"registration.json"); sealed = read(PARENT/"result-seal.json")
    published = read(PUBLIC/"football-predictive-v1.json")
    if file_hash(PARENT/"registration.json") != published["registration_sha256"] or read(PARENT/"summary.json") != published:
        raise ValueError("Published parent changed")
    dependencies = {"parent/"+n: h for n, h in sealed["outputs"].items()}
    for name, digest in sealed["outputs"].items():
        if file_hash(PARENT/name) != digest:
            raise ValueError("Parent capsule changed")
    risk = read(ROOT/"evaluation/reports/external/sequence-risk-v1/registration.json")
    if file_hash(ROOT/"evaluation/reports/external/sequence-risk-v1/registration.json") != read(PUBLIC/"sequence-risk-v1.json")["registration_sha256"]:
        raise ValueError("Risk provenance changed")
    fusion_root = ROOT/"evaluation/reports/external/scale-selective-v1"
    inherited = {}
    for fold in p["folds"]:
        name = f"evaluation/reports/external/scale-selective-v1/selections/fold-{fold}.json"
        if file_hash(ROOT/name) != risk["dependencies"][name]:
            raise ValueError("Frozen fusion selection changed")
        selected = read(ROOT/name)
        inherited[str(fold)] = {"params": selected["selected_params"]["multiseed_scale_selective"],
                               "primary_seed": selected["teacher"]["seed"]}
        dependencies[name] = file_hash(ROOT/name)
    name = "evaluation/reports/external/scale-selective-v1/registration.json"
    if file_hash(ROOT/name) != risk["dependencies"][name]:
        raise ValueError("Frozen fusion protocol changed")
    fusion_config = read(fusion_root/"registration.json")["protocol"]
    manifest = read(ROOT/"evaluation/reports/external/folds/idsse-only.json")["artifacts"]
    allowed = sorted(set(k for split in SPLITS.values() for k in split["train"]))
    inputs = {}
    for key in allowed:
        path = Path(manifest[key]["path"]).resolve()
        if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or file_hash(path) != manifest[key]["sha256"]:
            raise ValueError("Public training provenance failed")
        inputs[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": manifest[key]["sha256"]}
    stamp = {"protocol": p, "sources": sources(), "units": units(), "splits": SPLITS, "inputs": inputs,
             "dependencies": dependencies, "inherited": inherited, "fusion_config": fusion_config,
             "neural_recipe": read(ROOT/"configs/phase8_protocol.json"),
             "head_recipe": read(ROOT/"configs/temporal_probe_protocol.json"),
             "geometry_recipe": parent["protocol"], "parent_registration_sha256": file_hash(PARENT/"registration.json"),
             "environment": {"numpy": np.__version__, "torch": torch.__version__}}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp:
        raise ValueError("Immutable conditional fusion changed")
    if not marker.exists():
        write(marker, stamp)
    return stamp


class Store:
    def __init__(self, stamp):
        self.stamp = stamp; self.views = {}; self.matches = {}
    def match(self, key, allowed):
        if key not in allowed or key not in self.stamp["inputs"]:
            raise ValueError("Outside allowed inner role")
        if key not in self.matches:
            item = self.stamp["inputs"][key]; path = ROOT/item["path"]
            if file_hash(path) != item["sha256"]:
                raise ValueError("Input changed")
            m = Match.load(path)
            if m.key != key:
                raise ValueError("Public identity mismatch")
            self.matches[key] = m
        return self.matches[key]
    def view(self, key, allowed):
        if key not in allowed:
            raise ValueError("Outside allowed inner role")
        if key not in self.views:
            self.views[key] = harmonize(build_features(self.match(key, allowed)))
        return self.views[key]


def uid(unit):
    return f"fold-{unit['outer_fold']}/inner-{unit['inner_fold']}"


def fresh_teacher(unit, seed, store, stamp, budget):
    spec = dict(stamp["neural_recipe"], seed=seed)
    views = {k: store.view(k, unit["train"]) for k in unit["train"]}
    folder = DEST/"teachers"/uid(unit)/f"seed-{seed}"
    digest = file_hash(DEST/"registration.json")
    for stage, epochs, rate, offset in (("base", 12, .001, 0), ("binary", 8, .0003, 12)):
        path = folder/stage/"last.pt"; path.parent.mkdir(parents=True, exist_ok=True)
        seed_everything(seed); model = SpatialE0("set_transformer", bool(offset))
        if offset:
            previous = torch.load(folder/"base/last.pt", map_location="cpu", weights_only=False)
            missing, extra = model.load_state_dict(previous["model"], strict=False)
            if set(missing) != {"boundary_head.weight", "boundary_head.bias"} or extra:
                raise ValueError("Fresh base contract failed")
            seed_everything(seed+6000)
        optimizer = CheckedAdamW(model.parameters(), budget.check, lr=rate, weight_decay=.0001)
        generator = torch.Generator().manual_seed(seed+(6000 if offset else 0))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        targets = {k: transition_targets(store.match(k, unit["train"]), v, 1, .12) for k, v in views.items()} if offset else None
        if targets:
            positive = sum((y*v[:, None]).sum(0) for y, v in targets.values()); n = sum(v.sum() for _, v in targets.values())
            positive_weights = torch.tensor(np.minimum(40., (n-positive)/np.maximum(1, positive)), dtype=torch.float32)
        history = []
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["registration_sha256"] != digest or state["unit"] != unit:
                raise ValueError("Crossfit checkpoint mismatch")
            history = restore(state, model, optimizer, generator)
        for epoch in range(len(history)+1, epochs+1):
            budget.check(); started = time.perf_counter()
            degraded = corrupt(views, offset+epoch, spec)
            data = TransitionWindows(degraded, targets) if offset else Windows(degraded, 256, 15)
            loader = DataLoader(data, batch_size=4, shuffle=True, generator=generator,
                                collate_fn=compact_transition_collate if offset else compact_collate)
            stats = epoch_pass(model, loader, optimizer, positive_weights, "binary", spec) if offset else epoch_loss(
                model, loader, torch.device("cpu"), optimizer, scaler)
            history.append({"epoch": epoch, "seconds": time.perf_counter()-started, "train": stats})
            save_state(path, model, optimizer, generator, history, {"unit": unit, "seed": seed,
                       "stage": stage, "registration_sha256": digest})
            progress(budget, "teacher", outer_fold=unit["outer_fold"], inner_fold=unit["inner_fold"], seed=seed, phase=stage, epoch=epoch)
    model.eval()
    return model


def geometry_rows(store, key, allowed, stamp, budget):
    folder = DEST/"geometry-cache"/key.replace(":", "_")
    m, v = store.match(key, allowed), store.view(key, allowed)
    result = {}
    for scenario, (_, missing) in scenarios(v, key, {}).items():
        budget.check(); path = folder/(scenario+".npz")
        if not path.exists():
            x, active = features_for(m, v, missing, stamp["geometry_recipe"])
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, active=active, label=np.where(v.alive == 0, 2, v.team),
                                time=v.time, run=v.run)
        with np.load(path, allow_pickle=False) as z:
            result[scenario] = dict(z)
    return result


def fit_inner_geometry(unit, store, stamp, budget):
    x, y, w = [], [], []
    p = stamp["protocol"]
    for key in unit["train"]:
        for scenario, row in geometry_rows(store, key, unit["train"], stamp, budget).items():
            ix = np.concatenate([np.flatnonzero(row["run"] == r)[::5] for r in np.unique(row["run"])])
            keep = ix[row["active"][ix] & (row["label"][ix] < 2) & (row["label"][ix] >= 0)]
            x.append(row["x"][keep]); y.append((row["label"][keep] == 0).astype(float))
            w.append(np.full(len(keep), p["weights"][p["scenarios"].index(scenario)]/2/len(keep)))
    x, y, w = np.concatenate(x), np.concatenate(y), np.concatenate(w)
    folder = DEST/"inner-geometry"/uid(unit); folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"training.npz", x=x, y=y, weight=w)
    models = {}
    for arm in ("proximity", "football"):
        path = folder/(arm+".json")
        if not path.exists():
            model = fit_geometry(x, y, w, arm, stamp["geometry_recipe"], budget.check)
            model["unit"] = unit; write(path, model)
        models[arm] = read(path)
    return models


def inner_heads(unit, primary, store, stamp, budget):
    p = stamp["head_recipe"]; records = {}
    folder = DEST/"heads"/uid(unit)
    for key in unit["train"]:
        v = store.view(key, unit["train"])
        labels = np.where(v.alive == 0, 2, v.team)
        for scenario, (degraded, _) in scenarios(v, key, {}).items():
            budget.check(); path = folder/"embeddings"/key.replace(":", "_")/(scenario+".npz")
            if not path.exists():
                _, _, hidden = neural_predict(primary, degraded, embeddings=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, hidden=hidden, time=v.time, run=v.run, label=labels,
                    condition_weight=np.full(len(v.time), p["condition_weights"][scenario], np.float32))
            with np.load(path, allow_pickle=False) as z:
                records[key+"/"+scenario] = dict(z)
    dataset = EmbeddingWindows(records, 256, 32); models = []
    for temporal in (False, True):
        name = "multiscale" if temporal else "local"; path = folder/(name+".pt")
        seed_everything(p["seed"])
        model = TemporalHead(primary.team_head.weight, primary.team_head.bias, temporal=temporal,
            intervals=False, dilations=p["dilations"], kernel_size=p["kernel_size"], seed=p["seed"])
        opt = CheckedAdamW(model.parameters(), budget.check, lr=.001, weight_decay=.0001)
        generator = torch.Generator().manual_seed(p["seed"]); history = []
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["unit"] != unit or state["registration_sha256"] != file_hash(DEST/"registration.json"):
                raise ValueError("Crossfit head mismatch")
            history = restore(state, model, opt, generator)
        for epoch in range(len(history)+1, 9):
            budget.check(); started = time.perf_counter()
            stats = head_epoch(model, opt, dataset, generator, p, budget)
            history.append({"epoch": epoch, "seconds": time.perf_counter()-started, "train": stats})
            save_state(path, model, opt, generator, history, {"unit": unit, "registration_sha256": file_hash(DEST/"registration.json")})
            progress(budget, "head", outer_fold=unit["outer_fold"], inner_fold=unit["inner_fold"], phase=name, epoch=epoch)
        models.append(model.eval())
    return models


def oof_unit(unit, store, stamp, budget):
    if not eligible(unit["train"], [], unit["heldout"], unit["excluded"]):
        raise ValueError("Invalid OOF provenance")
    p = stamp["protocol"]; inherited = stamp["inherited"][str(unit["outer_fold"])]
    teachers = {seed: fresh_teacher(unit, seed, store, stamp, budget) for seed in p["teacher_seeds"]}
    primary = teachers[inherited["primary_seed"]]
    alternate = teachers[next(s for s in p["teacher_seeds"] if s != inherited["primary_seed"])]
    heads = inner_heads(unit, primary, store, stamp, budget)
    geometric = fit_inner_geometry(unit, store, stamp, budget)
    folder = DEST/"oof"/uid(unit); folder.mkdir(parents=True, exist_ok=True)
    for key_index, key in enumerate(unit["heldout"]):
        v = store.view(key, unit["heldout"])
        geometry = geometry_rows(store, key, unit["heldout"], stamp, budget)
        for scenario, (degraded, _) in scenarios(v, key, {}).items():
            path = folder/f"match-{key_index}-{scenario}.npz"
            if path.exists():
                continue
            mean_z, alive = np.zeros(len(v.time)), np.zeros(len(v.time))
            for swap in (False, True):
                for reflect in (False, True):
                    budget.check(); transformed = _transform_view(degraded, swap, reflect)
                    pp, _, hidden = neural_predict(primary, transformed, embeddings=True)
                    ap, _, _ = neural_predict(alternate, transformed)
                    inputs = {"hidden": hidden, "time": v.time, "run": v.run}
                    zl, _, _ = head_predict(heads[0], inputs)
                    zm, _, _ = head_predict(heads[1], inputs)
                    z, a = _fuse_member(pp, ap, zl, zm, transformed, inherited["params"], stamp["fusion_config"])
                    mean_z += .25*(-1 if swap else 1)*z; alive += .25*a
            base = emissions(mean_z, np.column_stack((alive, np.zeros(len(alive)), 1-alive)))
            row = geometry[scenario]
            np.savez_compressed(path, base=base, x=row["x"], active=row["active"], label=row["label"],
                time=v.time, run=v.run, **{a: geometry_predict(geometric[a], row["x"]) for a in geometric})
            model_paths = [DEST/"teachers"/uid(unit)/f"seed-{s}/binary/last.pt" for s in p["teacher_seeds"]]
            model_paths += [DEST/"heads"/uid(unit)/(a+".pt") for a in ("local", "multiscale")]
            model_paths += [DEST/"inner-geometry"/uid(unit)/(a+".json") for a in geometric]
            write(path.with_suffix(".provenance.json"), {"unit": unit, "key": key, "scenario": scenario,
                "sha256": file_hash(path), "fit_keys": unit["train"], "selection_keys": [],
                "model_hashes": {mp.relative_to(DEST).as_posix(): file_hash(mp) for mp in model_paths},
                "registration_sha256": file_hash(DEST/"registration.json"),
                "inherited_recipe_selected_on": SPLITS[str(unit["outer_fold"])]["validation"],
                "heldout_used_for_fitting": False})
            progress(budget, "oof", outer_fold=unit["outer_fold"], inner_fold=unit["inner_fold"], match_index=key_index, scenario=scenario)


def collect_gates(fold, stamp, budget):
    all_rows = []
    p = stamp["protocol"]
    for unit in stamp["units"]:
        if unit["outer_fold"] != fold:
            continue
        for key_index, key in enumerate(unit["heldout"]):
            for scenario in p["scenarios"]:
                path = DEST/"oof"/uid(unit)/f"match-{key_index}-{scenario}.npz"
                prov = read(path.with_suffix(".provenance.json"))
                if prov["sha256"] != file_hash(path) or not eligible(prov["fit_keys"], prov["selection_keys"], unit["heldout"], unit["excluded"]):
                    raise ValueError("OOF data provenance mismatch")
                if prov["registration_sha256"] != file_hash(DEST/"registration.json") or any(
                    file_hash(DEST/name) != digest for name, digest in prov["model_hashes"].items()):
                    raise ValueError("OOF expert identity changed")
                with np.load(path, allow_pickle=False) as z:
                    ix = np.concatenate([np.flatnonzero(z["run"] == r)[::5] for r in np.unique(z["run"])])
                    ix = ix[z["active"][ix] & (z["label"][ix] < 2) & (z["label"][ix] >= 0)]
                    base = z["base"][ix]; alive = base[:, :2].sum(-1)
                    row = {"base_home": np.divide(base[:, 0], alive, out=np.full(len(ix), .5), where=alive > 1e-12),
                        "x": z["x"][ix], "y": (z["label"][ix] == 0).astype(float),
                        "weight": np.full(len(ix), p["weights"][p["scenarios"].index(scenario)]/4/len(ix)),
                        "proximity": z["proximity"][ix], "football": z["football"][ix]}
                all_rows.append(row)
    rows = {name: np.concatenate([r[name] for r in all_rows]) for name in all_rows[0]}
    folder = DEST/"gates"; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/f"fold-{fold}-training.npz", **rows)
    models = {}
    for arm in ("proximity", "football"):
        model = fit_gate(rows["base_home"], rows[arm], rows["x"], rows["y"], rows["weight"], p, budget.check)
        model["outer_fold"] = fold; write(folder/f"fold-{fold}-{arm}.json", model); models[arm] = model
    return models


def objective(rows, p):
    return sum(w*rows[s]["spellF1"] for w, s in zip(p["weights"], p["scenarios"]))


def decisions(metrics, p, parent_p):
    refs = ("proximity_gate", "fixed_proximity", "fixed_football", "four_view_v2", "registered_multiseed")
    out = {r: gates(metrics["football_gate"], metrics[r], parent_p) for r in refs}
    out["gain_vs_proximity_gate"] = objective(metrics["football_gate"], p)-objective(metrics["proximity_gate"], p)
    out["gain_vs_four_view"] = objective(metrics["football_gate"], p)-objective(metrics["four_view_v2"], p)
    out["long_fragmentation"] = all(metrics["football_gate"]["clean"]["long_fragmented_gt"] <= metrics[r]["clean"]["long_fragmented_gt"] for r in refs)
    out["false_HA"] = all(metrics["football_gate"][s]["false_transitions_per_minute"]
        <= metrics[r][s]["false_transitions_per_minute"]+.1 for r in refs for s in p["scenarios"])
    out["passed"] = (all(out[r]["passed"] for r in refs) and out["long_fragmentation"] and out["false_HA"]
        and out["gain_vs_proximity_gate"] >= p["min_gain_vs_proximity_gate"]
        and out["gain_vs_four_view"] >= p["min_gain_vs_four_view"])
    return out


def validation_case(context, prior, data, geometry, gate, p, parent_p):
    probabilities = {a: data[a] for a in ("registered_multiseed", "four_view_v2")}
    alphas = {}
    for arm in ("proximity", "football"):
        h = geometry_predict(geometry[arm], data["x"])
        probabilities["fixed_"+arm] = combine(data["four_view_v2"], h, data["active"], parent_p)
        probabilities[arm+"_gate"], alphas[arm] = blend(data["four_view_v2"], h, data["x"], data["active"], gate[arm])
    metrics, hard = {}, {}
    for name, prob in probabilities.items():
        pred = context.decoded(prob, TemporalDecoder(parent_p["decoder"], prior))
        row = score(context, pred); diag = diagnostics(context, pred)
        row.update(long_fragmented_gt=diag["duration_groups"]["ge10"]["fragmented_gt"], boundaries_HA=diag["decoded_HA"])
        metrics[name] = row; hard[name] = pred
    return metrics, hard, {a: {"mean": float(v.mean()), "p10_p50_p90": np.quantile(v, [.1, .5, .9]).tolist()} for a, v in alphas.items()}


def evaluate(stamp, models, budget):
    p = stamp["protocol"]; result = {}
    for fold in p["folds"]:
        folder = DEST/f"validation/fold-{fold}"; folder.mkdir(parents=True, exist_ok=True)
        for name in ("context", "priors"):
            (folder/(name+".npz")).write_bytes((PARENT/f"validation/fold-{fold}/{name}.npz").read_bytes())
        geometry = {}
        for arm in ("proximity", "football"):
            path = PARENT/f"models/fold-{fold}-{arm}.json"
            (folder/(arm+".json")).write_bytes(path.read_bytes()); geometry[arm] = read(path)
        context = load_context(folder/"context.npz")
        with np.load(folder/"priors.npz", allow_pickle=False) as z:
            prior = dict(z)
        metrics = {r: {} for r in ("registered_multiseed", "four_view_v2", "fixed_proximity", "fixed_football", "proximity_gate", "football_gate")}
        weights = {}
        for scenario in p["scenarios"]:
            budget.check()
            source = PARENT/f"validation/fold-{fold}/{scenario}.npz"
            with np.load(source, allow_pickle=False) as z:
                data = {k: z[k] for k in ("x", "active", "registered_multiseed", "four_view_v2")}
            rows, hard, weights[scenario] = validation_case(context, prior, data, geometry, models[str(fold)], p, stamp["geometry_recipe"])
            for name, row in rows.items():
                metrics[name][scenario] = row
            np.savez_compressed(folder/(scenario+".npz"), **data, **{"hard__"+k: v for k, v in hard.items()})
            old = read(PUBLIC/"football-predictive-v1.json")["folds"][str(fold)]["metrics"]
            for reference, previous in (("fixed_proximity", "proximity"), ("fixed_football", "football"),
                                        ("registered_multiseed", "registered_multiseed"), ("four_view_v2", "four_view_v2")):
                if not np.isclose(rows[reference]["spellF1"], old[previous][scenario]["spellF1"], atol=1e-12, rtol=0):
                    raise ValueError("Reference parity failed")
            progress(budget, "validation", outer_fold=fold, scenario=scenario)
        result[str(fold)] = {"metrics": metrics, "weight_summaries": weights, "gates": decisions(metrics, p, stamp["geometry_recipe"]), "reference_parity": True}
    return result


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST/"running.lock"
    with lock.open("x") as stream:
        stream.write("One bounded conditional fusion pilot\n")
    budget = Budget(read(CONFIG)["max_compute_seconds"]); status = "failed"
    try:
        stamp = register(); p = stamp["protocol"]
        if (DEST/"result-seal.json").exists():
            status = "verified_existing"
            return verify()
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        store = Store(stamp)
        for unit in stamp["units"]:
            oof_unit(unit, store, stamp, budget)
        models = {str(f): collect_gates(f, stamp, budget) for f in p["folds"]}
        write(DEST/"gates-seal.json", {"validation_started": False,
              "models": {path.relative_to(DEST).as_posix(): file_hash(path) for path in (DEST/"gates").glob("*.json")}})
        summary = {"folds": evaluate(stamp, models, budget), "registration_sha256": file_hash(DEST/"registration.json"),
                   "test_accessed": False, "target_accessed": False, "automatic_promotion": False,
                   "seconds": budget.used(), "crossfit_completed": True, "validation_exploratory": True}
        summary["passes_both"] = all(r["gates"]["passed"] for r in summary["folds"].values())
        write(DEST/"summary.json", summary)
        selected = [DEST/"summary.json", DEST/"gates-seal.json"]
        for name in ("teachers", "heads", "inner-geometry", "oof", "gates", "validation"):
            selected.extend(path for path in (DEST/name).rglob("*") if path.is_file() and "embeddings" not in path.parts)
        write(DEST/"result-seal.json", {"registration_sha256": summary["registration_sha256"],
              "outputs": {path.relative_to(DEST).as_posix(): file_hash(path) for path in selected}})
        status = "complete"; return verify()
    except TimeoutError:
        status = "budget_reached"; progress(budget, status)
        write(DEST/"budget-stop.json", {"status": status, "used_seconds": budget.used(), "validation_completed": (DEST/"summary.json").exists()})
        return 1
    finally:
        budget.close(status); lock.unlink()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Conditional fusion source changed")
    for name, digest in seal["outputs"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Conditional fusion output changed")
    gates_seal = read(DEST/"gates-seal.json")
    if gates_seal["validation_started"] or len(gates_seal["models"]) != 4:
        raise ValueError("Invalid pre-validation seal")
    for name, digest in gates_seal["models"].items():
        if file_hash(DEST/name) != digest:
            raise ValueError("Gate changed after seal")
    summary = read(DEST/"summary.json")
    for row in summary["folds"].values():
        if row["gates"] != decisions(row["metrics"], stamp["protocol"], stamp["geometry_recipe"]):
            raise ValueError("Gate decision mismatch")
    return 0


def replay():
    verify(); stamp = read(DEST/"registration.json"); p = stamp["protocol"]
    torch.set_num_threads(4); result = read(DEST/"summary.json")
    for fold in p["folds"]:
        models = {}
        with np.load(DEST/f"gates/fold-{fold}-training.npz", allow_pickle=False) as z:
            for arm in ("proximity", "football"):
                models[arm] = read(DEST/f"gates/fold-{fold}-{arm}.json")
                fitted = fit_gate(z["base_home"], z[arm], z["x"], z["y"], z["weight"], p)
                np.testing.assert_allclose(fitted["theta"], models[arm]["theta"], atol=1e-10, rtol=0)
        folder = DEST/f"validation/fold-{fold}"; context = load_context(folder/"context.npz")
        geometry = {a: read(folder/(a+".json")) for a in ("proximity", "football")}
        with np.load(folder/"priors.npz", allow_pickle=False) as z:
            prior = dict(z)
        for scenario in p["scenarios"]:
            with np.load(folder/(scenario+".npz"), allow_pickle=False) as z:
                rows, hard, weights = validation_case(context, prior, dict(z), geometry, models, p, stamp["geometry_recipe"])
                if weights != result["folds"][str(fold)]["weight_summaries"][scenario]:
                    raise ValueError("Weight replay differs")
                for name, pred in hard.items():
                    np.testing.assert_array_equal(pred, z["hard__"+name])
                    if rows[name] != result["folds"][str(fold)]["metrics"][name][scenario]:
                        raise ValueError("Metric replay differs")
    return 0


def export():
    complete = (DEST/"result-seal.json").exists()
    if complete:
        verify(); seal = read(DEST/"result-seal.json")
        files = sorted([*seal["outputs"], "registration.json", "result-seal.json", "compute.json"])
        public_summary = read(DEST/"summary.json")
    else:
        stop = read(DEST/"budget-stop.json")
        if stop["status"] != "budget_reached" or read(DEST/"registration.json")["sources"] != sources():
            raise ValueError("Only intact budget-stopped partial runs may be exported")
        files = ["registration.json", "compute.json", "budget-stop.json", "public-progress.json"]
        for directory in ("teachers", "heads", "inner-geometry", "oof", "gates", "validation"):
            files.extend(p.relative_to(DEST).as_posix() for p in (DEST/directory).rglob("*")
                         if p.is_file() and "embeddings" not in p.parts and not p.name.endswith(".tmp"))
        files = sorted(set(files))
        public_summary = {"complete": False, "reason": "three_hour_budget_reached",
            "registration_sha256": file_hash(DEST/"registration.json"), "compute": read(DEST/"compute.json"),
            "validation_completed": False, "automatic_promotion": False,
            "test_accessed": False, "target_accessed": False}
    archive = ROOT/"evaluation/reports/handoff/northmind-conditional-fusion-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in files}
    prefix = "evaluation/reports/external/conditional-fusion-v1/"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            info = zipfile.ZipInfo(prefix+name, (2026, 10, 29, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (DEST/name).read_bytes())
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != v["sha256"] for n, v in manifest.items()):
            raise ValueError("Archive verification failed")
    write(PUBLIC/"conditional-fusion-v1.json", public_summary)
    write(PUBLIC/"conditional-fusion-v1-package.json", {"archive": archive.name, "sha256": file_hash(archive),
          "bytes": archive.stat().st_size, "files": len(files), "verified": True, "complete": complete,
          "scope": "Public auxiliary OOF checkpoints, gate training data/models, validation replay; no original tracking or logs"})
    return 0
