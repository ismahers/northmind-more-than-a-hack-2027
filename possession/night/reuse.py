"""Two reuse-only ablations: checkpoint criterion and probability ensembles."""

import json
import time

import numpy as np
import torch

from possession.learning.corruption import hide_ball, stable_seed, block_mask
from possession.learning.metrics import score_match
from possession.learning.study import ScoreContext, save_probabilities
from possession.learning.temporal import TemporalDecoder
from possession.learning.training import TrainConfig, new_model
from scripts.phase3 import CachedRunDecoder, metric_row
from scripts.train_e0 import manifest
from .cache import PublicStore
from .common import ROOT, DEST, OLD, atomic_json, objective, robust_eligible, progress, source_hashes
from .decoding_study import load_predictions, quick_scores, guard_reasons, selected_source
from .training import cap_result, old_directory, infer


def register_reuse():
    path = ROOT / "configs/night_reuse_protocol.json"
    value = json.loads(path.read_text())
    frozen = DEST / "reuse_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != value:
        raise ValueError("Reuse-only protocol changed")
    if not frozen.exists():
        atomic_json(frozen, value)
    source = {"protocol": value, "source": source_hashes(["possession/night/reuse.py", "possession/night/training.py",
        "possession/night/decoding_study.py", "possession/night/cache.py"])}
    source_path = DEST / "reuse_source.json"
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise ValueError("Reuse-only source changed; preserve the existing run")
    if not source_path.exists():
        atomic_json(source_path, source)
    return value


def mean_probabilities(arrays):
    return np.mean(np.asarray(arrays, dtype=np.float64), axis=0).astype(np.float32)


def validation_seed(store, data, i, condition, seed, protocol):
    previous = json.loads((OLD / f"selections/fold-{i}.json").read_text())
    cap = previous["conditions"][condition]["cap"]
    key = data["folds"][i]["validation"][0]
    values, model = {}, None
    for name in protocol["primary_scenarios"]:
        if seed == 2027:
            _, p = load_predictions(OLD / f"validation_predictions/fold-{i}/{condition}/{name}.npz")
        elif name == "clean":
            _, p = load_predictions(OLD / f"training/{condition}/seed-{seed}/fold-{i}/validation-cap-{cap}.npz")
        else:
            path = DEST / f"reuse_predictions/validation/fold-{i}/seed-{seed}/{name}.npz"
            if path.exists():
                _, p = load_predictions(path)
            else:
                view = store.view(key)
                if model is None:
                    state = torch.load(OLD / f"training/{condition}/seed-{seed}/fold-{i}/cap-{cap}.pt", map_location="cpu", weights_only=False)
                    assert key in state["fingerprint"]["fold"]["validation"]
                    assert not set(state["fingerprint"]["fold"]["train"]) & set(data["folds"][i]["test"])
                    model = new_model(TrainConfig.from_dict(state["config"]))
                    model.load_state_dict(state["model"])
                length = 5. if name == "missing40_5s" else 15.
                corrupted, _ = hide_ball(view, .4, length, stable_seed(key, 701))
                p = infer(model, corrupted, protocol)
                save_probabilities(path, view, p)
        values[name] = p
    return values


def ensembles(store, data, protocol, supplement, budget):
    for i in protocol["decoder_folds"]:
        target = DEST / f"rows/ensemble-fold-{i}.json"
        if target.exists():
            continue
        start = time.perf_counter()
        previous, condition, bases, priors = selected_source(i)
        validation_key = data["folds"][i]["validation"][0]
        view, _ = load_predictions(OLD / f"validation_predictions/fold-{i}/{condition}/clean.npz")
        context = ScoreContext(store.match(validation_key), view)
        sources = {seed: validation_seed(store, data, i, condition, seed, protocol) for seed in (2027, 2028, 2029)}
        trials, baseline = [], None
        for members in supplement["ensemble_members"]:
            budget.check()
            metrics = {}
            for name in protocol["primary_scenarios"]:
                p = mean_probabilities([sources[seed][name] for seed in members])
                metrics[name] = quick_scores(context, p, TemporalDecoder(bases["frozen"], priors), name == "clean")
            if baseline is None:
                baseline = metrics
            reasons = guard_reasons(metrics, baseline, protocol)
            trials.append({"members": members, "metrics": metrics,
                "objective": objective({k:v["spellF1"] for k,v in metrics.items()}, protocol),
                "eligible": not reasons, "rejection_reasons": reasons})
        winner = max((t for t in trials if t["eligible"]), key=lambda t:t["objective"])
        selection = {"outer_test_used":False,"condition":condition,"trials":trials,"winner":winner,
            "validation_s":time.perf_counter()-start,"source":source_hashes(["possession/night/reuse.py","configs/night_reuse_protocol.json"])}
        selection_path = DEST / f"reuse_selections/ensemble-fold-{i}.json"
        if selection_path.exists():
            frozen = json.loads(selection_path.read_text())
            if frozen['winner'] != winner or frozen['trials'] != trials:
                raise ValueError('Previously frozen ensemble decision differs')
        else:
            atomic_json(selection_path,selection)
        test_key = data["folds"][i]["test"][0]
        match = store.match(test_key)
        rows = []
        settings = previous["conditions"][condition]
        old_rows = json.loads((OLD / f"rows/fold-{i}.json").read_text())
        for case in json.loads((OLD / "protocol.json").read_text())["corruptions"]:
            budget.check()
            name = case["name"]
            loaded = {seed:load_predictions(OLD / f"test/seed-{seed}/fold-{i}/{condition}/{name}/probabilities.npz") for seed in (2027,2028,2029)}
            timeline = loaded[2027][0]
            for v,_ in loaded.values():np.testing.assert_array_equal(v.time,timeline.time)
            fraction = next(r["actual_hidden_fraction"] for r in old_rows if r["condition"]==condition and r["scenario"]==name)
            cache={}
            for family,members in (("ensemble_reference",[2027]),("ensemble_all3",[2027,2028,2029]),("ensemble_selected",winner["members"])):
                key=tuple(members)
                if key not in cache:
                    p=mean_probabilities([loaded[s][1] for s in members])
                    path=DEST/f"reuse_predictions/test/fold-{i}/ensemble-{'-'.join(map(str,members))}/{name}.npz"
                    save_probabilities(path,timeline,p)
                    if members==[2027]:
                        metrics=json.loads((OLD/f"test/seed-2027/fold-{i}/{condition}/{name}/{previous['winner']['family']}.json").read_text())
                    else:
                        metrics=score_match(match,timeline,p,TemporalDecoder(bases['frozen'],priors))
                    cache[key]=(metrics,str(path))
                metrics,path=cache[key]
                row=metric_row(metrics,i,condition,name,family,2027,settings['cap'],settings['selected_epoch'],fraction,family=='ensemble_selected')
                row.update(experiment='ensemble',members=members,probability_path=path,
                    validation_objective=winner['objective'] if family=='ensemble_selected' else next(t['objective'] for t in trials if t['members']==members))
                rows.append(row)
        atomic_json(target,rows)
        progress('ensemble_complete',fold=i)


def checkpoint_criterion(store,data,protocol,supplement,budget):
    for i in supplement['checkpoint_folds']:
        target=DEST/f"rows/checkpoint-fold-{i}.json"
        if target.exists():continue
        start=time.perf_counter()
        previous=json.loads((OLD/f"selections/fold-{i}.json").read_text())
        validation_key=data['folds'][i]['validation'][0]
        view=store.view(validation_key)
        context=ScoreContext(store.match(validation_key),view)
        trials=[]
        for condition in protocol['training_conditions']:
            folder=DEST/f"training/{condition}/fold-{i}"
            training=json.loads((folder/'training.json').read_text())
            old_epoch=json.loads((old_directory(i,condition)/'training.json').read_text())['chosen']['selected_epoch']
            epochs=sorted(set([old_epoch]+[int(p.stem.split('-')[-1]) for p in folder.glob('epoch-*.pt') if int(p.stem.split('-')[-1])>=12]))
            for epoch in epochs:
                budget.check()
                path=DEST/f"reuse_checkpoint_trials/fold-{i}/{condition}/epoch-{epoch}.json"
                if path.exists():trial=json.loads(path.read_text())
                else:
                    loss=training['history'][epoch-1]['validation']['loss']
                    torch_rng=torch.get_rng_state()
                    trial=cap_result(folder,i,condition,48,len(training['history']),epoch,loss,context,view,protocol)
                    torch.set_rng_state(torch_rng)
                    trial['condition']=condition
                    trial['eligible']=robust_eligible(trial['validation_scores'][trial['frozen_family']],previous['winner']['scores'],protocol)
                    atomic_json(path,trial)
                trials.append(trial)
        winner=max((t for t in trials if t['eligible']),key=lambda t:t['objective'])
        selection_path=DEST/f"reuse_selections/checkpoint-fold-{i}.json"
        selection={'outer_test_used':False,'trials':trials,'winner':winner,
            'validation_s':time.perf_counter()-start,'source':source_hashes(['possession/night/reuse.py','configs/night_reuse_protocol.json'])}
        if selection_path.exists():
            frozen=json.loads(selection_path.read_text())
            if frozen['winner']!=winner or frozen['trials']!=trials:raise ValueError('Previously frozen checkpoint choice differs')
        else:atomic_json(selection_path,selection)
        test_key=data['folds'][i]['test'][0]
        match,view=store.match(test_key),store.view(test_key)
        condition,epoch,family=winner['condition'],winner['selected_epoch'],winner['frozen_family']
        state=torch.load(winner['checkpoint'],map_location='cpu',weights_only=False)
        model=new_model(TrainConfig.from_dict(state['config']));model.load_state_dict(state['model'])
        with np.load(old_directory(i,condition)/'priors.npz') as z:priors=dict(z)
        specification=previous['conditions'][condition]['by_family'][family]['specification']
        rows=[]
        for case in json.loads((OLD/'protocol.json').read_text())['corruptions']:
            budget.check()
            name=case['name']
            path=DEST/f"reuse_predictions/checkpoint-test/fold-{i}/{condition}/epoch-{epoch}/{name}.npz"
            existing=DEST/f"training_test_predictions/fold-{i}/{condition}/epoch-{epoch}/{name}.npz"
            old_epoch=json.loads((old_directory(i,condition)/'training.json').read_text())['chosen']['selected_epoch']
            if epoch==old_epoch:
                source=OLD/f"test/seed-2027/fold-{i}/{condition}/{name}/probabilities.npz"
                _,p=load_predictions(source)
            elif existing.exists():_,p=load_predictions(existing)
            elif path.exists():_,p=load_predictions(path)
            else:
                corrupted,_=hide_ball(view,case['fraction'],case['block_seconds'],stable_seed(test_key,701))
                p=infer(model,corrupted,protocol)
            save_probabilities(path,view,p)
            metrics=score_match(match,view,p,TemporalDecoder(specification,priors))
            fraction=float(block_mask(view.run,case['fraction'],round(case['block_seconds']*10),stable_seed(test_key,701)).mean())
            row=metric_row(metrics,i,condition,name,family,2027,48,epoch,fraction,True)
            row.update(experiment='checkpoint_criterion',criterion='robust_spell_on_saved_states',probability_path=str(path),
                validation_objective=winner['objective'],epochs_executed=winner['epochs_executed'])
            rows.append(row)
            atomic_json(DEST/f"reuse_checkpoint_metrics/fold-{i}/{name}.json",metrics)
        atomic_json(target,rows)
        progress('checkpoint_criterion_complete',fold=i)


def run_reuse(protocol,budget):
    supplement=register_reuse()
    data,_=manifest('idsse-only')
    store=PublicStore(data['artifacts'])
    checkpoint_criterion(store,data,protocol,supplement,budget)
    ensembles(store,data,protocol,supplement,budget)
