"""A fixed 2x2 interaction test: ensemble and decoder, with no new parameter search."""

import json
import time

import numpy as np

from possession.learning.metrics import score_match
from possession.learning.study import ScoreContext
from scripts.phase3 import CachedRunDecoder, metric_row
from scripts.train_e0 import manifest
from .cache import PublicStore
from .common import ROOT, DEST, OLD, atomic_json, objective, source_hashes, progress
from .decoders import NightDecoder
from .decoding_study import selected_source, load_predictions, quick_scores, guard_reasons
from .reuse import validation_seed, mean_probabilities


def run_combination(protocol,budget):
    configuration=json.loads((ROOT/'configs/night_combination_protocol.json').read_text())
    frozen=DEST/'combination_protocol.json'
    provenance={'configuration':configuration,'source':source_hashes(['possession/night/combination.py','possession/night/decoders.py'])}
    if frozen.exists() and json.loads(frozen.read_text())!=provenance:raise ValueError('Combination protocol/source changed')
    if not frozen.exists():atomic_json(frozen,provenance)
    data,_=manifest('idsse-only')
    store=PublicStore(data['artifacts'])
    for i in protocol['decoder_folds']:
        target=DEST/f'rows/combination-fold-{i}.json'
        if target.exists():continue
        started=time.perf_counter()
        previous,condition,bases,priors=selected_source(i)
        decoder_selection=json.loads((DEST/f'decoder_selections/fold-{i}.json').read_text())
        ensemble_selection=json.loads((DEST/f'reuse_selections/ensemble-fold-{i}.json').read_text())
        adapted=decoder_selection['winner']['specification']
        members=ensemble_selection['winner']['members']
        variants={'original':([2027],{'family':'frozen'}),'decoder':([2027],adapted),
                  'ensemble':(members,{'family':'frozen'}),'both':(members,adapted)}
        seeds=sorted(set([2027]+members))
        sources={s:validation_seed(store,data,i,condition,s,protocol) for s in seeds}
        key=data['folds'][i]['validation'][0]
        timeline,_=load_predictions(OLD/f'validation_predictions/fold-{i}/{condition}/clean.npz')
        context=ScoreContext(store.match(key),timeline)
        trials,baseline=[],None
        for name,(seedset,specification) in variants.items():
            budget.check()
            metrics={case:quick_scores(context,mean_probabilities([sources[s][case] for s in seedset]),
                NightDecoder(specification,bases,priors),case=='clean') for case in protocol['primary_scenarios']}
            if baseline is None:baseline=metrics
            reasons=guard_reasons(metrics,baseline,protocol)
            trials.append({'variant':name,'members':seedset,'specification':specification,'metrics':metrics,
                'objective':objective({k:v['spellF1'] for k,v in metrics.items()},protocol),
                'eligible':not reasons,'rejection_reasons':reasons})
        winner=max((t for t in trials if t['eligible']),key=lambda t:t['objective'])
        selection={'outer_test_used':False,'trials':trials,'winner':winner,'validation_s':time.perf_counter()-started}
        selection_path=DEST/f'combination_selections/fold-{i}.json'
        if selection_path.exists():
            old=json.loads(selection_path.read_text())
            if old['winner']!=winner or old['trials']!=trials:raise ValueError('Combination choice changed')
        else:atomic_json(selection_path,selection)
        # Only validation files have been read to choose the final branch.
        branches=[{'branch':'combined','objective':winner['objective']}]
        if i in protocol['training_folds']:
            checkpoint=json.loads((DEST/f'reuse_selections/checkpoint-fold-{i}.json').read_text())['winner']
            branches.append({'branch':'checkpoint_criterion','objective':checkpoint['objective']})
        integrated={'outer_test_used':False,'candidates':branches,'winner':max(branches,key=lambda b:b['objective']),
            'warning':'extra training/checkpoint alternatives exist only on the two predeclared folds; not uniform seven-fold training'}
        integrated_path=DEST/f'integrated_selections/fold-{i}.json'
        if integrated_path.exists() and json.loads(integrated_path.read_text())!=integrated:raise ValueError('Integrated selection changed')
        if not integrated_path.exists():atomic_json(integrated_path,integrated)
        # New held-out evaluation starts here. Previous rows are never consulted for selection.
        match=store.match(data['folds'][i]['test'][0])
        settings=previous['conditions'][condition]
        previous_rows=json.loads((OLD/f'rows/fold-{i}.json').read_text())
        rows=[]
        for case in json.loads((OLD/'protocol.json').read_text())['corruptions']:
            budget.check()
            name=case['name']
            test_sources={s:load_predictions(OLD/f'test/seed-{s}/fold-{i}/{condition}/{name}/probabilities.npz') for s in seeds}
            timeline=test_sources[2027][0]
            fraction=next(r['actual_hidden_fraction'] for r in previous_rows if r['condition']==condition and r['scenario']==name)
            metrics_cache={}
            for label,variant in (('combination_both','both'),('combined_selected',winner['variant'])):
                seedset,specification=variants[variant]
                cache_key=json.dumps([seedset,specification],sort_keys=True)
                if cache_key not in metrics_cache:
                    p=mean_probabilities([test_sources[s][1] for s in seedset])
                    metrics=score_match(match,timeline,p,CachedRunDecoder(NightDecoder(specification,bases,priors)))
                    metrics_cache[cache_key]=metrics
                    atomic_json(DEST/f'combination_metrics/fold-{i}/{name}/{variant}.json',metrics)
                metrics=metrics_cache[cache_key]
                row=metric_row(metrics,i,condition,name,label,2027,settings['cap'],settings['selected_epoch'],fraction,label=='combined_selected')
                row.update(experiment='combination',variant=variant,members=seedset,decoder_specification=specification,
                    validation_objective=next(t['objective'] for t in trials if t['variant']==variant))
                rows.append(row)
        atomic_json(target,rows)
        progress('combination_complete',fold=i)
    # Assemble already scored rows only after independent branch choices have been frozen.
    for i in protocol['decoder_folds']:
        chosen=json.loads((DEST/f'integrated_selections/fold-{i}.json').read_text())['winner']['branch']
        source=DEST/f'rows/{"combination" if chosen=="combined" else "checkpoint"}-fold-{i}.json'
        rows=[r for r in json.loads(source.read_text()) if r['validation_selected_pipeline']]
        output=[]
        for r in rows:
            output.append({**r,'experiment':'integrated_selection','source_experiment':r['experiment'],
                'branch':chosen,'decoder':'integrated_selected'})
        atomic_json(DEST/f'rows/integrated-fold-{i}.json',output)
