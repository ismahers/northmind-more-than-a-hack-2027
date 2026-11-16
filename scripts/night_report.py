"""Reports and cross-artifact verification, exclusively public external study files."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from possession.external_data.schema import file_hash
from possession.learning.corruption import hide_ball, stable_seed
from possession.learning.data import Windows, collate
from possession.learning.metrics import SCORER_PATH, score_match
from possession.learning.study import ScoreContext
from possession.learning.temporal import TemporalDecoder
from possession.learning.training import (TrainConfig, new_model, seed_everything, epoch_loss,
    implementation_hashes, predict)
from possession.night.cache import PublicStore, predict_batched
from possession.night.common import ROOT, OLD, DEST, atomic_json, objective, source_hashes
from possession.night.decoders import NightDecoder
from possession.night.decoding_study import selected_source, load_predictions, guard_reasons
from possession.night.training import equal_state, old_directory
from possession.night.reuse import mean_probabilities
from scripts.phase3 import CachedRunDecoder, extra_diagnostics
from scripts.train_e0 import manifest


def read(path):
    return json.loads(Path(path).read_text())


def all_rows():
    return sorted([r for p in (DEST/'rows').glob('*.json') for r in read(p)],
        key=lambda r:(r['experiment'],r['fold'],r['condition'],r['scenario'],r['decoder'],r.get('epoch_cap',0)))


def csv_rows(path,rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def pipelines(rows):
    previous=[r for r in read(OLD/'all_rows.json') if r['seed']==2027 and r['validation_selected_pipeline']]
    return {
        'phase3':previous,
        'decoder_selected':[r for r in rows if r['experiment']=='cached_decoder' and r['validation_selected_pipeline']],
        'ensemble_all3':[r for r in rows if r['experiment']=='ensemble' and r['decoder']=='ensemble_all3'],
        'ensemble_selected':[r for r in rows if r['experiment']=='ensemble' and r['validation_selected_pipeline']],
        'both_fixed':[r for r in rows if r['experiment']=='combination' and r['decoder']=='combination_both'],
        'combined_selected':[r for r in rows if r['experiment']=='combination' and r['validation_selected_pipeline']],
        'training_selected':[r for r in rows if r['experiment']=='training_duration' and r['validation_selected_pipeline']],
        'checkpoint_selected':[r for r in rows if r['experiment']=='checkpoint_criterion'],
        'integrated_selected':[r for r in rows if r['experiment']=='integrated_selection'],
    }


def summarize(values,reference):
    result={'folds':len(values),'fold_ids':[r['fold'] for r in values]}
    for field in ('spellF1','macroF1','alive_dead_F1','boundary_precision_1s','boundary_recall_1s','boundary_MAE_s'):
        numbers=[r[field] for r in values if r.get(field) is not None]
        result[field+'_mean']=float(np.mean(numbers)) if numbers else None
        result[field+'_sample_SD']=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None
    for field in ('gt_spells','pred_spells','fragmented_gt','fused_predictions','false_D_interruptions','false_team_changes'):
        result[field+'_total']=sum(r[field] for r in values)
        result[field+'_reference_total']=sum(reference[r['fold']][field] for r in values)
    differences=np.array([r['spellF1']-reference[r['fold']]['spellF1'] for r in values])
    result['paired_delta_spellF1']=float(differences.mean())
    if len(values)>=3:
        boot=np.random.default_rng(2027).choice(differences,(5000,len(values)),replace=True).mean(-1)
        result['paired_match_95CI']=np.percentile(boot,[2.5,97.5]).tolist()
    else:result['paired_match_95CI']=None
    return result


def learning_curves():
    curves=[]
    for path in sorted((DEST/'training').glob('*/fold-*/training.json')):
        condition=path.parent.parent.name
        fold=int(path.parent.name.split('-')[-1])
        for e in read(path)['history']:
            execution='reused_phase3' if e['epoch']<=6 else 'verification_replay' if e['epoch']<=12 else 'new_epoch'
            curves.append({'fold':fold,'condition':condition,'epoch':e['epoch'],'train_loss':e['train']['loss'],
                'validation_loss':e['validation']['loss'],'execution':execution,'elapsed_s':e.get('elapsed_s'),
                'gradient_norm_mean':e['train'].get('gradient_norm_mean'),'gradient_norm_max':e['train'].get('gradient_norm_max')})
    csv_rows(DEST/'learning_curves.csv',curves)
    if curves:
        import os
        os.environ.setdefault('MPLCONFIGDIR',str(DEST/'mpl_cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for ax,(fold,condition) in zip(axes.flat,((0,'A_clean'),(0,'A_blocks'),(6,'A_clean'),(6,'A_blocks'))):
            values=[r for r in curves if r['fold']==fold and r['condition']==condition]
            if not values:continue
            ax.plot([r['epoch'] for r in values],[r['train_loss'] for r in values],label='Train')
            ax.plot([r['epoch'] for r in values],[r['validation_loss'] for r in values],label='Inner validation')
            ax.axvline(12,color='gray',linestyle='--',linewidth=1)
            ax.set_title(f'Fold {fold} · {condition}')
            ax.set_xlabel('Epoch');ax.set_ylabel('Masked BCE (state + team)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('E0: reused prefix, exact replay, and continuation with early stopping')
        fig.tight_layout()
        fig.savefig(DEST/'learning_curves.png',dpi=160)
        fig.savefig(DEST/'learning_curves.svg')
        plt.close(fig)
    return curves


def table(lines,headers,rows):
    lines.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    lines.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
    lines.append('')


def f(value):
    return '—' if value is None else f'{value:.4f}'


def report(protocol):
    rows=all_rows()
    atomic_json(DEST/'all_rows.json',rows)
    csv_rows(DEST/'all_rows.csv',rows)
    for i in range(7):csv_rows(DEST/f'per_fold/fold-{i}.csv',[r for r in rows if r['fold']==i])
    groups=pipelines(rows)
    summaries=[]
    for name,values in groups.items():
        for scenario in [c['name'] for c in read(OLD/'protocol.json')['corruptions']]:
            selected=[r for r in values if r['scenario']==scenario]
            if not selected:continue
            reference={r['fold']:r for r in groups['phase3'] if r['scenario']==scenario}
            summaries.append({'pipeline':name,'scenario':scenario,**summarize(selected,reference)})
    curves=learning_curves()
    training=[]
    for path in sorted((DEST/'training').glob('*/fold-*/training.json')):
        result=read(path)
        i=int(path.parent.name.split('-')[-1]);condition=path.parent.parent.name
        for cap in result['caps']:
            measured={r['scenario']:r for r in rows if r['experiment']=='training_duration' and r['fold']==i and r['condition']==condition
                and r['epoch_cap']==cap['cap'] and r['decoder']==cap['frozen_family']}
            training.append({'fold':i,'condition':condition,'cap':cap['cap'],'epoch':cap['selected_epoch'],
                'epochs_executed':cap['epochs_executed'],'validation_objective':cap['objective'],
                'frozen_decoder':cap['frozen_family'],'metrics':measured,'stopped_early':result['stopped_early'],
                'new_training_s':result['new_epochs_s_this_session'],'replay_s':result['replay_s_this_session'],
                'trajectory_total_s':result['total_s_this_session']})
    decoder_search=[]
    reasons=Counter()
    for path in sorted((DEST/'decoder_selections').glob('fold-*.json')):
        result=read(path)
        for trial in result['trials']:
            decoder_search.append({'fold':int(path.stem.split('-')[-1]),**trial})
            reasons.update(trial['rejection_reasons'])
    clean_selected={r['fold']:r for r in groups['integrated_selected'] if r['scenario']=='clean'}
    fold_table=[]
    for i in range(7):
        value={'fold':i}
        for name,values in groups.items():
            match=next((r for r in values if r['fold']==i and r['scenario']=='clean'),None)
            if match:value[name+'_spellF1']=match['spellF1']
        if i in clean_selected:value.update({k:clean_selected[i][k] for k in ('macroF1','fragmented_gt','fused_predictions','false_D_interruptions','gt_spells','pred_spells','branch')})
        fold_table.append(value)
    diagnostics=[]
    duration_errors=[]
    duration_groups=defaultdict(lambda:Counter())
    for path in sorted((DEST/'final_diagnostics').glob('fold-*.json')):
        value=read(path)
        i=int(path.stem.split('-')[-1])
        diagnostics.append({'fold':i,**value['extra']})
        previous,condition,_,_=selected_source(i)
        reference=read(OLD/f"test/seed-2027/fold-{i}/{condition}/clean/{previous['winner']['family']}.json")
        before=reference['per_possession'];after=value['metrics']['per_possession']
        assert len(before)==len(after)
        for a,b in zip(before,after):
            assert (a['start_canonical_frame'],a['stop_canonical_frame'],a['team'])==(b['start_canonical_frame'],b['stop_canonical_frame'],b['team'])
            group='<3s' if b['duration_s']<3 else '3-10s' if b['duration_s']<10 else '10-30s' if b['duration_s']<30 else '>=30s'
            counts=duration_groups[group];counts['gt_spells']+=1
            counts['phase3_matched_iou_0_5']+=int(a['matched_at_iou_0_5'])
            counts['final_matched_iou_0_5']+=int(b['matched_at_iou_0_5'])
            counts['recovered']+=int(not a['matched_at_iou_0_5'] and b['matched_at_iou_0_5'])
            counts['lost']+=int(a['matched_at_iou_0_5'] and not b['matched_at_iou_0_5'])
    for group,counts in duration_groups.items():
        duration_errors.append({'duration':group,**counts,
            'phase3_recall_iou_0_5':counts['phase3_matched_iou_0_5']/counts['gt_spells'],
            'final_recall_iou_0_5':counts['final_matched_iou_0_5']/counts['gt_spells']})
    state={'source':'real public IDSSE/Sportec and existing public-data E0 checkpoints; no private target GT',
        'rows':len(rows),'pipeline_summaries':summaries,'training':training,'fold_table':fold_table,
        'decoder_trials':len(decoder_search),'decoder_rejection_reasons':dict(reasons),
        'compute':read(DEST/'compute.json') if (DEST/'compute.json').exists() else None,
        'profile':read(DEST/'profile.json') if (DEST/'profile.json').exists() else None,
        'final_diagnostics':diagnostics,'limit_seconds':protocol['compute_budget_seconds'],
        'per_possession_duration_diagnostics':duration_errors,
        'uniform_comparisons':'phase3 / decoder / ensemble / combined use frozen original neural checkpoints on all seven folds',
        'two_fold_limit':'extension and checkpoint-criterion experiments only on folds0/6; integrated selection is heterogeneous and reported separately',
        'checkpoint_execution':'caps24/48 alias the same result when patience stops earlier; not 48 executed epochs',
        'research_limit':'same seven-match cohort already inspected in prior phases; exploratory, not a fresh untouched confirmation set'}
    atomic_json(DEST/'summary.json',state)
    atomic_json(DEST/'fold_table.json',fold_table)
    csv_rows(DEST/'fold_table.csv',fold_table)
    lines=['# Experimentos nocturnos — resultados reales','',
        f'Filas de evaluación: {len(rows)}. Las filas reutilizadas y los presupuestos equivalentes están identificados; no son experimentos independientes.','',
        '## Protocolo e integridad','',
        'Dos folds prefijados para prolongar entrenamiento: 0 y 6, extremos del orden canónico y de dos competiciones. '
        'Cada fold conserva cinco IDSSE train, un partido de validación interna y uno test. '
        'CPU, cuatro hilos y una única ejecución de cómputo a la vez. '
        'El evaluador oficial y los archivos de Fase 3 no se modifican.','',
        'La selección usa 50% spellF1 limpio y 25% para cada oclusión 40%/5 s y 40%/15 s. '
        'El suelo de robustez permite como máximo −0,02 absoluto por escenario oculto respecto a la referencia. '
        'Los decoders/ensembles/combinaciones además exigen no aumentar fusiones de validación y limitan '
        'el aumento de fragmentaciones a max(2, ceil(5% del original)). Estas garantías son de validación; '
        'los errores test se informan aunque empeoren. La ablación de entrenamiento/checkpoint solo usa el suelo de robustez.','',
        'Las elecciones se guardan desde sus validaciones antes de leer sus filas test. '
        'No se elige el mayor promedio test como configuración. Las tres semillas ya existían; '
        'se combinan sus probabilidades, no sus puntuaciones.','',
        '## Tiempo y reutilización','']
    if state['compute']:
        table(lines,['Etapa','Estado','Tiempo activo s'],[[r['stage'],r['status'],f(r['elapsed_s'])] for r in state['compute']['sessions']])
        lines.append(f"Tiempo activo instrumentado acumulado: {state['compute']['seconds']:.1f} s; límite {protocol['compute_budget_seconds']/3600:.0f} h. No incluye redacción/inspección del agente ni entrenamiento anterior.\n")
    if state['profile']:
        q=state['profile'];lines.append(f"Medición inicial de inferencia individual/lotes: {q['single_window_inference_s']:.3f} / {q['batched_inference_s']:.3f} s. "
            f"Máxima diferencia de probabilidades {q['maximum_probability_difference']:.3g}; paths argmax/HMM/HSMM idénticos en la validación comprobada. "
            'Es una medición inicial; `verification.json` incluye un benchmark alternado adicional.\n')
    lines.extend(['Ahorros principales: no repetir las siete LOMO completas, no volver a cargar veinte SkillCorner por condición, '
        'features mmap reutilizadas, emisiones previas para decoders/ensembles y deduplicación de checkpoints/predicciones idénticos. '
        'La inferencia por lotes no cambia entrenamiento, pérdidas ni contexto.','',
        '## Entrenamiento 12/24/48 con decoders congelados',''])
    table(lines,['Fold','Condición','Cap','Época elegida','Ejecutadas','Val objetivo','Test limpio','40%/5s','40%/15s'],
        [[r['fold'],r['condition'],r['cap'],r['epoch'],r['epochs_executed'],f(r['validation_objective']),
          *[f(r['metrics'].get(s,{}).get('spellF1')) for s in protocol['primary_scenarios']]] for r in training])
    lines.extend(['El checkpoint se elige por pérdida dentro de cada cap; luego se elige cap/condición por spellF1 interno robusto. '
        'La reanudación reconstruye el generador del DataLoader, comprueba ese RNG contra un checkpoint original completo '
        'y reproduce cap6→cap12 contrastando pesos, momentos AdamW y RNG de Torch bit a bit. '
        'Las nuevas épocas guardan todos los estados para continuar exactamente.','',
        '![Curvas de aprendizaje](learning_curves.png)','',
        'Las curvas completas y los tiempos por época están en `learning_curves.csv`; las épocas ≤6 se reutilizan, '
        '7–12 se reproducen para verificar compatibilidad y las posteriores son entrenamiento nuevo.','',
        '## Comparación por fold — spellF1 limpio',''])
    table(lines,['Fold','Fase3','Decoder','Ensemble','Combinación elegida','Checkpoint spell (2 folds)','Integrado*'],
        [[r['fold'],*[f(r.get(k+'_spellF1')) for k in ('phase3','decoder_selected','ensemble_selected','combined_selected','checkpoint_selected','integrated_selected')]] for r in fold_table])
    lines.extend(['* Integrado incluye alternativas de entrenamiento solo en los folds 0/6. '
        'La comparación uniforme de siete folds con pesos originales es `combined_selected`.','',
        '## Métricas y variabilidad de los pipelines',''])
    table(lines,['Pipeline','Escenario','Folds','spellF1 ± SD','macroF1 ± SD','Δ emparejado vs Fase3','IC95% exploratorio'],
        [[r['pipeline'],r['scenario'],r['folds'],f(r['spellF1_mean'])+' ± '+f(r['spellF1_sample_SD']),
          f(r['macroF1_mean'])+' ± '+f(r['macroF1_sample_SD']),f(r['paired_delta_spellF1']),
          '—' if r['paired_match_95CI'] is None else ' / '.join(f(x) for x in r['paired_match_95CI'])] for r in summaries])
    lines.extend(['Los IC remuestrean partidos completos, no frames. Dos folds no permiten una inferencia robusta; '
        'los equipos repetidos y las búsquedas sucesivas sobre esta pequeña cohorte limitan todas las conclusiones. '
        'No se ha medido Barcelona–Athletic ni se ha usado GT privado.','',
        '## Fusiones, fragmentaciones y D falsas — limpio',''])
    table(lines,['Pipeline','Folds','Reales/pred.','Fragmentadas','Fusiones','D internas falsas','Cambios H/A incorrectos'],
        [[r['pipeline'],r['folds'],f"{r['gt_spells_total']} / {r['pred_spells_total']}",r['fragmented_gt_total'],r['fused_predictions_total'],
          r['false_D_interruptions_total'],r['false_team_changes_total']] for r in summaries if r['scenario']=='clean'])
    lines.extend(['Los solapes múltiples del mismo equipo describen errores de segmentación; no son categorías causales disjuntas. '
        'Cada agregado se compara con los mismos folds de referencia; el JSON guarda esos totales emparejados.','',
        '## Selección exclusivamente interna y descartes',''])
    choices=[]
    for path in sorted((DEST/'combination_selections').glob('fold-*.json')):
        i=int(path.stem.split('-')[-1]);winner=read(path)['winner']
        core=read(DEST/f'decoder_selections/fold-{i}.json')['winner']['specification']
        integrated=read(DEST/f'integrated_selections/fold-{i}.json')['winner']
        choices.append([i,winner['variant'],','.join(map(str,winner['members'])),json.dumps(core,sort_keys=True),f(winner['objective']),integrated['branch']])
    table(lines,['Fold','Variante combinada','Semillas','Decoder fijado en búsqueda previa','Val objetivo','Rama integrada'],choices)
    families=defaultdict(lambda:Counter())
    for t in decoder_search:
        family=t['specification']['family'];families[family]['trials']+=1
        families[family]['eligible']+=int(t['eligible'])
    table(lines,['Familia','Candidatos internos','Admisibles'],[[k,v['trials'],v['eligible']] for k,v in sorted(families.items())])
    lines.append('Motivos de rechazo (pueden coexistir): '+json.dumps(dict(reasons),sort_keys=True)+'.\n')
    lines.extend(['Se probaron penalizaciones direccionales H/A, evidencia local derivada de probabilidades, '
        'escalado de hazards HMM/HSMM con D conservado, refinamiento de fronteras sin cambiar el orden de segmentos '
        'e histéresis de equipo con máscara D congelada. La evidencia local no es una etiqueta ni una cabeza de frontera aprendida. '
        'Las familias sin candidato admisible no se evalúan en test como finalistas.','',
        'Los caps 48 no fuerzan más entrenamiento después de early stopping. '
        'No se añadieron ASFormer, PathCRF, visión artificial ni nuevas búsquedas de arquitectura.','',
        '## Diagnósticos adicionales del pipeline integrado',''])
    if diagnostics:
        table(lines,['Fold','F1 cabeza vivo','Exactitud equipo GT vivo','Oráculo D','Oráculo equipo','MAE inicio / final s'],
            [[r['fold'],f(r['head_alive_F1_at_0_5']),f(r['conditional_team_accuracy_gt_alive']),
              f(r['oracles_analysis_only']['true_D_predicted_conditional_team']['spell_f1_mean']),
              f(r['oracles_analysis_only']['true_team_where_defined_predicted_D']['spell_f1_mean']),
              f(r['matched_start_MAE_s'])+' / '+f(r['matched_end_MAE_s'])] for r in diagnostics])
        lines.append('Oráculos solo para diagnóstico. Durante D real no hay equipo GT; el oráculo de equipo conserva esos falsos vivos. '
            'El oráculo de D utiliza equipo condicional sin suavizar. MAE de extremos solo en posesiones emparejadas a IoU ≥0,5.\n')
        table(lines,['Duración GT','Posesiones','Recall IoU0,5 Fase3','Recall IoU0,5 final','Recuperadas','Perdidas'],
            [[r['duration'],r['gt_spells'],f(r['phase3_recall_iou_0_5']),f(r['final_recall_iou_0_5']),r['recovered'],r['lost']] for r in duration_errors])
        lines.append('Estos estratos muestran recall de posesiones a IoU0,5, no un spellF1 artificial sobre secuencias recortadas.\n')
    lines.extend(['## Artefactos y reproducción','',
        '- `all_rows.csv` / `.json` y `per_fold/`: todas las configuraciones realmente evaluadas y los cinco IoU oficiales.',
        '- `summary.json`, `fold_table.csv`, `learning_curves.csv`, `learning_curves.png` / `.svg`.',
        '- `*_selections/`: candidatos, objetivos internos, guardas y decisiones anteriores a sus filas test.',
        '- `training/`: checkpoints completos por época; `decoder_trials/`: búsqueda reanudable.',
        '- `verification.json`: integridad, replay de una época y comprobación del pipeline final.',
        '- `docs/NIGHT_EXPERIMENTS.md` en el repositorio: comandos y límites.',
        '', 'Las conclusiones interpretativas y el siguiente experimento están en `conclusions.md` cuando se complete la revisión.',''])
    (DEST/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def restore_and_replay_one_epoch(data,protocol):
    store=PublicStore(data['artifacts']);i=6;condition='A_blocks'
    folder=DEST/f'training/{condition}/fold-{i}'
    before=torch.load(folder/'epoch-014.pt',map_location='cpu',weights_only=False)
    expected=torch.load(folder/'epoch-015.pt',map_location='cpu',weights_only=False)
    config=TrainConfig.from_dict(before['config']);seed_everything(config.seed)
    model=new_model(config);model.load_state_dict(before['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    optimizer.load_state_dict(before['optimizer']);scaler=torch.amp.GradScaler('cuda',enabled=False)
    generator=torch.Generator();generator.set_state(before['loader_rng'])
    views={k:store.view(k) for k in data['folds'][i]['train']}
    valkey=data['folds'][i]['validation'][0];validation=store.view(valkey)
    epoch=15
    corrupted={key:hide_ball(v,.4,(1.,5.,15.)[(epoch-1)%3],stable_seed(key,config.seed+1000*epoch))[0] for key,v in views.items()}
    loader=DataLoader(Windows(corrupted,config.core_frames,model.context),batch_size=config.batch_size,shuffle=True,generator=generator,collate_fn=collate)
    val_loader=DataLoader(Windows({valkey:validation},config.core_frames,model.context),batch_size=config.batch_size,collate_fn=collate)
    torch.set_rng_state(before['torch_rng']);np.random.set_state(before['numpy_rng'])
    import random
    random.setstate(before['python_rng'])
    epoch_loss(model,loader,torch.device('cpu'),optimizer,scaler)
    epoch_loss(model,val_loader,torch.device('cpu'))
    result={'model':equal_state(model.state_dict(),expected['model']),
        'optimizer':equal_state(optimizer.state_dict(),expected['optimizer']),
        'torch_rng':torch.equal(torch.get_rng_state(),expected['torch_rng']),
        'loader_rng':torch.equal(generator.get_state(),expected['loader_rng'])}
    assert all(result.values())
    return result


def final_metrics(data,protocol):
    store=PublicStore(data['artifacts']);results=[]
    for i in protocol['decoder_folds']:
        previous,condition,bases,priors=selected_source(i)
        branch=read(DEST/f'integrated_selections/fold-{i}.json')['winner']['branch']
        if branch=='combined':
            winner=read(DEST/f'combination_selections/fold-{i}.json')['winner']
            loaded={s:load_predictions(OLD/f'test/seed-{s}/fold-{i}/{condition}/clean/probabilities.npz') for s in winner['members']}
            view=next(iter(loaded.values()))[0]
            p=mean_probabilities([v[1] for v in loaded.values()])
            decoder=NightDecoder(winner['specification'],bases,priors)
        else:
            winner=read(DEST/f'reuse_selections/checkpoint-fold-{i}.json')['winner']
            condition=winner['condition'];epoch=winner['selected_epoch'];family=winner['frozen_family']
            view,p=load_predictions(DEST/f'reuse_predictions/checkpoint-test/fold-{i}/{condition}/epoch-{epoch}/clean.npz')
            with np.load(old_directory(i,condition)/'priors.npz') as archive:priors=dict(archive)
            decoder=TemporalDecoder(previous['conditions'][condition]['by_family'][family]['specification'],priors)
        decoder=CachedRunDecoder(decoder)
        match=store.match(data['folds'][i]['test'][0])
        metrics=score_match(match,view,p,decoder)
        recorded=next(r for r in read(DEST/f'rows/integrated-fold-{i}.json') if r['scenario']=='clean')
        assert metrics['spell_f1_mean']==recorded['spellF1'] and metrics['macro_f1']==recorded['macroF1']
        extra=extra_diagnostics(match,view,p,decoder)
        atomic_json(DEST/f'final_diagnostics/fold-{i}.json',{'metrics':metrics,'extra':extra,'branch':branch})
        results.append({'fold':i,'spellF1':metrics['spell_f1_mean'],'macroF1':metrics['macro_f1']})
    return results


def inference_benchmark(data,protocol):
    path=DEST/'inference_benchmark.json'
    if path.exists():return read(path)
    store=PublicStore(data['artifacts']);i=0
    view=store.view(data['folds'][i]['validation'][0])
    state=torch.load(old_directory(i,'A_clean')/'cap-12.pt',map_location='cpu',weights_only=False)
    model=new_model(TrainConfig.from_dict(state['config']));model.load_state_dict(state['model'])
    predict_batched(model,view,batch_size=protocol['inference_batch'])
    timings={'individual':[],'batched':[]}
    for repeat in range(5):
        for name in (('individual','batched') if repeat%2==0 else ('batched','individual')):
            start=time.perf_counter()
            if name=='individual':p=predict(model,view,256,torch.device('cpu'))
            else:p=predict_batched(model,view,batch_size=protocol['inference_batch'])
            timings[name].append(time.perf_counter()-start)
    result={'timings_s':timings,'median_individual_s':float(np.median(timings['individual'])),
        'median_batched_s':float(np.median(timings['batched'])),'repeats':5,'warmup':True,'alternating_order':True}
    result['median_speedup']=result['median_individual_s']/result['median_batched_s']
    atomic_json(path,result)
    return result


def verify(protocol):
    data,_=manifest('idsse-only')
    rows=all_rows();assert rows
    for i in protocol['training_folds']:
        for condition in protocol['training_conditions']:
            path=DEST/f'training/{condition}/fold-{i}'
            result=read(path/'training.json')
            assert result['prefix_verified']
            assert result['fingerprint']['old_fingerprint']['implementation']==implementation_hashes()
            assert result['fingerprint']['source']==source_hashes(result['fingerprint']['source'].keys())
            assert [c['cap'] for c in result['caps']]==protocol['caps']
            verification=read(path/'prefix_verification.json')
            assert all(verification[k] for k in ('model','optimizer','torch_rng','loader_rng_verified_against_complete_baseline'))
    for i in protocol['decoder_folds']:
        selection=read(DEST/f'decoder_selections/fold-{i}.json')
        assert selection['outer_test_used'] is False
        assert selection['winner']['objective']==max(t['objective'] for t in selection['trials'] if t['eligible'])
        for t in selection['trials']:
            assert t['objective']==objective({k:v['spellF1'] for k,v in t['metrics'].items()},protocol)
            assert t['rejection_reasons']==guard_reasons(t['metrics'],selection['trials'][0]['metrics'],protocol)
        for category in ('ensemble','checkpoint'):
            path=DEST/f'reuse_selections/{category}-fold-{i}.json'
            if not path.exists():continue
            choice=read(path);assert choice['outer_test_used'] is False
            assert choice['winner']['objective']==max(t['objective'] for t in choice['trials'] if t['eligible'])
        choice=read(DEST/f'combination_selections/fold-{i}.json');assert choice['outer_test_used'] is False
        assert choice['winner']['objective']==max(t['objective'] for t in choice['trials'] if t['eligible'])
        integrated=read(DEST/f'integrated_selections/fold-{i}.json')
        assert integrated['outer_test_used'] is False
        assert integrated['winner']['objective']==max(v['objective'] for v in integrated['candidates'])
        for experiment in ('cached_decoder','ensemble','combination','integrated_selection'):
            selected=[r for r in rows if r['fold']==i and r['experiment']==experiment and r['validation_selected_pipeline']]
            assert len(selected)==7 and len({r['scenario'] for r in selected})==7
    for path in (DEST/'reuse_predictions').rglob('*.npz'):
        with np.load(path) as archive:
            p=archive['probabilities'];assert np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
    replay=restore_and_replay_one_epoch(data,protocol)
    final=final_metrics(data,protocol)
    benchmark=inference_benchmark(data,protocol)
    reference=read(OLD/'verification.json')
    assert file_hash(SCORER_PATH)==reference['scorer_sha256']
    # No prior artifact should have been written since registering this isolated experiment.
    registration=(DEST/'protocol.json').stat().st_mtime
    changed=[str(p.relative_to(OLD)) for p in OLD.rglob('*') if p.is_file() and p.stat().st_mtime>registration]
    assert not changed
    state={'status':'OK','rows':len(rows),'exact_new_checkpoint_resume':replay,'final_metrics_recomputed':final,
        'official_scorer_sha256':file_hash(SCORER_PATH),'phase3_files_modified_since_registration':changed,
        'inference_benchmark':benchmark,
        'source':source_hashes([str(p.relative_to(ROOT)) for p in (ROOT/'possession/night').glob('*.py')]+
            ['scripts/night_experiments.py','scripts/night_report.py','configs/night_protocol.json',
             'configs/night_reuse_protocol.json','configs/night_combination_protocol.json']),
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'device':'cpu'}}
    atomic_json(DEST/'verification.json',state)
