"""Summarize only public-external phase5 artifacts; no model selection here."""

import csv
import json
from pathlib import Path

import numpy as np

from possession.night.common import atomic_json
from possession.night.common import OLD
from possession.phase5.experiment import DEST


def aggregate(rows):
    result = {"folds": len(rows), "spellF1": float(np.mean([r["spellF1"] for r in rows])),
              "spellF1_sd": float(np.std([r["spellF1"] for r in rows], ddof=1)),
              "macroF1": float(np.mean([r["macroF1"] for r in rows]))}
    for key in ("gt_spells", "pred_spells", "fragmented_gt", "fused_predictions", "false_D_interruptions", "false_team_changes"):
        result[key] = sum(r[key] for r in rows)
    result["duration_groups"] = {}
    for group in rows[0]["duration_groups"]:
        values = [r["duration_groups"][group] for r in rows]
        item = {key: sum(v[key] for v in values) for key in ("gt", "pred", "fragmented_gt", "fused_predictions")}
        item["iou"] = {}
        for tau in values[0]["iou"]:
            counts = {key: sum(v["iou"][tau][key] for v in values) for key in ("matched_gt", "matched_pred", "fp", "fn")}
            counts.update(recall=counts["matched_gt"] / max(1, item["gt"]),
                          precision=counts["matched_pred"] / max(1, item["pred"]))
            item["iou"][tau] = counts
        result["duration_groups"][group] = item
    for kind in ("head_HA", "decoded_HA"):
        result[kind] = {}
        for tolerance in rows[0][kind]:
            counts = {key: sum(r[kind][tolerance][key] for r in rows) for key in ("tp", "fp", "fn")}
            p, r = counts["tp"] / max(1, counts["tp"] + counts["fp"]), counts["tp"] / max(1, counts["tp"] + counts["fn"])
            counts.update(precision=p, recall=r, f1=2*p*r/max(1e-12,p+r))
            result[kind][tolerance] = counts
    for key in ("head_alive_F1_at_0_5", "conditional_team_accuracy_gt_alive", "matched_start_MAE_s", "matched_end_MAE_s"):
        values = [r["independent"][key] for r in rows if r["independent"][key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired(rows, left, right, scenario="clean"):
    a = sorted((r for r in rows if r["system"]==left and r["scenario"]==scenario), key=lambda r:r["fold"])
    b = sorted((r for r in rows if r["system"]==right and r["scenario"]==scenario), key=lambda r:r["fold"])
    delta = np.array([x["spellF1"]-y["spellF1"] for x,y in zip(a,b)])
    rng=np.random.default_rng(2027)
    means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(-1)
    return {"delta_mean":float(delta.mean()),"paired_fold_bootstrap_95":np.quantile(means,[.025,.975]).tolist(),
            "fold_deltas":delta.tolist(),"note":"exploratory; seven reused matches, not a new untouched confirmation set"}


def main():
    rows=json.loads((DEST/"all_rows.json").read_text())
    selections=[json.loads((DEST/f"selections/fold-{i}.json").read_text()) for i in range(7)]
    protocol=json.loads((DEST/"registration.json").read_text())["protocol"]
    selected=[{**r,"system":"selected"} for r in rows if r["validation_selected_pipeline"]]
    all_rows=rows+selected
    systems=["A","B","C","D","control_matched","control_best","raw_A","raw_C","selected"]
    summary={case:{s:aggregate([r for r in all_rows if r["scenario"]==case and r["system"]==s]) for s in systems}
             for case in protocol["scenarios"]}
    comparisons={f"{a}-{b}":paired(all_rows,a,b) for a,b in (("B","A"),("C","A"),("D","C"),
                                                               ("C","control_matched"),("selected","A"))}
    training=[json.loads((DEST/f"training/fold-{i}/{kind}/training.json").read_text())
              for i in range(7) for kind in protocol["losses"]]
    compute=json.loads((DEST/"compute.json").read_text())
    result={"metrics":summary,"comparisons":comparisons,"selections":selections,"compute":compute,
            "training_seconds":sum(r["seconds"] for r in training),"epochs_executed":sum(len(r["history"]) for r in training),
            "training_arms":len(training),"test_rows":len(rows),
            "peak_rss_bytes":max(r["memory"]["process_peak_rss_bytes"] for r in training)}
    for name in ("verification","error_analysis"):
        path=DEST/(name+".json")
        if path.exists():
            result[name]=json.loads(path.read_text())
    atomic_json(DEST/"summary.json",result)
    flat=[]
    for r in all_rows:
        row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        selection=selections[r["fold"]]
        system=selection["winner"] if r["system"]=="selected" else r["system"]
        epoch=(selection["boundary_epoch"] if system in ("C","D","raw_C") else
               selection["control_matched_epoch"] if system=="control_matched" else
               selection["control_best_epoch"] if system=="control_best" else 0)
        previous=json.loads((OLD/f"selections/fold-{r['fold']}.json").read_text())
        original=previous["conditions"][selection["baseline_condition"]]
        row.pop("selected_epoch")
        row["phase3_epoch_cap"]=row.pop("epoch_cap")
        row.update(phase3_source_epoch=original["selected_epoch"],phase5_epoch=epoch,phase5_max_epochs=protocol["max_epochs"])
        flat.append(row)
    write_csv(DEST/"fold_metrics.csv",flat)
    duration_rows=[]
    boundary_rows=[]
    for r in all_rows:
        for group,item in r["duration_groups"].items():
            for tau,m in item["iou"].items():
                duration_rows.append({"fold":r["fold"],"system":r["system"],"scenario":r["scenario"],
                    "duration_group":group,"IoU":tau,"gt":item["gt"],"pred":item["pred"],
                    "fragmented_gt":item["fragmented_gt"],"fused_predictions":item["fused_predictions"],**m})
        for kind in ("head_HA","decoded_HA"):
            for tolerance,item in r[kind].items():
                boundary_rows.append({"fold":r["fold"],"system":r["system"],"scenario":r["scenario"],
                                      "kind":kind,"tolerance_s":tolerance,**item})
    write_csv(DEST/"duration_metrics.csv",duration_rows)
    write_csv(DEST/"boundary_metrics.csv",boundary_rows)
    curves=[]
    for t in training:
        for h in t["history"]:
            curves.append({"fold":t["fold"],"loss":t["kind"],"epoch":h["epoch"],"seconds":h["seconds"],
                           "train_state":h["training"]["state_loss"],"train_team":h["training"]["team_loss"],
                           "train_boundary":h["training"]["boundary_loss_batch_mean"],
                           "validation_state":h["validation_loss"]["state_loss"],
                           "validation_team":h["validation_loss"]["team_loss"],
                           "validation_boundary":h["validation_loss"]["boundary_loss_batch_mean"],
                           "validation_objective":h["selection"]["objective"],
                           "gradient_norm_max":h["training"]["gradient_norm_max"]})
    write_csv(DEST/"learning_curves.csv",curves)
    plot_curves(training)
    report=render(result,rows,training,protocol)
    (DEST/"report.md").write_text(report)
    return 0


def plot_curves(training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    colors={"control":"#64748b","bce":"#2563eb","focal":"#ea580c"}
    for index,fold in enumerate((0,6)):
        for kind in colors:
            t=next(t for t in training if t["fold"]==fold and t["kind"]==kind)
            x=[h["epoch"] for h in t["history"]]
            axes[index,0].plot(x,[h["validation_loss"]["base_loss"] for h in t["history"]],label=kind,color=colors[kind])
            axes[index,1].plot(x,[h["selection"]["objective"] for h in t["history"]],label=kind,color=colors[kind])
        axes[index,0].set(title=f"Fold {fold}: validación estado + equipo",xlabel="Época de ajuste",ylabel="BCE")
        axes[index,1].set(title=f"Fold {fold}: selección interna",xlabel="Época de ajuste",ylabel="spellF1 ponderado")
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    fig.savefig(DEST/"learning_curves.png",dpi=150)
    fig.savefig(DEST/"learning_curves.svg")
    plt.close(fig)


def render(result,rows,training,protocol):
    metrics=result["metrics"]
    lines=["# Fase 5 — resultados medidos con IDSSE público", "",
        "E0 conserva DeepSets y la TCN bidireccional. La cabeza H/A añade 65 parámetros; la cabeza vivo/muerto sigue separada. "
        "Los targets son cambios directos H↔A nativos a 25 Hz con banda ±120 ms. D, reinicios, desconocidos y discontinuidades no crean positivos. "
        "Las etiquetas son únicamente targets; inferencia recibe las mismas observaciones y máscaras que E0.","",
        "A: checkpoint original y decodificador congelado de Fase 3, semilla 2027. B: mismas emisiones con costes H/A adaptativos derivados de sus probabilidades. "
        "C: ajuste con cabeza auxiliar y decodificador original. D: exactamente el modelo de C con costes adaptativos derivados de la cabeza. "
        "Se añade control sin pérdida auxiliar con el mismo ajuste, datos, corrupción y optimizador. `control_matched` usa exactamente la época de C; `control_best` elige su propia época interna.","",
        "## Protocolo y límites de la inferencia", "",
        "Pilotos en folds canónicos 0 y 6, sin nuevos tests. BCE ponderada y focal (γ=2) forman una cuadrícula pequeña fijada antes del experimento. "
        "Se conservan ambas formulaciones en los otros folds para que cada uno elija exclusivamente con su partido de validación; "
        "no se transfieren decisiones entre folds cuyos modelos han visto otros tests durante el entrenamiento. "
        "Cada ajuste usa cinco partidos IDSSE, ocho épocas máximas, AdamW nuevo a 3e-4, semilla 2027, cuatro hilos CPU, corrupción por bloques del 40 %. "
        "El checkpoint inicial es el seleccionado previamente para ese fold; en el fold 4 contiene preentrenamiento con SkillCorner permitido por su partición. "
        "No se presenta este conjunto como una nueva referencia IDSSE-only pura.","",
        "La época maximiza 0,5×spellF1 limpio + 0,25×ocultación 5 s + 0,25×ocultación 15 s con el decodificador congelado. "
        "Early stopping: mínimo cuatro épocas, paciencia tres, mejora mínima 0,0001; el control ejecuta ocho para permitir comparación a la misma época. "
        "Ponderación positiva min(40, negativos/positivos), normalizada por masa de pesos; pérdida auxiliar ×0,5. "
        "Cada fold elige BCE/focal, intensidad adaptativa {0;0,5;1;2} y umbral {0,25;0,5;0,75} con su validación. "
        "Guardas frente a A: caída máxima 0,02 en cada escenario de ocultación y 0,03 de recall a IoU 0,5 en posesiones ≥10 s. "
        "Las opciones que fallan se informan pero no pueden ganar la selección final. Intensidad cero siempre recupera el decodificador original.","",
        "Las siete selecciones quedaron guardadas antes de la evaluación externa de Fase 5. No se normaliza con test, no se mezclan ventanas entre partidos "
        "y se reutilizan priors originales estimados en entrenamiento. Son experimentos exploratorios sobre siete partidos ya estudiados en fases anteriores; "
        "no equivalen a una confirmación en una cohorte nueva. El ensemble histórico de tres semillas no se promociona por su resultado de test.","",
        "El evaluador oficial permanece intacto. Se decodifica a 10 Hz por tramo continuo y se compara a 25 Hz con GT nativo, sin interpolar etiquetas. "
        "Los costes adaptativos actúan únicamente sobre H/A; las tablas de duración y costes de D permanecen congelados. "
        "No hay un mínimo de posesión de tres segundos. La cabeza usa un filtro de picos de 200 ms sólo para diagnosticar eventos, no para descartar segmentos del decodificador.","",
        "## spellF1 por partido, tracking limpio", "",
        "| Fold / partido | A | B | C | D | Control misma época | Elegido en validación |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in range(7):
        r={r["system"]:r for r in rows if r["fold"]==fold and r["scenario"]=="clean"}
        selection=result["selections"][fold]
        vals=[r[s]["spellF1"] for s in ("A","B","C","D","control_matched",selection["winner"])]
        lines.append(f"| {fold} / {r['A']['match'].split('-')[-1]} | "+" | ".join(f"{x:.5f}" for x in vals)+" |")
    lines += ["", "## Resumen global y robustez", "", "| Escenario | Sistema | spellF1 ± DE | macroF1 | Fragmentadas | Fusiones | Falsas D internas |", "|---|---|---:|---:|---:|---:|---:|"]
    for case in protocol["scenarios"]:
        for s in ("A","B","C","D","control_matched","selected"):
            m=metrics[case][s]
            lines.append(f"| {case} | {s} | {m['spellF1']:.5f} ± {m['spellF1_sd']:.5f} | {m['macroF1']:.5f} | {m['fragmented_gt']} | {m['fused_predictions']} | {m['false_D_interruptions']} |")
    lines += ["", "Media y desviación entre partidos (DE muestral); recuentos sumados. La ocultación mantiene intacto todo el GT y se puntúa el partido completo. "
              "Se eliminan XY/Z, offsets jugador-balón y flags; se recalcula la antigüedad sin convertir ausencia en D. Máscaras y semilla 701 coinciden con Fase 3.", "",
              "## Recuperación por duración, tracking limpio", "",
              "| Sistema | Duración | GT | Recall IoU 0,3 | Recall IoU 0,5 | Recall IoU 0,7 | FN @0,5 | FP @0,5 | Fragmentadas | Fusiones |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ("A","B","C","D","selected"):
        for group,g in metrics["clean"][s]["duration_groups"].items():
            lines.append(f"| {s} | {group} | {g['gt']} | {g['iou']['0.3']['recall']:.3%} | {g['iou']['0.5']['recall']:.3%} | {g['iou']['0.7']['recall']:.3%} | {g['iou']['0.5']['fn']} | {g['iou']['0.5']['fp']} | {g['fragmented_gt']} | {g['fused_predictions']} |")
    lines += ["", "Intervalos: [0,1), [1,3), [3,10), [10,∞) segundos. Matching global greedy oficial, mismo equipo y uno-a-uno antes de estratificar. "
              "FN y recall se asignan por duración GT; FP y precisión por duración predicha. Los TP de ambos lados pueden pertenecer a bins distintos, "
              "por lo que no se fabrica un F1 por bin mezclando denominadores. Fragmentación/fusión son grados de solapamiento con varias posesiones del mismo equipo, no causas exclusivas. "
              "Los CSV incluyen los cinco IoU y los tres escenarios, por partido.", "",
              "## Detección de H/A y efecto del decodificador", "",
              "| Sistema | Señal | Tolerancia | Precisión | Recall | F1 | TP | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ("A","C","D","raw_A","raw_C","selected"):
        for kind in ("head_HA","decoded_HA"):
            for tolerance in ("0.2","0.5","1.0"):
                m=metrics["clean"][s][kind][tolerance]
                lines.append(f"| {s} | {kind} | {tolerance} s | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ["", "La cabeza no predice dirección; sus eventos se emparejan sin tipo. Los cambios H/A decodificados exigen origen y destino correctos. "
              "La sigmoid, entrenada con ponderación y bandas de tolerancia, no se presupone calibrada como probabilidad de evento instantáneo. "
              "A utiliza evidencia proxy de las probabilidades, C/D picos de la cabeza multiplicados por P(vivo), siempre sin GT en la puerta de inferencia. "
              "Umbrales seleccionados con validación a tolerancia 0,5 s. `raw_A`/`raw_C` usan argmax para mostrar qué cambios desaparecen al decodificar. "
              "La comparación C − control misma época separa supervisión auxiliar de tiempo adicional de entrenamiento.", "",
              "## Selecciones internas congeladas", "",
              "| Fold | Pérdida / época C | BCE / focal objetivo | B intensidad | D intensidad / umbral | Umbral eventos | Sistema elegido |", "|---|---|---:|---:|---|---:|---|"]
    for selection in result["selections"]:
        vals={t["kind"]:t["objective"] for t in selection["boundary_trials"]}
        b,d=selection["grids"]["B"]["winner"],selection["grids"]["D"]["winner"]
        lines.append(f"| {selection['fold']} | {selection['boundary_kind']} / {selection['boundary_epoch']} | {vals['bce']:.4f} / {vals['focal']:.4f} | {b['strength']} | {d['strength']} / {d['threshold']} | {selection['head_threshold']} | {selection['winner']} |")
    lines += ["", "## Diferencias pareadas de spellF1 limpio", "", "| Comparación | Diferencia media | Intervalo bootstrap 95 % entre partidos |", "|---|---:|---|"]
    for name,c in result["comparisons"].items():
        lines.append(f"| {name} | {c['delta_mean']:+.5f} | [{c['paired_fold_bootstrap_95'][0]:+.5f}, {c['paired_fold_bootstrap_95'][1]:+.5f}] |")
    lines += ["", "Bootstrap pareado de 10.000 remuestreos, semilla 2027. No corrige múltiples comparaciones ni convierte la cohorte ya observada en evaluación confirmatoria.", "",
              "## Coste real y curvas", "",
              f"Se ejecutaron {result['training_arms']} ajustes, {result['epochs_executed']} épocas nuevas y {result['test_rows']} filas de evaluación. "
              f"Tiempo acumulado de procesos: {result['compute']['seconds']:.1f} s; preparación, entrenamiento, validación y selección incluidos. "
              f"Tiempo de los ajustes: {result['training_seconds']:.1f} s. Pico RSS: {result['peak_rss_bytes']/2**30:.2f} GiB. CPU, cuatro hilos, sin GPU/Colab.","",
              "| Fold | Brazo | Épocas ejecutadas | Época elegida | Segundos | Ponderación positiva |", "|---|---|---:|---:|---:|---:|"]
    for t in training:
        lines.append(f"| {t['fold']} | {t['kind']} | {len(t['history'])} | {t['best_epoch']} | {t['seconds']:.1f} | {t['positive_weight']:.1f} |")
    lines += ["", "![Curvas de los pilotos](learning_curves.png)", "",
              "`learning_curves.csv` registra pérdidas de estado, equipo y frontera, norma de gradiente, tiempo y objetivo interno por época para todos los folds. "
              "`fold_metrics.csv` distingue época del checkpoint de Fase 3 y época de ajuste de Fase 5; cero ajuste significa reutilizar el checkpoint original. "
              "Las pérdidas de frontera BCE/focal tienen escalas distintas; no se comparan sus valores absolutos como criterio de selección.", "",
              "## Reutilizado y descartado", "",
              "Reutilizados: auditorías, siete folds, features mmap, checkpoints, priors y probabilidades originales. No se repiten las 910 ni las 597 evaluaciones anteriores. "
              "No se ejecutaron ASFormer, PathCRF, visión ni búsqueda extensa. Tampoco ensemble nuevo: la comparación factorial mantiene una semilla común. "
              "No se entrenó una rama a 25 Hz ni un refinador geométrico nuevo. Se priorizó el fallo de detección: el oráculo previo de etiquetas muestreadas a 10 Hz ronda 0,995, "
              "de modo que aumentar la rejilla no explica por sí solo la baja recuperación de posesiones cortas. Ese resultado previo es contexto, no un experimento nuevo de Fase 5.", "",
              "## Reproducción", "", "```bash", ".venv/bin/python -m scripts.run_local test-phase5",
              ".venv/bin/python -m scripts.run_local run-phase5 --stage pilots",
              ".venv/bin/python -m scripts.run_local run-phase5 --stage validation",
              ".venv/bin/python -m scripts.run_local run-phase5 --stage test",
              ".venv/bin/python -m scripts.run_local verify-phase5",
              ".venv/bin/python -m scripts.run_local analyse-phase5",
              ".venv/bin/python -m scripts.run_local report-phase5", "```", "",
              "Volver a ejecutar reutiliza pasos terminados y reanuda desde checkpoint con RNG de Python, NumPy, Torch y DataLoader. "
              "El registro congela hashes del código experimental y del evaluador; un cambio exige una versión nueva de experimento. "
              "Todos los artefactos están en `evaluation/reports/external/phase5-v1/`, ignorado por Git. No se ha subido ningún dataset ni checkpoint."]
    lines += component_sections(result) + error_sections(result) + interpretation(result)
    return "\n".join(lines)+"\n"


def component_sections(result):
    lines=["", "## Equipo, balón vivo/muerto y localización de límites", "",
           "| Escenario | Sistema | F1 cabeza vivo/muerto | Exactitud equipo con GT vivo | MAE inicio @IoU0,5 | MAE final @IoU0,5 |",
           "|---|---|---:|---:|---:|---:|"]
    for case,systems in result["metrics"].items():
        for s in ("A","C","D","selected"):
            m=systems[s]
            start=m["matched_start_MAE_s"]
            end=m["matched_end_MAE_s"]
            lines.append(f"| {case} | {s} | {m['head_alive_F1_at_0_5']:.4f} | {m['conditional_team_accuracy_gt_alive']:.4f} | {start:.3f} s | {end:.3f} s |")
    lines += ["", "F1 vivo/muerto usa P(vivo)≥0,5 antes de decodificar; la exactitud de equipo usa argmax condicional H/A sobre frames realmente vivos. "
              "Los errores de inicio/final son medias por partido y sólo incluyen posesiones emparejadas a IoU 0,5: no describen los fallos no recuperados. "
              "C y D comparten cabezas; difieren únicamente en decodificación."]
    return lines


def error_sections(result):
    if "error_analysis" not in result:
        return []
    error=result["error_analysis"]
    lines=["", "## Cambios detectados y retenidos", "",
           "| Salida | GT H/A | Detectados por cabeza C | Cabeza y salida | Cabeza solamente | Correctos argmax C eliminados | Correctos añadidos frente a argmax C | Falsos H/A salida |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s,m in error["clean_boundary_retention"].items():
        lines.append(f"| {s} | {m['gt_HA']} | {m['head_detected']} | {m['head_and_decoded']} | {m['head_only']} | {m['raw_true_removed']} | {m['true_added_over_raw']} | {m['decoded_fp']} |")
    lines += ["", "Tolerancia efectiva 12 frames = 0,48 s. `Cabeza solamente` es detección potencial sin dirección, "
              "no prueba de que insertar ese cambio produzca una posesión correcta. Se empareja cada señal de manera independiente; "
              "las fronteras cercanas pueden ser ambiguas. A se compara aquí con la misma cabeza C para diagnosticar el desacuerdo.", "",
              "| Sistema | Duración | GT con frontera H/A / recall @0,5 | GT con D a ambos lados / recall @0,5 |", "|---|---|---|---|"]
    for s in ("A","C","D"):
        for group in ("lt1","1to3","3to10","ge10"):
            strata=error["clean_possession_strata"][s][group]
            a,b=strata["has_HA_edge"],strata["both_D_edges"]
            lines.append(f"| {s} | {group} | {a['gt']} / {a['recall']:.2%} | {b['gt']} / {b['recall']:.2%} |")
    lines += ["", "Los extremos censurados no necesariamente pertenecen a esas dos categorías. "
              "Este diagnóstico describe el test después de congelar las decisiones y nunca alimenta la selección."]
    return lines


def interpretation(result):
    metrics=result["metrics"]["clean"]
    short={}
    for s in ("A","C","D","selected"):
        groups=metrics[s]["duration_groups"]
        count=sum(groups[g]["gt"] for g in ("lt1","1to3"))
        matched=sum(groups[g]["iou"]["0.5"]["matched_gt"] for g in ("lt1","1to3"))
        short[s]=(matched,count,matched/max(1,count))
    aux=result["comparisons"]["C-control_matched"]
    head=metrics["C"]["head_HA"]["0.5"]
    note=("El intervalo cruza cero: no se demuestra que la pérdida auxiliar por sí sola mejore la generalización."
          if aux['paired_fold_bootstrap_95'][0]<=0<=aux['paired_fold_bootstrap_95'][1] else
          "El intervalo es descriptivo en sólo siete partidos y requiere confirmación independiente.")
    return ["", "## Interpretación y siguiente experimento", "",
            f"La política elegida exclusivamente en validación obtiene spellF1 {metrics['selected']['spellF1']:.5f}. "
            f"La diferencia pareada frente a A es {result['comparisons']['selected-A']['delta_mean']:+.5f}; "
            "esa política puede escoger un sistema distinto en cada fold y no se sustituye por el mayor promedio externo de A/B/C/D.", "",
            f"En posesiones <3 s, A recupera {short['A'][0]}/{short['A'][1]} ({short['A'][2]:.2%}); "
            f"C, {short['C'][0]}/{short['C'][1]} ({short['C'][2]:.2%}); D, {short['D'][0]}/{short['D'][1]} ({short['D'][2]:.2%}); "
            f"la selección interna, {short['selected'][0]}/{short['selected'][1]} ({short['selected'][2]:.2%}), a IoU 0,5.", "",
            f"El efecto auxiliar C − control a la misma época es {aux['delta_mean']:+.5f}, "
            f"intervalo pareado [{aux['paired_fold_bootstrap_95'][0]:+.5f}, {aux['paired_fold_bootstrap_95'][1]:+.5f}]. {note}", "",
            f"La cabeza alcanza recall {head['recall']:.2%} y precisión {head['precision']:.2%} a tolerancia 0,5 s. "
            "Detectar una frontera potencial no basta para decidir su dirección ni para aceptar una posesión completa: "
            "persisten muchas falsas alarmas y las emisiones de equipo pueden oponerse al cambio. "
            f"D reduce fusiones de {metrics['C']['fused_predictions']} a {metrics['D']['fused_predictions']} frente a C, "
            f"pero aumenta fragmentaciones de {metrics['C']['fragmented_gt']} a {metrics['D']['fragmented_gt']}. "
            "La ganancia D−C es pequeña y su intervalo pareado cruza cero. "
            "B sólo evalúa un proxy de cambios de probabilidad con tres frames por lado y esta cuadrícula de intensidades; su resultado no descarta todos los decodificadores adaptativos posibles.", "",
            "Recomendación: continuar con TCN y probar primero una cabeza direccional H→A/A→H y calibración de sus candidatos, "
            "comparadas con la cabeza binaria actual, con el mismo presupuesto y sin cambiar el decodificador inicialmente. "
            "La hipótesis es mejorar la precisión de los cambios antes de relajar más la continuidad. "
            "Como segunda ablación, añadir representación espacial centrada en el balón "
            "(distancias, velocidades y evidencia de control por equipo con máscaras de ausencia), manteniendo la TCN y los folds. "
            "La supervisión actual justifica conservar esta referencia; no demuestra que ASFormer vaya a superar a TCN. "
            "Conviene incorporar partidos adicionales para confirmación independiente antes de seguir afinando sobre esta misma cohorte. "
            "Estas son hipótesis de siguiente fase, no mejoras medidas ni componentes elegidos usando estos tests."]


if __name__=="__main__":
    raise SystemExit(main())
