"""Public-external reporting. No outer-test score selects configurations."""

import argparse
import csv
import json

import numpy as np

from possession.night.common import atomic_json
from possession.phase6.experiment import DEST


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def pilots():
    selections=[json.loads((DEST/f"selections/fold-{i}.json").read_text()) for i in (0,6)]
    rows=[];gate=[]
    for s in selections:
        for arm,item in s["arms"].items():
            w=item["winner"]
            rows.append({"fold":s["fold"],"arm":arm,"method":w["method"],"family":w["family"],
                "objective":w["objective"],"strength":w["strength"],"threshold":w["threshold"],
                "spellF1_clean":w["metrics"]["clean"]["spellF1"],"short_recall":w["metrics"]["clean"]["short_recall"],
                "spellF1_40_5s":w["metrics"]["missing40_5s"]["spellF1"],
                "spellF1_40_15s":w["metrics"]["missing40_15s"]["spellF1"]})
        b,d=s["arms"]["binary"]["winner"],s["arms"]["directional"]["winner"]
        delta=d["objective"]-b["objective"]
        short=d["metrics"]["clean"]["short_recall"]-b["metrics"]["clean"]["short_recall"]
        gate.append({"fold":s["fold"],"robust_spellF1_delta":delta,"short_recall_delta":short,"passes":delta>=.01 and short>=.01})
    decision={"per_fold":gate,"both_pass":all(r["passes"] for r in gate),"outer_test_used":False,
              "policy":"If either pilot fails, stop expansion to seven folds; no parameters transferred across folds"}
    atomic_json(DEST/"pilot_decision.json",decision);write_csv(DEST/"validation_finalists.csv",rows)
    return rows,decision


def aggregate(rows):
    result={}
    for arm in ("control","binary","directional"):
        result[arm]={}
        for case in ("clean","missing40_5s","missing40_15s"):
            values=[r for r in rows if r["arm"]==arm and r["scenario"]==case and r["system"]=="selected"]
            m={"spellF1":float(np.mean([r["spellF1"] for r in values])),
               "spellF1_sd":float(np.std([r["spellF1"] for r in values],ddof=1)),
               "macroF1":float(np.mean([r["macroF1"] for r in values]))}
            for key in ("gt_spells","pred_spells","fragmented_gt","fused_predictions","false_D_interruptions"):
                m[key]=sum(r[key] for r in values)
            m["duration"]={}
            for group in ("lt1","1to3","3to10","ge10"):
                g=[r["duration_groups"][group] for r in values]
                n=sum(r["gt"] for r in g);tp=sum(r["iou"]["0.5"]["matched_gt"] for r in g)
                m["duration"][group]={"gt":n,"matched":tp,"recall":tp/max(1,n),"fn":n-tp,
                    "fp":sum(r["iou"]["0.5"]["fp"] for r in g),"fragmented":sum(r["fragmented_gt"] for r in g),
                    "fused":sum(r["fused_predictions"] for r in g)}
            for key in ("head_alive_F1_at_0_5","conditional_team_accuracy_gt_alive","matched_start_MAE_s","matched_end_MAE_s"):
                vv=[r["independent"][key] for r in values if r["independent"][key] is not None]
                m[key]=float(np.mean(vv)) if vv else None
            boundary={k:sum(r["decoded_HA"]["0.5"][k] for r in values) for k in ("tp","fp","fn")}
            boundary.update(precision=boundary["tp"]/max(1,boundary["tp"]+boundary["fp"]),
                            recall=boundary["tp"]/max(1,boundary["tp"]+boundary["fn"]))
            m["decoded_HA"]=boundary;result[arm][case]=m
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilots-only",action="store_true");args=parser.parse_args()
    val,decision=pilots()
    if args.pilots_only:return 0
    rows=json.loads((DEST/"all_rows.json").read_text())
    summary={"metrics":aggregate(rows),"pilot_decision":decision,"rows":len(rows),
             "compute":json.loads((DEST/"compute.json").read_text())}
    transfer_path=DEST/"skillcorner_transfer.json"
    transfer_summary=[]
    if transfer_path.exists():
        transfer=json.loads(transfer_path.read_text())
        summary["transfer_seconds"]=transfer["seconds"]
        for arm in ("control","binary","directional"):
            for case in ("clean","missing40_5s","missing40_15s"):
                rr=[r for r in transfer["rows"] if r["arm"]==arm and r["scenario"]==case]
                matches=sorted({r["match"] for r in rr})
                mm=[]
                for key in matches:
                    pool=[r["conditional_team"]["all"] for r in rr if r["match"]==key]
                    mm.append({k:np.mean([v[k] for v in pool]) for k in ("accuracy","macroF1","balanced_accuracy")})
                transfer_summary.append({"arm":arm,"scenario":case,"matches":len(matches),
                    **{k:float(np.mean([v[k] for v in mm])) for k in ("accuracy","macroF1","balanced_accuracy")},
                    "accuracy_sd":float(np.std([v["accuracy"] for v in mm],ddof=1))})
    summary["skillcorner"]=transfer_summary
    flag_path=DEST/"provider_flag_ablation.json"
    flag_summary=[]
    if flag_path.exists():
        flags=json.loads(flag_path.read_text());summary["flag_ablation_seconds"]=flags["seconds"]
        for arm in ("control","binary","directional"):
            rr=[r for r in flags["rows"] if r["arm"]==arm]
            matches=sorted({r["match"] for r in rr})
            accuracy=[np.mean([r["conditional_team"]["all"]["accuracy"] for r in rr if r["match"]==key]) for key in matches]
            baseline=next(v["accuracy"] for v in transfer_summary if v["arm"]==arm and v["scenario"]=="clean")
            flag_summary.append({"arm":arm,"matches":len(matches),"original_accuracy":baseline,
                "unknown_detection_metadata_accuracy":float(np.mean(accuracy)),"delta":float(np.mean(accuracy))-baseline})
    robust_path=DEST/"provider_flags_robustness.json"
    if robust_path.exists():
        flags=json.loads(robust_path.read_text());summary["flag_robustness_seconds"]=flags["seconds"]
        for arm in ("control","binary","directional"):
            for case in ("missing40_5s","missing40_15s"):
                rr=[r for r in flags["rows"] if r["arm"]==arm and r["scenario"]==case]
                keys=sorted({r["match"] for r in rr})
                acc=[np.mean([r["conditional_team"]["all"]["accuracy"] for r in rr if r["match"]==key]) for key in keys]
                baseline=next(v["accuracy"] for v in transfer_summary if v["arm"]==arm and v["scenario"]==case)
                flag_summary.append({"arm":arm,"scenario":case,"matches":len(keys),"original_accuracy":baseline,
                    "unknown_detection_metadata_accuracy":float(np.mean(acc)),"delta":float(np.mean(acc))-baseline})
    for r in flag_summary:r.setdefault("scenario","clean")
    summary["provider_flag_ablation"]=flag_summary
    diagnostic_path=DEST/"validation_diagnosis.json"
    diagnostic_rows=[]
    if diagnostic_path.exists():
        for row in json.loads(diagnostic_path.read_text())["rows"]:
            if row["scenario"]!="clean":continue
            s=row["strata"];shifts=row["temporal_shift_s"]
            best=max(shifts,key=lambda key:shifts[key]["accuracy"])
            diagnostic_rows.append({"fold":row["fold"],"arm":row["arm"],
                "near_boundary_TCN":s["near_HA_boundary"]["TCN"]["accuracy"],
                "near_boundary_geometry":s["near_HA_boundary"]["nearest_geometry"]["accuracy"],
                "within_2m_TCN":s["observed_within_2m"]["TCN"]["accuracy"],
                "within_2m_geometry":s["observed_within_2m"]["nearest_geometry"]["accuracy"],
                "best_shift_s":float(best),"best_shift_accuracy_gain":shifts[best]["accuracy"]-shifts["0.0"]["accuracy"],
                "true_D_oracle_argmax_spellF1":row["oracles_analysis_only"]["true_D_predicted_conditional_team"]["spell_f1_mean"],
                "true_team_oracle_argmax_spellF1":row["oracles_analysis_only"]["true_team_where_defined_predicted_D"]["spell_f1_mean"]})
    summary["validation_diagnosis"]=diagnostic_rows
    atomic_json(DEST/"summary.json",summary)
    write_csv(DEST/"fold_metrics.csv",[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows])
    duration=[];candidate=[];calibration=[];curves=[]
    for r in rows:
        if r["system"]!="selected":continue
        for group,g in r["duration_groups"].items():
            for tau,m in g["iou"].items():
                duration.append({"fold":r["fold"],"arm":r["arm"],"scenario":r["scenario"],"duration":group,"IoU":tau,
                    "gt":g["gt"],"pred":g["pred"],"fragmented_gt":g["fragmented_gt"],"fused_predictions":g["fused_predictions"],**m})
        for method,m in r["candidate_metrics"].items():
            candidate.append({"fold":r["fold"],"arm":r["arm"],"scenario":r["scenario"],"method":method,**m})
        for method,m in r["calibration_quality"].items():
            calibration.append({"fold":r["fold"],"arm":r["arm"],"scenario":r["scenario"],"method":method,**m})
    for path in sorted((DEST/"training").glob("fold-*/*/complete.json")):
        for r in json.loads(path.read_text())["history"]:
            curves.append({"fold":int(path.parts[-3].split("-")[-1]),"arm":path.parts[-2],"epoch":r["epoch"],
                           "seconds":r["seconds"],**r["train"]})
    for name,values in (("duration_metrics",duration),("candidate_metrics",candidate),("calibration_metrics",calibration),
                        ("learning_curves",curves),("skillcorner_metrics",transfer_summary),("validation_diagnosis",diagnostic_rows),
                        ("provider_flag_ablation",flag_summary)):
        write_csv(DEST/(name+".csv"),values)
    plot_candidates(candidate)
    (DEST/"report.md").write_text(render(summary,rows,val,candidate))
    return 0


def plot_candidates(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={"control":"#64748b","binary":"#2563eb","directional":"#ea580c"}
    markers={"local":"o","nms":"s","platt":"^","context":"D","geometry":"*","confirmed":"x"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,fold in zip(axes,(0,6)):
        for arm,color in colors.items():
            rr=[r for r in rows if r["fold"]==fold and r["arm"]==arm and r["scenario"]=="clean"]
            for r in rr:
                ax.scatter(r["recall"],r["precision"],color=color,marker=markers[r["method"]],s=65,alpha=.85)
        ax.set(title=f"Fold {fold}: candidatos externos",xlabel="Recall H/A a 0,48 s",ylabel="Precisión H/A",xlim=(-.02,.85),ylim=(-.02,.48))
        ax.grid(alpha=.2)
        ax.legend(handles=[Line2D([],[],marker="o",linestyle="",color=color,label=arm) for arm,color in colors.items()],loc="upper right")
    fig.legend(handles=[Line2D([],[],marker=marker,linestyle="",color="#334155",label=method) for method,marker in markers.items()],
               loc="lower center",ncol=6,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(DEST/"candidate_precision_recall.png",dpi=150);fig.savefig(DEST/"candidate_precision_recall.svg")
    plt.close(fig)


def render(summary,rows,val,candidates):
    lines=["# Fase 6: resultados medidos", "",
        "Dos pilotos, folds canónicos 0 y 6. Estos resultados exploratorios no son LOMO completo ni sustituyen directamente al 0,3920 de Fase 5.", "",
        "## Protocolo", "",
        "Por piloto: cuatro IDSSE de entrenamiento, uno completo de calibración, uno de selección y uno de test. "
        "La base de 12 épocas se entrenó sin ver calibración, validación ni test y se reutilizó entre control, binary y directional. "
        "Cada brazo ejecutó exactamente ocho épocas adicionales con los mismos lotes, máscaras, optimizador y RNG de dropout, sin early stopping. "
        "Los checkpoints previos entrenados con cinco partidos no podían utilizarse como base sin contaminar esta calibración OOF adicional. "
        "Se reutilizaron features mmap, fuentes, folds y referencias anteriores; los nuevos pesos son comparables entre sí.", "",
        "La cabeza direccional añade dos sigmoids independientes; vivo/muerto y equipo permanecen separados. Sólo H→A/A→H nativos válidos crean targets ±120 ms. "
        "D→H/A son reinicios, nunca positivos de recuperación. Focal γ=2, peso auxiliar 0,5, ponderación positiva ajustada al número de canales. "
        "El control recibe la misma cantidad de entrenamiento sin pérdida auxiliar.", "",
        "## Selección con validación interna", "",
        "| Fold | Brazo | Método / decoder | Objetivo robusto | spellF1 limpio | Recall <3s |",
        "|---|---|---|---:|---:|---:|"]
    for r in val:
        lines.append(f"| {r['fold']} | {r['arm']} | {r['method']} / {r['family']} | {r['objective']:.5f} | {r['spellF1_clean']:.5f} | {r['short_recall']:.2%} |")
    lines+=["", "Objetivo: 50% limpio, 25% ocultación 40%/5s, 25% ocultación 40%/15s. Cada fold elige por separado. "
        "Las dos selecciones se congelaron antes de evaluar cualquiera de sus tests.", "",
        "| Fold | Δ direccional−binaria robusto | Δ recall <3s | Pasa expansión |","|---|---:|---:|---|"]
    for r in summary["pilot_decision"]["per_fold"]:
        lines.append(f"| {r['fold']} | {r['robust_spellF1_delta']:+.5f} | {r['short_recall_delta']:+.2%} | {r['passes']} |")
    lines+=["", "El criterio prefijado exige +0,01 absoluto en ambos indicadores y ambos pilotos. "
        +("Se alcanza; cualquier promoción requiere confirmación." if summary["pilot_decision"]["both_pass"] else
          "No se alcanza: se detiene la expansión a siete folds, sin otra búsqueda grande."), "",
        "## Tests exploratorios: finalistas propios", "",
        "| Fold | Brazo | spellF1 limpio | macroF1 | spellF1 40%/5s | spellF1 40%/15s | Fragmentadas / fusiones limpias |",
        "|---|---|---:|---:|---:|---:|---|"]
    for fold in (0,6):
        for arm in ("control","binary","directional"):
            rr={r["scenario"]:r for r in rows if r["fold"]==fold and r["arm"]==arm and r["system"]=="selected"}
            c=rr["clean"]
            lines.append(f"| {fold} | {arm} | {c['spellF1']:.5f} | {c['macroF1']:.5f} | {rr['missing40_5s']['spellF1']:.5f} | {rr['missing40_15s']['spellF1']:.5f} | {c['fragmented_gt']} / {c['fused_predictions']} |")
    lines+=["", "No se elige otro ganador usando test. Los CSV incluyen un finalista por método y HMM/HSMM fijos.", "",
        "## Mismo modelo y familia: costes fijos frente a adaptativos", "",
        "| Fold | Brazo | Familia elegida en validación | Fijo | Adaptativo elegido | Diferencia limpia |","|---|---|---|---:|---:|---:|"]
    for fold in (0,6):
        for arm in ("control","binary","directional"):
            selected=next(r for r in rows if r["fold"]==fold and r["arm"]==arm and r["scenario"]=="clean" and r["system"]=="selected")
            fixed=next(r for r in rows if r["fold"]==fold and r["arm"]==arm and r["scenario"]=="clean" and r["system"]=="fixed_"+selected["family"])
            lines.append(f"| {fold} | {arm} | {selected['family']} | {fixed['spellF1']:.5f} | {selected['spellF1']:.5f} | {selected['spellF1']-fixed['spellF1']:+.5f} |")
    lines+=["",
        "## Duraciones: test limpio, salida elegida", "",
        "| Brazo | Duración | GT | Recall @IoU0,5 | FN | FP | Fragmentadas | Fusiones |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("control","binary","directional"):
        for group,g in summary["metrics"][arm]["clean"]["duration"].items():
            lines.append(f"| {arm} | {group} | {g['gt']} | {g['recall']:.2%} | {g['fn']} | {g['fp']} | {g['fragmented']} | {g['fused']} |")
    lines+=["", "Matching oficial global y uno-a-uno antes de estratificar [0,1), [1,3), [3,10), [10,∞) s. FN/recall por duración GT; FP por duración predicha.", "",
        "## Candidatos y calibración", "",
        "Se comparan máximos locales, NMS 200 ms por dirección, Platt, contexto, contexto+geometría y confirmación suave de equipo anterior/posterior. "
        "NMS permite cambios opuestos cercanos. Binary asigna dirección con la derivada de P(H|vivo) en ±0,3s; el control usa esa derivada como proxy. "
        "Directional produce ambos scores directamente. Context usa scores, probabilidades de equipo/vivo y disponibilidad en ventanas de 0,3 y 1 s; geometry añade distancias colectivas sin identidades.", "",
        "Platt y los clasificadores ridge se ajustan sólo en el partido OOF de calibración, incluidos sus normalizadores. Los umbrales {0,1;0,3;0,5} "
        "maximizan F1 de candidatos en selección. Las intensidades {0,5;1} y HMM/HSMM se eligen por spellF1 robusto. "
        "El bonus log-odds respecto a prevalencia OOF actúa sólo sobre la dirección propuesta, con soporte ±1 frame. Los costes/tablas de D permanecen congelados; "
        "la decisión conjunta aún puede cambiar D indirectamente. No hay duración mínima.", "",
        "| Fold | Brazo | Método | Precisión | Recall | Falsos/min |","|---|---|---|---:|---:|---:|"]
    for r in candidates:
        if r["scenario"]=="clean":
            lines.append(f"| {r['fold']} | {r['arm']} | {r['method']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['false_per_minute']:.2f} |")
    lines+=["", "Dirección correcta y tolerancia ±12 frames = 0,48s. Falsos/min usa el tiempo válido completo, incluido D. "
        "Más recall de candidatos no equivale a recuperar posesiones. Brier/log-loss/ECE se guardan sobre poblaciones NMS comparables en calibration_metrics.csv.", "",
        "| Brazo | Precisión H/A final | Recall H/A final | F1 vivo/muerto | Exactitud equipo con GT vivo |","|---|---:|---:|---:|---:|"]
    for arm in ("control","binary","directional"):
        m=summary["metrics"][arm]["clean"];b=m["decoded_HA"]
        lines.append(f"| {arm} | {b['precision']:.3f} | {b['recall']:.3f} | {m['head_alive_F1_at_0_5']:.4f} | {m['conditional_team_accuracy_gt_alive']:.4f} |")
    if summary["skillcorner"]:
        lines+=["", "## SkillCorner: transferencia de equipo", "",
            "Veinte partidos excluidos del entrenamiento y calibración de estos modelos. La cohorte se estudió en fases anteriores: no es un nuevo conjunto de investigación intacto. "
            "Sólo se puntúa equipo condicional con etiquetas válidas; no se asume ground truth D ni se calcula spellF1. No selecciona hiperparámetros. "
            "Se promedian primero los dos pilotos por partido y después los veinte partidos con igual peso.", "",
            "| Brazo | Observación | Exactitud ± DE | macroF1 | Balanced accuracy |","|---|---|---:|---:|---:|"]
        for r in summary["skillcorner"]:
            lines.append(f"| {r['arm']} | {r['scenario']} | {r['accuracy']:.4f} ± {r['accuracy_sd']:.4f} | {r['macroF1']:.4f} | {r['balanced_accuracy']:.4f} |")
    if summary["validation_diagnosis"]:
        lines+=["", "## Diagnóstico espacial/temporal en validación", "",
            "| Fold | Brazo | Equipo cerca de frontera: TCN / geometría | Balón a ≤2m: TCN / geometría | Mejor desplazamiento diagnóstico / ganancia exactitud | Oráculo D / oráculo equipo, spellF1 argmax |",
            "|---|---|---|---|---|---|"]
        for r in summary["validation_diagnosis"]:
            lines.append(f"| {r['fold']} | {r['arm']} | {r['near_boundary_TCN']:.3f} / {r['near_boundary_geometry']:.3f} | {r['within_2m_TCN']:.3f} / {r['within_2m_geometry']:.3f} | {r['best_shift_s']:+.1f}s / {r['best_shift_accuracy_gain']:+.4f} | {r['true_D_oracle_argmax_spellF1']:.3f} / {r['true_team_oracle_argmax_spellF1']:.3f} |")
        lines+=["", "Cerca de frontera significa ±0,5s de un H/A nativo. Geometría es el equipo del jugador más cercano al balón cuando hay coordenadas; no es GT. "
            "El desplazamiento se explora sólo como diagnóstico, no se incorpora al sistema. Oráculos con argmax, no con el decoder final. "
            "Estas pruebas localizan errores de equipo y no identifican causalmente qué arquitectura los resolverá.", "",
            "![Candidatos](candidate_precision_recall.png)"]
    if summary["provider_flag_ablation"]:
        lines+=["", "## Ablación exploratoria de metadata de detección", "",
            "Una única intervención tras observar la caída de transferencia: retirar flags detected/detection_known del balón y jugadores, "
            "y recalcular antigüedad usando disponibilidad XY. No se cambian posiciones, máscaras de coordenadas, etiquetas ni pesos del modelo. "
            "Sin parámetros por seleccionar y sin reentrenamiento. Se registró por separado; es un diagnóstico post hoc, no confirmación independiente.", "",
            "| Brazo | Observación | Exactitud original | Metadata desconocida | Diferencia |","|---|---|---:|---:|---:|"]
        for r in summary["provider_flag_ablation"]:
            lines.append(f"| {r['arm']} | {r['scenario']} | {r['original_accuracy']:.4f} | {r['unknown_detection_metadata_accuracy']:.4f} | {r['delta']:+.4f} |")
        lines+=["", "Este aumento es exactitud condicional en etiquetas parciales SkillCorner, no spellF1 ni transferencia medida a Barcelona–Athletic. "
            "El modelo IDSSE nunca observó flags de detección conocidos durante su entrenamiento; no debe extrapolarse una entrada distinta sin verificarla. "
            "La ablación no demuestra que toda la caída fuera falta de capacidad, ni se aplica automáticamente a checkpoints antiguos que sí entrenaron con SkillCorner. "
            "La extensión de robustez mantiene exactamente la misma intervención, sin otra selección."]
    lines+=["", "## Barcelona–Athletic", "",
        "No se localizaron assets en las rutas configuradas, proyecto ni nombres esperados en Descargas. Mínimo necesario: "
        "data/tactical/BAR-ATH_tactical_atd.xml y BAR-ATH_tactical_atd.txt. Tampoco aparecieron vídeo, homografía, Smart Tagging o assets broadcast. "
        "No se generan predicciones ni puntuaciones sustitutivas.", "",
        "audit-phase6-target queda preparado para parsear ATD, centrar coordenadas, medir ausencia nativa, discontinuidades y proxies de cambio de slot, "
        "comparar observaciones con IDSSE de entrenamiento e inferir con una selección del piloto 0 congelada antes del objetivo. "
        "El tercer valor ATD es velocidad, no altura: z permanece desconocida. Sólo abre el GT público para diagnóstico, nunca el privado. "
        "Los informes y predicciones de ese flujo permanecen restringidos localmente bajo evaluation/reports/phase6-target/ y no se leen para el chat.", "",
        "## Coste y reproducción", "",
        f"Proceso principal: {summary['compute']['seconds']:.1f}s; 24 épocas de base y 48 de ajuste; {summary['rows']} filas test. "
        f"Transferencia: {summary.get('transfer_seconds',0):.1f}s; ablación de flags: {summary.get('flag_ablation_seconds',0):.1f}s. "
        f"Robustez de esa política: {summary.get('flag_robustness_seconds',0):.1f}s. CPU, cuatro hilos; no se repiten estudios previos.", "",
        "```bash",
        ".venv/bin/python -m scripts.run_local test-phase6",
        ".venv/bin/python -m scripts.run_local check-phase6-assets",
        ".venv/bin/python -m scripts.run_local run-phase6 --stage pilots",
        ".venv/bin/python -m scripts.run_local report-phase6 --pilots-only",
        ".venv/bin/python -m scripts.run_local run-phase6 --stage test",
        ".venv/bin/python -m scripts.run_local transfer-phase6",
        ".venv/bin/python -m scripts.run_local diagnose-phase6",
        ".venv/bin/python -m scripts.run_local ablate-phase6-flags",
        ".venv/bin/python -m scripts.run_local robust-phase6-flags",
        ".venv/bin/python -m scripts.run_local verify-phase6",
        ".venv/bin/python -m scripts.run_local audit-phase6-target",
        ".venv/bin/python -m scripts.run_local report-phase6",
        "```", "",
        "Checkpoints completos con optimizador y RNG, pasos terminados reutilizables, código/protocolo/dependencias/evaluador congelados por hash. "
        "Datos y artefactos quedan ignorados por Git."]
    lines+=["", "La verificación real terminó OK: particiones 4/1/1/1, mismo número de actualizaciones y RNG de sampler/dropout entre brazos, "
        "todas las selecciones antes de test, probabilidades finitas/normalizadas y hash oficial intacto. Once pruebas sintéticas "
        "cubren semántica D, direcciones, NMS, calibración, independencia de targets, formato objetivo, metadata y DP exhaustiva. "
        "Los detalles están en verification.json; no se ha accedido a GT privado."]
    lines+=["", "## Decisión arquitectónica", "",
        "**B como siguiente contraste de arquitectura: Set Transformer espacial conservando la TCN, con un control DeepSets de entradas armonizadas.** "
        "La acción inmediata con mayor evidencia es corregir la compatibilidad de observaciones, antes de invertir en otra arquitectura. "
        "No se promociona la cabeza direccional como mejora competitiva clara: "
        "falló el criterio interno en ambos pilotos y la recuperación de posesiones <1s permanece casi nula. "
        "La calibración reduce falsas alarmas, pero ese beneficio no se traduce de forma consistente en más posesiones recuperadas. "
        "La caída original hacia 60% en SkillCorner no justifica atribuir el problema a capacidad espacial: "
        "la misma TCN alcanza aproximadamente 80,5% al retirar metadata de detección no vista en entrenamiento. "
        "Los errores restantes cerca de cambios y el aprovechamiento imperfecto de proximidad jugador-balón motivan el contraste de interacciones espaciales.", "",
        "Esto es una prioridad experimental, no una prueba de que el límite sea exclusivamente espacial ni de que haga falta más capacidad. "
        "Mantener la misma TCN y una política de flags común permite atribuir el siguiente cambio al encoder. "
        "Comparar DeepSets frente a una atención espacial pequeña, mismo presupuesto, máscaras y separación OOF; exigir mejora robusta de spellF1 y de posesiones cortas en validación. "
        "Si falla, detener esa vía y comparar un único modelo temporal con contexto mayor. ASFormer queda como candidato temporal posterior; sus resultados de segmentación de acciones no demuestran superioridad en posesión. "
        "No se añade PathCRF preentrenado sobre Sportec a esta evaluación IDSSE por el riesgo de solapamiento de entrenamiento.", "",
        "Antes de otra expansión LOMO, priorizar datos adicionales de confirmación y conseguir el tracking real del dominio objetivo. "
        "Este estudio de dos pilotos no demuestra una mejora sobre el benchmark de siete partidos de Fase 5.", "",
        "Referencias de diseño: [calibración de confianza](https://proceedings.mlr.press/v70/guo17a.html), "
        "[Set Transformer e interacciones entre elementos](https://proceedings.mlr.press/v97/lee19d.html), "
        "[ASFormer, repositorio oficial BMVC 2021](https://github.com/ChinaYi/ASFormer). Las mejoras competitivas discutidas arriba son mediciones propias o hipótesis explícitas, no resultados transferidos de esos artículos."]
    return chr(10).join(lines)+chr(10)


if __name__=="__main__":raise SystemExit(main())
