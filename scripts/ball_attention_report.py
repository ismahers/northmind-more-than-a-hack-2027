"""Render only public aggregate pilot statistics; no datasets or logs opened."""
from pathlib import Path
from possession.temporal_probe.common import read_json
from possession.ball_attention.common import ROOT,DEST
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json


def count(metric,group):
    g = metric["duration_groups"][group]
    return f'{g["iou"]["0.5"]["matched_gt"]}/{g["gt"]}'


def table(rows,header):
    return ["| "+" | ".join(header)+" |","|"+"|".join("---" for _ in header)+"|"] + ["| "+" | ".join(map(str,r))+" |" for r in rows]


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    for col,fold in enumerate(("0","6")):
        for row,side in enumerate(("entry","exit")):
            ax = axes[row,col]
            for arm,label in (("original","A: fixed queries"),("ball","B: ball queries"),("capacity","C: capacity")):
                curves = summary["results"]["validation"][fold]["diagnosis"][arm]["clean"]["curves"]
                values = sorted((float(k.split("/")[-1]),v["probability"]["mean"])
                    for k,v in curves.items() if k.startswith("HA_HA/"+side+"/"))
                ax.plot([v[0] for v in values],[v[1] for v in values],marker="o",markersize=3,label=label)
            ax.axhline(.5,color="gray",lw=.7,ls=":");ax.axvline(0,color="gray",lw=.7,ls=":")
            ax.set_title(f"Validation fold {fold} — {side}");ax.grid(alpha=.15);ax.set_ylim(0,1)
    axes[0,0].legend(fontsize=8,loc="upper left")
    axes[1,0].set_xlabel("Seconds relative to GT boundary");axes[1,1].set_xlabel("Seconds relative to GT boundary")
    fig.supylabel("P(short-spell team | alive), mean per spell")
    fig.suptitle("Direct H/A short spells — own validation, clean")
    fig.tight_layout(rect=(.025,0,1,.95))
    folder = DEST/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/"validation_transition_probabilities.png",dpi=160)
    fig.savefig(folder/"validation_transition_probabilities.svg")
    plt.close(fig)


def main():
    s = read_json(DEST/"summary.json")
    if not s["verification_passed"]:
        raise ValueError("Verified final pilot required")
    success = s["ball_passes_both_validations"]
    lines = ["# Ball-conditioned spatial attention — resultados","","23 de octubre de 2026.","",
        "**"+("Evidencia favorable en ambas validaciones: plantear ampliación controlada, sin ejecutarla." if success else
        "Cerrar esta variante arquitectónica. Conservar ST+TCN IDSSE-only; no ampliar a siete folds.")+"**","",
        "Protocolo prefijado: [BALL_CONDITIONED_ATTENTION.md](BALL_CONDITIONED_ATTENTION.md). "
        "Seis modelos nuevos, folds0/6 y semillas2027/2028; 12 épocas base y ocho auxiliares, "
        "CPU FP32/cuatro hilos. Cambia solo la consulta PMA; SAB, TCN, cabezas, features, pérdidas y HMM permanecen iguales.","",
        "A es una réplica nueva emparejada de ST+TCN con sus cabezas propias. No lleva la MLP local posterior del diagnóstico209/133. "
        "Ese conteo anterior no es un baseline idéntico para atribuir diferencias a esta atención. "
        "Se comparan también cohortes fijadas desde el A nuevo, por partición y escenario.","",
        "## Validación y decisión","",
        "Objetivo robusto0,50·limpio+0,25·oculto5+0,25·oculto15. Las dos selecciones se sellaron antes de test. "
        "B debe mejorar robusto≥0,005 y recall corto≥0,01 frente A, superar C en≥0,005 robusto y respetar las guardas.",""]
    rows = []
    for fold,sel in s["selections"].items():
        objectives = sel["objectives"]
        rows.append([fold,*[f"{objectives[a]:.6f}" for a in ("original","ball","capacity")],
            f'{objectives["ball"]-objectives["original"]:+.6f}',sel["winner"],sel["gates"]["ball"]["passed"]])
    lines += table(rows,["Fold","A robusto","B balón","C capacidad","B−A","Elegido","B pasa"])
    for fold,sel in s["selections"].items():
        failed = [k for k,v in sel["gates"]["ball"]["checks"].items() if not v and not k.startswith("vs_teacher_")]
        lines += ["",f"Fold {fold}, guardas de B incumplidas: "+(", ".join(failed) if failed else "ninguna")+"."]
    lines += ["","En este contraste las guardas se aplican al A nuevo. El helper histórico conserva "
        "en JSON campos duplicados `vs_teacher_*`, cuyo baseline aquí es el mismo A; no es una guarda adicional frente a otro profesor.","",
        "La mejora de B en fold0 es menor que0,005 y no supera C; además las fragmentaciones limpias crecen16→23 "
        "(máximo permitido18). En fold6 el recall largo limpio baja94/110→90/110 (−0,03636, frente al suelo−0,03); "
        "con ocultación5, las falsas posesiones/min suben1,80196→1,92841 (>+0,1) y las fragmentaciones33→41 "
        "(máximo35). C tampoco pasa ambos folds: su mejora en fold0 incumple fragmentación y en fold6 pierde robustez.","",
        "En conjunto, las validaciones mantienen159 cortas perdidas sin evidencia fuerte (A159/B159), "
        "y los casos sin ningún frame condicional correcto empeoran104→117. La cohorte fija de104 casos originales "
        "solo aporta1 recuperación HMM en B; C también aporta1. Los errores operativos dominados por clasificación "
        "pasan138→140, mientras las recuperaciones crudas perdidas por HMM bajan16→14. No es una mejora consistente "
        "de discriminación H/A atribuible al condicionamiento dinámico.","",
        "B sí aprende consultas variables: norma media del residual observado1,070/1,260 en los dos folds, "
        "variación media por componente0,071/0,097; C tiene variación0. No se explica el resultado por una rama "
        "inactiva ni por un fallback roto. La evidencia del equipo nuevo a+0,5s en entradas H↔A no mejora "
        "(A/B:0,5244/0,5206 y0,4515/0,4400). Se cierra esta receta; no demuestra que cualquier representación "
        "espacial condicionada sea incapaz de ayudar."]
    for part,title in (("validation","Validación propia"),("test","Test descriptivo, posterior al sello")):
        lines += ["",f"## {title}","",
            "Recuperación con matching oficial global a IoU0,5; largo significa≥10s. "
            "Las tasas de falsas posesiones/transiciones se normalizan por minutos puntuables.",""]
        rows = []
        for fold,value in s["results"][part].items():
            for arm,metrics in value["metrics"].items():
                m = metrics["clean"]
                rows.append([fold,arm,f'{m["spellF1"]:.5f}',count(m,"lt1"),count(m,"1to3"),count(m,"ge10"),
                    f'{metrics["missing40_5s"]["spellF1"]:.5f}',f'{metrics["missing40_15s"]["spellF1"]:.5f}',
                    f'{m["false_possessions_per_minute"]:.3f}',m["fused_predictions"],m["fragmented_gt"],f'{m["false_transitions_per_minute"]:.3f}'])
        lines += table(rows,["Fold","Brazo","spellF1 limpio","<1s","1–3s","≥10s","Oculto5","Oculto15","Falsas pos/min","Fusiones","Fragmentaciones","Falsas H↔A/min"])
        lines += ["","### Emisiones y decodificación, cortas limpias","",
            "Evidencia fuerte: Pcorrecta≥0,60 en≥50% del spell. Fuertes/insuficientes particionan las no recuperadas. "
            "Sin frame correcto y suprimidas por HMM son subconjuntos; no sumarlos como causas disjuntas.",""]
        rows = []
        for fold,value in s["results"][part].items():
            for arm,diagnosis in value["diagnosis"].items():
                d = diagnosis["clean"]["all"];errors = d["error_types"];stats = d["statistics"]
                rows.append([fold,arm,d["n"],d["hmm_recovered"],d["A_strong_emissions_missed"],d["B_insufficient_emissions_missed"],
                    d["B_no_correct_conditional_frame"],d["C_raw_recovered_HMM_lost"],errors["classification"],errors["alive_masking"],errors["temporal_extent"],
                    f'{stats["p_correct"]["mean"]:.4f}',f'{stats["signed_logit_margin"]["mean"]:+.4f}'])
        lines += table(rows,["Fold","Brazo","N cortas","HMM recupera","Fuertes perdidas","Insuficientes","Sin frame correcto","Crudo sí/HMM no","Clasificación","Máscara viva","Extensión/intermitencia","Pcorrecta","Margen logit"])
        lines += ["","### Cohortes congeladas desde A, limpio","",
            "Las cohortes siguientes conservan los mismos spells GT al comparar B/C, sin redefinirse según sus mejoras.",""]
        rows = []
        for fold,value in s["results"][part].items():
            for arm,scenarios in value["frozen_original_cohorts"].items():
                for cohort in ("no_correct_frame","insufficient","decoder_lost","physical_conflict"):
                    c = scenarios["clean"][cohort]
                    rows.append([fold,arm,cohort,c["n"],c["recovered"],c["raw_recovered"],c["any_correct_frame"],c["majority_correct"],
                        f'{c["p_correct_change"]:+.4f}' if c["p_correct_change"] is not None else "—"])
        lines += table(rows,["Fold","Brazo","Cohorte A","N","Recupera HMM","Recupera crudo","Algún frame correcto","Mayoría correcta","Δ Pcorrecta"])
    legacy = read_json(DEST/"legacy-cohorts.json")
    if not legacy["ancestry_checks_passed"] or legacy["pilot_summary_sha256"]!=file_hash(DEST/"summary.json"):
        raise ValueError("Legacy diagnostic ancestry mismatch")
    lines += ["","## Los133 casos del diagnóstico original, conservados","",
        "Alineación adicional de las cohortes HA_DISCRIMINATION con la misma secuencia GT de spells; solo se leen casos "
        "ya calculados, después de los sellos. Cero entrenamiento o inferencias nuevas. No es un efecto arquitectónico "
        "emparejado frente a local8, porque cambia la cabeza posterior; el efecto emparejado es el A/B/C anterior.","",
        "B no recupera ninguna de las133 cortas test originales sin un solo frame correcto, ni ninguna de las30 "
        "con conflicto físico. Introduce algún frame correcto en17/133, pero solo nueve alcanzan mayoría correcta "
        "y ninguna completa el matching HMM. De las209 originalmente insuficientes recupera3. "
        "En validación recupera1/91 sin frame y1/26 conflictos físicos originales. Tampoco se resuelve esa cohorte prioritaria.",""]
    rows = []
    for part,folds in legacy["results"].items():
        for fold,arms in folds.items():
            for arm,cohorts_by_name in arms.items():
                for name in ("no_correct_frame","insufficient","physical_conflict"):
                    c = cohorts_by_name[name]
                    rows.append([part,fold,arm,name,c["n"],c["recovered"],c["any_correct_frame"],c["majority_correct"]])
    lines += table(rows,["Partición","Fold","Brazo nuevo","Cohorte local8 original","N","HMM recupera","Algún frame correcto","Mayoría correcta"])
    lines += ["","## Transiciones y disponibilidad","",
        "Curvas de P(equipo del spell corto|vivo), posesiones delimitadas por H↔A en ambos extremos. "
        "Son perfiles descriptivos que pueden contener otros cambios próximos; no estiman latencia causal.",""]
    rows = []
    for fold,value in s["results"]["validation"].items():
        for arm,d in value["diagnosis"].items():
            curves = d["clean"]["curves"]
            for side in ("entry","exit"):
                values = {float(k.split("/")[-1]):v["probability"]["mean"] for k,v in curves.items() if k.startswith("HA_HA/"+side+"/")}
                rows.append([fold,arm,side,*[f"{values[x]:.4f}" for x in (-.5,0.,.5)]])
    lines += table(rows,["Fold","Brazo","Frontera","−0,5s","0s","+0,5s"])
    lines += ["", "Las curvas se generan localmente con `report-ball-attention` en "
        "`evaluation/reports/external/ball-attention-v1/figures/validation_transition_probabilities.png` "
        "y `.svg`. Esos artefactos permanecen ignorados; no se distribuyen con el código.","",
        "Disponibilidad XY no equivale a visibilidad visual. Distancias/movimiento se calculan solo para diagnóstico, "
        "sin introducirlos en el encoder ni usar GT como feature. Las consultas de B con balón ausente reproducen exactamente "
        "las seeds aprendidas en las ocultaciones de ambos folds; C permanece constante entre frames.","",
        "## Coste, controles y límites","",
        f'Coste acumulado registrado: **{s["compute"]["seconds"]:.2f}s**, incluidos **{s["training_seconds"]:.2f}s** de entrenamiento. '
        f'Actualizaciones de entrenamiento: **{sum(s["updates"].values())}**, seis modelos y120épocas. '
        "Excluye desarrollo/pruebas sintéticas/renderizado/alineación adicional de cohortes y no cuenta las pruebas cortas como épocas.","",
        "Parámetros finales: "+", ".join(f'{a} {n:,}' for a,n in s["parameters"].items())+". B/C tienen exactamente el mismo número; "
        "el control constante iguala parámetros, no la clase funcional dinámica.","",
        "Pasan24 pruebas:14 propias,7 de Fase7 y3 de Fase8. El control A pasa68 comprobaciones exactas sobre ventanas reales train: "
        "forward, pérdidas base/auxiliar, gradientes, AdamW, RNG y reanudación. Las salidas iniciales B/C coinciden exactamente; "
        "gradientes comunes difieren como máximo2,98e-8 por orden de suma FP32.","",
        f'La verificación final pasa **{s["verification_count"]} comprobaciones**, incluidos fuentes/ancestros, '
        "preservación histórica, mismos RNG/sampler/ventanas/actualizaciones entre brazos, probabilidades normalizadas, "
        "mismas ocultaciones, fallback y sello previo a test.","",
        "Los errores de clasificación, máscara viva y extensión son categorías operativas, no atribuciones causales puras. "
        "Ausencia de evidencia en emisiones no prueba ausencia de información latente. No se aislaron por ablationes "
        "las contribuciones del SAB, fallback o TCN; se probó una sola receta de consultas. "
        "Una semilla prefijada por fold y dos validaciones no constituyen confirmación independiente. "
        "Los tests IDSSE ya consultados no intervienen en selección ni autorizan búsqueda adicional.","",
        "Artefactos nuevos ignorados: `evaluation/reports/external/ball-attention-v1/`. "
        "No se accedió a materiales del desafío ni se leyeron logs. Se preservaron evaluador, fuentes/protocolos "
        "históricos y checkpoints existentes. CPU sigue como predeterminado; no se integra esta arquitectura en recetas principales.",""]
    costs = []
    for fold in (0,6):
        for arm in ("original","ball","capacity"):
            stages = [s["costs"][f"{fold}/{arm}/{stage}"] for stage in ("base","binary")]
            train_s = sum(r["training_seconds"] for stage in stages for r in stage["history"])
            wall_s = sum(stage["seconds"] for stage in stages)
            rss = max(stage["memory"]["process_peak_rss_bytes"] or 0 for stage in stages)
            costs.append([fold,arm,f"{train_s:.2f}",f"{wall_s:.2f}",f"{rss/1024**3:.3f}"])
    lines += table(costs,["Fold","Brazo","Entrenamiento s","Etapas con validación s","RSS máximo proceso GiB"])
    lines += ["","RSS es acumulado de un único proceso y caché de datos; no memoria aislada de cada modelo. "
        "El tiempo de etapas incluye evaluación de pérdida, checkpoints y preparación propia; el total incluye además selección/diagnóstico/inferencia.",""]
    plot(s)
    (ROOT/"docs/BALL_CONDITIONED_ATTENTION_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    (DEST/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    atomic_json(DEST/"presentation.json",{"script_sha256":file_hash(Path(__file__)),"summary_sha256":file_hash(DEST/"summary.json"),
        "report_sha256":file_hash(ROOT/"docs/BALL_CONDITIONED_ATTENTION_RESULTS.md"),
        "figure_sha256":file_hash(DEST/"figures/validation_transition_probabilities.png"),"new_training":0})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
