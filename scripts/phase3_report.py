"""Readable tables from completed public-external study rows; no model selection."""

from pathlib import Path

import numpy as np


def table(lines, headers, rows):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    lines.append("")


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def write_report(destination, rows, summary, folds, protocol):
    lines = ["# Fase 3 — resultados medidos con datos externos públicos", "",
        f"Filas evaluadas: {len(rows)}. Solo se muestran unidades terminadas.", "",
        "Cada fold retiene un IDSSE para test, otro para validación interna y cinco para entrenamiento. "
        "B añade los veinte SkillCorner con supervisión parcial de equipo. Ninguna puntuación test elige "
        "presupuesto, checkpoint, condición ni parámetros de decodificación de su propio fold.", "",
        "A_clean = IDSSE limpio; A_blocks = IDSSE con ocultación sintética; B_clean/B_blocks añaden SkillCorner. "
        "B mantiene las actualizaciones de A, con la mitad de ventanas de cada fuente. "
        "La desviación entre partidos usa ddof=1. Los oráculos son únicamente diagnósticos.", "",
        "## Referencia y pipeline seleccionado en validación", ""]
    table(lines, ["Fold / test", "E0 2ép", "A entrenado argmax", "Condición / decoder", "spellF1 limpio",
                  "macroF1", "40% / 5s", "40% / 15s"],
        [[str(f["fold"]) + " / " + f["test_match"].split("-")[-1], fmt(f.get("E0_2epochs_spellF1")),
          fmt(f.get("trained_argmax_spellF1")), f.get("condition", "—") + " / " + f.get("decoder", "—"),
          fmt(f.get("clean_selected_spellF1")), fmt(f.get("macroF1")),
          fmt(f.get("missing40_5s_selected_spellF1")), fmt(f.get("missing40_15s_selected_spellF1"))] for f in folds])
    lines.extend(["## Entrenamiento: decisión interna, anterior al test", ""])
    table(lines, ["Fold", "Cap elegido", "Época checkpoint", "Épocas ejecutadas", "Val spellF1 2 / 6 / 12", "Tiempo s"],
        [[s["fold"], s["chosen"]["budget"], s["chosen"]["selected_epoch"], s["epochs_executed"],
          " / ".join(fmt(next((c["validation_spellF1"] for c in s["caps"] if c["budget"] == cap), None))
                     for cap in protocol["epoch_caps"]), fmt(s["total_s"])] for s in summary["training_schedules"]])
    lines.extend(["## Todas las comparaciones limpias: mismas emisiones por condición", "",
        "Cada familia usa sus parámetros elegidos por validación interna; no son máximos sobre test.", ""])
    clean = [r for r in rows if r["seed"] == 2027 and r["scenario"] == "clean"]
    configurations = sorted(set((r["condition"], r["decoder"]) for r in clean))
    table(lines, ["Condición / decoder"] + ["Fold " + str(i) for i in range(7)] + ["Media ± SD"],
        [[condition + " / " + decoder] + [fmt(next((r["spellF1"] for r in clean if r["fold"] == i
            and r["condition"] == condition and r["decoder"] == decoder), None)) for i in range(7)] +
         [fmt(np.mean(v)) + " ± " + (fmt(np.std(v, ddof=1)) if len(v) > 1 else "—")]
         for condition, decoder in configurations
         for v in [[r["spellF1"] for r in clean if r["condition"] == condition and r["decoder"] == decoder]]])
    lines.extend(["## Robustez: todas las condiciones y decoders", "",
        "spellF1 sobre la secuencia completa; las etiquetas no cambian al ocultar coordenadas. "
        "La fracción efectiva de cada máscara se conserva en all_rows.csv.", ""])
    table(lines, ["Condición", "Escenario", "Decoder", "Folds", "spellF1 media ± SD", "macroF1 media ± SD"],
        [[r["condition"], r["scenario"], r["decoder"], r["folds"],
          fmt(r["spellF1_mean"]) + " ± " + fmt(r["spellF1_sample_SD"]),
          fmt(r["macroF1_mean"]) + " ± " + fmt(r.get("macroF1_sample_SD"))] for r in summary["configurations"]])
    lines.extend(["## Errores del pipeline elegido por validación: test limpio", ""])
    table(lines, ["Fold", "Reales / predichas", "Fragmentadas", "Fusiones", "D falsas internas", "MAE inicio s", "MAE final s"],
        [[f["fold"], str(f.get("gt_spells", "—")) + " / " + str(f.get("pred_spells", "—")),
          f.get("fragmented_gt", "—"), f.get("fused_predictions", "—"), f.get("false_D_interruptions", "—"),
          fmt(f.get("boundary_start_MAE_s")), fmt(f.get("boundary_end_MAE_s"))] for f in folds])
    lines.extend(["MAE de extremos: solo posesiones emparejadas a IoU ≥ 0.5; el número emparejado está en las filas JSON. "
        "Fragmentaciones y fusiones cuentan solapes múltiples del mismo equipo, no atribuciones causales exclusivas.", "",
        "## Oráculos y cabezas: análisis sobre test limpio", ""])
    diagnostic_rows = [r for r in clean if r.get("oracles_analysis_only") and (
        r["validation_selected_pipeline"] or (r["condition"] == "A_clean" and r["decoder"] == "argmax"))]
    table(lines, ["Fold", "Condición / decoder", "spellF1", "D verdadero + equipo pred.",
                  "Equipo verdadero + D pred.", "F1 cabeza vivo", "Exactitud equipo en GT vivo"],
        [[r["fold"], r["condition"] + " / " + r["decoder"], fmt(r["spellF1"]),
          fmt(r["oracles_analysis_only"]["true_D_predicted_conditional_team"]["spell_f1_mean"]),
          fmt(r["oracles_analysis_only"]["true_team_where_defined_predicted_D"]["spell_f1_mean"]),
          fmt(r["head_alive_F1_at_0_5"]), fmt(r["conditional_team_accuracy_gt_alive"])] for r in diagnostic_rows])
    lines.extend(["Durante D real el equipo GT no está definido: el oráculo de equipo deja sin corregir esos falsos vivos. "
        "El oráculo de D usa el equipo condicional sin suavizar y por ello no aísla perfectamente un decoder temporal.", "",
        "## Efectos emparejados: argmax y siete partidos idénticos", ""])
    table(lines, ["Control → experimento", "Escenario", "Δ spellF1", "IC exploratorio 95% por partidos"],
        [[r["control"] + " → " + r["experiment"], r["scenario"], fmt(r["mean_delta_spellF1"]),
          " / ".join(fmt(v) for v in r["paired_match_95CI"])] for r in summary["paired_effects_argmax"]])
    lines.extend(["## Efectos emparejados de decodificación: test limpio", "",
        "Se mantienen los modelos y emisiones; cada decoder conserva su selección interna. "
        "El JSON incluye además los seis escenarios de ausencia.", ""])
    table(lines, ["Condición", "Control → decoder", "Δ spellF1", "IC exploratorio 95% por partidos"],
        [[r["condition"], r["control"] + " → " + r["experiment"], fmt(r["mean_delta_spellF1"]),
          " / ".join(fmt(v) for v in r["paired_match_95CI"])]
         for r in summary["paired_effects_decoders"] if r["scenario"] == "clean"])
    lines.extend(["## Variación de semillas del pipeline elegido internamente", ""])
    table(lines, ["Escenario", "Filas", "Medias por semilla", "Media total", "SD medias semillas", "SD partidos semilla 2027"],
        [[r["scenario"], r["rows"], " / ".join(fmt(v) for v in r["seed_means"]), fmt(r["mean_spellF1"]),
          fmt(r["SD_across_seed_means"]), fmt(r.get("seed_2027_match_sample_SD"))] for r in summary["validation_selected_pipeline"]])
    lines.extend(["Las medias de semillas no son 21 partidos independientes. Los intervalos de siete partidos, "
        "con equipos repetidos, no establecen por sí solos significación ni rendimiento en el partido del reto.", "",
        "## Archivos", "", "- `all_rows.csv`: cada fold, condición, escenario, decoder y semilla, con IoU y errores.",
        "- `all_rows.json`: lo anterior más oráculos anidados.",
        "- `summary.json`: agregados, incertidumbre, horarios y decisiones de validación.",
        "- `selections/`: búsqueda interna completa, congelada antes del test.",
        "- `training/`, `validation_predictions/`, `test/`: checkpoints y probabilidades reutilizables.", ""])
    Path(destination, "results.md").write_text("\n".join(lines), encoding="utf-8")
