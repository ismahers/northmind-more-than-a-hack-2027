# Balón vivo neuronal v1: mejora del objetivo, sin promoción automática

17 de noviembre de 2026. [Protocolo previo](LIVE_BALL_V1_2026-11-17.md),
[agregado externo revisado](../public-results/2026-11-17/live-ball-v1.json).
La candidata oficial del 16 de noviembre se conserva. Se preparó una alternativa
experimental separada, sin seleccionar, sin evaluar el target y sin retocar la
receta después de observar métricas.

## Comparación externa

| Fold | SpellF1 limpio base → variante | Objetivo robusto base → variante | Delta robusto | Guardas |
|---|---:|---:|---:|---|
| 0 | 0,52552 → 0,53077 | 0,41859 → 0,47487 | +0,05628 | Falla recall corto limpio |
| 6 | 0,52964 → 0,57842 | 0,42860 → 0,52002 | +0,09142 | Pasa todas |

El objetivo robusto es 0,50 limpio +0,25 ocultación de 5 s +0,25 ocultación de
15 s; no es un peor caso ni un indicador exclusivo de oclusiones. SpellF1
mejora en los seis casos fold/escenario, pero **no pasan todas las guardas**.

| Recall de posesiones cortas, matching IoU 0,5 | Base | Variante |
|---|---:|---:|
| Fold 0, limpio | 20/142 | 19/142 |
| Fold 0, ocultación 5 s | 14/142 | 9/142 |
| Fold 0, ocultación 15 s | 12/142 | 13/142 |
| Fold 6, limpio | 20/155 | 22/155 |
| Fold 6, ocultación 5 s | 12/155 | 11/155 |
| Fold 6, ocultación 15 s | 13/155 | 12/155 |

Las falsas transiciones H/A limpias disminuyen de 0,83782 a 0,73564/min y de
1,16969 a 0,98001/min, respectivamente. Aumentan en algunos escenarios ocultos:
fold 0 en ambos, y fold 6 con ocultación de 15 s. Las guardas secundarias del
protocolo se aplican solo al escenario limpio; no ocultar estos otros cambios.
El fallo de la guarda limpia impide seleccionar automáticamente la alternativa.

La mejora global externa es prometedora, pero el intercambio no se reduce a
una sola posesión: bajo oclusión también empeora la recuperación de cortas.
No se recalibran umbrales ni se relajan las guardas después de verlo.

## Comprobaciones y alcance

- 39 pruebas sintéticas del prior, decoder y validación: correctas.
- Cálculo, verificación y exportación externa: OK.
- Paridad de las emisiones de identidad en los seis casos: correcta.
- Baseline igual al ensayo anterior en las cinco métricas de cada caso,
  tolerancia absoluta 1e-10. La comparación cambia la máscara D, sin memoria.
- Priors binarios ajustados solo con train externo de cada fold; modelos
  neuronales y parámetros congelados, sin entrenamiento neural nuevo.
- Registro SHA-256:
  `82a866b73e50155cab3413277c4bec86151617cc510fce847be90f9507254d88`.

Validaciones IDSSE previamente consultadas, cada modelo en su propio fold y
con priors tácticos neutros. No evaluación independiente ni reproducción de
todo el ensemble con Smart Tagging del target. No se accedió a arrays de
calibration/test o a GT/vídeo target. Fuente pública IDSSE fijada en
`configs/external_sources.json`, revisión
`a715a38dfbaf5f58e431727c2b78d174101a703c`, declarada CC-BY-4.0; atribución
a pysport/idsse-data. Se exportan métricas y matrices 2x2 propias, no tracking.

## Candidata experimental concreta

Archivo privado: `evaluation/reports/live-ball-candidate-v1/Northmind-live-ball-v1.csv`.
La entrega oficial `evaluation/reports/final-delivery/Northmind.csv` no se ha
sustituido. El nuevo archivo no está seleccionado y no tiene puntuación target.

25 pruebas sintéticas del preparador y adaptador correctas. Dos generaciones
desde ATD sin caché geométrica producen bytes idénticos y cumplen el contrato
oficial. El baseline reconstruido coincide exactamente con el CSV original.
La recuperación de anclas vivas 10→25 Hz exige igualdad exacta con el caché.
Preparación y verificación final: OK; estados públicos sin hashes ni datos
target en `live-ball-candidate-status.json`.

El prior de inferencia es la media de probabilidades de transición de folds
0/6 fijada antes de evaluar, guardada en `configs/live_ball_prior_v1.json`.
Se utiliza el caché target de probabilidades float32; el piloto externo
promedió en float64. La igualdad del round trip acredita reconstrucción del
caché, no equivalencia de precisión o rendimiento entre ambas evaluaciones.
No se ha probado aún esta alternativa desde otra raíz o entorno nuevo.

La escritura atómica usa rename sin sustitución en Windows y hard link en
otros sistemas. Una prueba sintética específica verifica que una carrera de
creación no sobrescribe el archivo de otro escritor. No se leen logs ante fallos.

```text
python -m scripts.run_local test-live-ball-candidate
python -m scripts.run_local prepare-live-ball-candidate
python -m scripts.run_local verify-live-ball-candidate
```

## Documento de entrega

El PDF en inglés sigue describiendo la receta oficial `northmind_sequence`.
Se precisó que la referencia puede recurrir a D cuando una ventana no aporta
evidencia de equipo y se separaron los ensayos externos de la receta presentada.
PDF regenerado: cuatro páginas A4, texto seleccionable, tres enlaces y revisión
visual de todas las páginas. No se atribuyen al CSV oficial las mejoras de la
alternativa. El nuevo PDF se ofrece por separado; no se afirma haber reemplazado
el asset histórico de la Release.
