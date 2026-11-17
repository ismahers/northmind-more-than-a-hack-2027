# Memoria de control: protocolo fijo del 17 de noviembre

Hipótesis única: la proximidad durante un vuelo o una oclusión no demuestra
un cambio de posesión. Una memoria breve de control geométrico sostenido puede
reducir cambios falsos. No se copia código, pesos ni correcciones por frame del
participante externo. No se cambia la candidata de entrega del 16 de noviembre.

## Regla y contraste

`configs/control_memory_validation_v1.json` fija los parámetros antes de medir.
Control exige distancia <=1,8 m, velocidad <=5,5 m/s y ventaja geométrica >=0,75 m
sostenidas 0,20 s. La velocidad se estima causalmente sobre 0,12 s. La memoria
decae linealmente durante un máximo de dos segundos; un control rival sostenido
la transfiere. Se reinicia en D/X, cambio de segmento o hueco de reloj. Ausencia
de observación no crea ni refresca control.

La intervención suma 0,15 veces esta memoria exclusivamente cuando el módulo
de la evidencia base es menor que 0,25. Baseline y candidata comparten modelos,
geometría, máscaras D, parámetros, suavizado y decodificación. No se corrige a
la vez el suavizado que atraviesa D, ni se retocan costes u otros parámetros.
El delta de memoria se suaviza por separado dentro de cada tramo vivo predicho,
para que la intervención añadida no atraviese D. El suavizado de la evidencia
base y de la referencia geométrica permanece idéntico en los dos brazos.

Se usan solo train y validation de los folds IDSSE 0/6 para sus roles declarados;
no se abre calibration, test ni el partido del reto. El piloto no entrena modelos.
La inferencia utiliza cada modelo solo en la validación de su propio fold.
Se reconstruyen las emisiones de cuatro vistas y se exige paridad de identidad
con la referencia sellada. La geometría se expresa a 25 Hz por segmento y las
oclusiones artificiales se aplican también a sus observaciones.

IDSSE no contiene los mismos Smart Tags. Por ello los priors tácticos son cero
en ambos brazos. Este es un contraste externo con priors neutros, **no una
validación completa de la receta de entrega**. Las validaciones han sido
consultadas en estudios anteriores; no constituyen una prueba independiente
nueva ni garantizan mejora en BAR-ATH.

## Decisión previa y presupuesto

Una sola variante, sin barrido ni nuevo ajuste después de observar métricas.
En cada fold: ganancia robusta mínima 0,005; spellF1 limpio y recall corto no
disminuyen; recall largo admite -0,01 y macroF1 -0,005; las falsas transiciones
H/A limpias no aumentan. Objetivo robusto: 0,50 limpio +0,25 ocultación de 5 s
+0,25 ocultación de 15 s. Fallar cualquier criterio impide promover la variante.
Pasarlos tampoco autoriza promoción automática: requiere revisar transferencia,
reproducibilidad y tiempo suficiente para publicar y verificar la entrega.

Máximo de cálculo: 3.600 s, cuatro hilos. El registro congela código, protocolo,
modelos y entradas públicas antes de medir. El resultado se sella después.
No se sustituyen registros ni resultados históricos. No se compara salida con
vídeo, no se entrena/calibra con GT o vídeo del reto y no se suben sus datos a IA.

## Ejecución y entrega

Las tareas nuevas revisadas son `test-control-memory`, `run-control-memory`,
`verify-control-memory` y `export-control-memory`, mediante `scripts.run_local`.
El lanzador captura stdout/stderr desde la creación del hijo y solo devuelve
OK/FAILED. No se leen sus logs, tampoco si una tarea falla.

Resultados locales: `evaluation/reports/external/control-memory-validation-v1/`.
El escritor revisado solo exporta agregados de IDSSE y comprobaciones de ámbito
a `docs/public-results/2026-11-17/control-memory-v1.json`; nunca arrays,
predicciones por frame o datos del target. La candidata actual permanece en
`evaluation/reports/final-delivery/Northmind.csv`.

Estado final: 39 pruebas sintéticas, cálculo, verificación y exportación OK.
Ambos folds fallan los gates: [cierre sin promoción](CONTROL_MEMORY_V1_RESULT_2026-11-17.md).
No se modifica la entrega actual ni se ajusta otra variante. Los accesos
Git/GitHub fallan; no afirmar publicación compartida hasta confirmar
commit/push/PR e issues en el fork privado.
