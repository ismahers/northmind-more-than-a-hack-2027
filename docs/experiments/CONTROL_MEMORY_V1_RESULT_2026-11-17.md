# Memoria de control v1: cierre sin promoción

17 de noviembre de 2026. [Protocolo previo](CONTROL_MEMORY_V1_2026-11-17.md),
[agregado revisado](../public-results/2026-11-17/control-memory-v1.json) y
[pruebas sintéticas](../public-results/2026-11-17/control-memory-tests.json).

**Decisión: no incorporar esta variante a la entrega.** La candidata
`northmind_sequence` del 16 de noviembre y sus fuentes permanecen intactas.
No se calculó una puntuación del partido del reto ni se accedió a su GT o vídeo.
No se ajustará otra variante después de observar este resultado.

## Resultado externo

Los dos folds fallan las condiciones fijadas antes de calcular. El objetivo
robusto combina spellF1 con pesos 0,50 limpio, 0,25 ocultación de 5 s y 0,25
ocultación de 15 s. No es un peor caso ni una métrica exclusiva de oclusiones.

| Fold | Limpio base → memoria | Robusto base → memoria | Delta robusto | Falsas H/A por minuto, limpio | Decisión |
|---|---:|---:|---:|---:|---|
| 0 | 0,52552 → 0,53564 | 0,41859 → 0,42483 | +0,00625 | 0,83782 → 0,86847 | Falla falsas H/A |
| 6 | 0,52964 → 0,52928 | 0,42860 → 0,42823 | -0,00037 | 1,16969 → 1,20131 | Falla robusto, limpio y falsas H/A |

En fold 0 mejoran las posesiones cortas limpias de 0,14085 a 0,15493; en fold 6
se mantiene 0,12903. Recall largo limpio no cambia. Esos indicadores no compensan
los fallos. Las guardas de clean, corto, largo, macroF1 y falsas H/A corresponden
al escenario **limpio**. Los otros escenarios contribuyen al objetivo robusto.

La mejora de fold 0 está dominada por el escenario limpio. En ocultación de
15 s, spellF1 apenas cambia de 0,36248 a 0,36259; no demuestra resolver las
oclusiones. Las métricas completas y los seis checks por fold están en el JSON.

## Qué se comprobó

- 39 pruebas sintéticas correctas, cero fallos, errores u omisiones.
- Cálculo externo, verificación del registro/sello y exportación revisada: OK.
- Paridad de identidad contra las emisiones neuronales históricas selladas
  en los seis escenarios de fold/ocultación: correcta.
- Dos modelos por rol de su propio fold; no ensemble entre folds sobre sus
  validaciones. No entrenamiento, barrido, arrays test/calibration o target.
- Referencia y variante comparten geometría, observaciones ocultadas, máscara
  D, suavizado base, modelos y decoder; cambia solo el delta de memoria.
- Las fuentes de inferencia y preparación de la entrega anterior no cambiaron.

Entorno: Windows, Python 3.13.17, intérprete existente del sprint. Cuatro hilos
y presupuesto máximo de 3.600 s; sin instalar dependencias. El registro local
se escribió antes de cargar etiquetas externas para puntuar; sus fuentes e
inputs se comprobaron al terminar. SHA-256 del registro:

```text
2fe3c7975448b5ba958dc902afd146e2fcf782b1fcf5fea65b8835200f304ff5
```

Hubo un intento abortado en setup porque dos pruebas sintéticas se añadieron
después de emitir el primer receipt. No produjo métricas. Se revisó el cambio,
se congelaron las fuentes, se repitieron las 39 pruebas y se ejecutó la misma
hipótesis, sin cambiar parámetros ni leer logs. No se sustituyó un registro
de resultado anterior ni se efectuó selección retrospectiva.

## Alcance y reproducción

Son validaciones IDSSE previamente consultadas, con anotaciones del proveedor;
no constituyen confirmación independiente ni resultado de BAR-ATH. Fuente:
[pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), revisión
`a715a38dfbaf5f58e431727c2b78d174101a703c`, CC-BY-4.0 según la configuración
de procedencia del proyecto. No se redistribuye tracking bruto.

Los priors Smart Tagging/reinicios son cero en ambos brazos porque IDSSE no
ofrece señales equivalentes. Este contraste **no reproduce toda la receta
target**. La proximidad y la velocidad baja son proxies, no anotaciones de
control. La velocidad por desplazamiento neto en 0,12 s puede subestimar
trayectorias curvas o de ida y vuelta. El extractor es causal; el suavizado
de la fusión es offline y usa contexto futuro dentro del tramo vivo.

Para reproducir hacen falta los mismos NPZ públicos externos, los modelos
y los registros/dependencias históricos, con sus bytes y hashes originales.
La cápsula pública de entrega no contiene necesariamente todo el contexto
externo de validación; no afirmar que permite repetir por sí sola este ensayo.
No se ha probado una restauración fría, otro host o Linux de este piloto.
No se crearon nuevos checkpoints ni cachés de inferencia necesarios para la
entrega; sus modelos publicados siguen siendo los mismos.

```text
python -m scripts.run_local test-control-memory
python -m scripts.run_local run-control-memory
python -m scripts.run_local verify-control-memory
python -m scripts.run_local export-control-memory
```

Usar el intérprete y las fuentes exactas del receipt. La ejecución completada
se verifica sin recalcular ni reemplazar métricas. No modificar los parámetros
o relanzar versiones históricas bajo su mismo registro. Los resultados y logs
originales permanecen en `evaluation/reports/`; el agente solo abrió el agregado
que produjo el escritor público revisado.

## Entrega al equipo

Base de trabajo: `339c63b`; los nuevos archivos se identifican por sus hashes,
no por un commit nuevo inexistente. [Handoff](../handoff/README.md) y cuerpos
de issues de Dani/Miguel listos. Git/GitHub bloquean publicación en esta sesión:
`.git` no escribible, CLI sin conexión y conector privado 404. No hay commit,
push, PR o issues nuevos publicados que declarar. El trabajo no está disponible
para los compañeros en GitHub hasta resolver ese acceso y publicarlo.
