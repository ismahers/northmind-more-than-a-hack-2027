# Northmind — documento técnico

16 de noviembre de 2026 · More Than a Hack 2027

## Enfoque

Inferimos la posesión por frame del partido BAR–ATH mediante una combinación
de geometría, señales tácticas y un modelo aprendido en datos externos.
La salida tiene 199.319 filas con índice de vídeo y una etiqueta H, A, D o X.
Las etiquetas finales se generan por código; no se corrigen manualmente.

La geometría compara las distancias entre balón y jugadores de ambos equipos.
La confianza disminuye cuando el balón describe un vuelo rápido y recto: estar
cerca de su trayectoria no demuestra control. Smart Tagging aporta un prior
de equipo y señales de reinicios; las reglas geométricas identifican paradas.

La evidencia aprendida procede de Set Transformer y cabezas temporales TCN.
La transferencia combina dos folds externos, 0 y 6, con profesores de semillas
2027 y 2028 y cabezas local y multiescala: ocho checkpoints en total.
Los parámetros de fusión scale-selective se recuperan de las selecciones
externas registradas y selladas, sin reajustarlos al partido objetivo.
Cada miembro se evalúa en cuatro vistas: identidad, intercambio de equipos,
reflexión espacial y combinación de ambas. Tras invertir las transformaciones,
se promedian las evidencias; los dos folds tienen el mismo peso.

El decodificador combina esa señal con geometría y priors tácticos. La
ponderación neural es 0,25 con observación y 0,45 sin ella. Una programación
dinámica penaliza los cambios H/A, con coste adaptativo a la fuerza de la
evidencia. Un filtro acotado elimina oscilaciones locales y alinea transiciones
con la geometría. Se procesan por separado las mitades y los tramos de juego.
La candidata final es `northmind_sequence`; el recibo local conserva su receta.

## Assets, entrenamiento y licencias

| Fuente | Uso |
|---|---|
| Tracking automático ATD del paquete BAR–ATH | Posiciones de balón/jugadores, alineación temporal y señales geométricas |
| Smart Tagging automático oficial BAR–ATH | Flags automáticos para prior de equipo y reinicios; no etiquetas humanas de posesión |
| IDSSE público, con anotaciones del proveedor | Entrenamiento y selección de los modelos externos; no son etiquetas BAR–ATH |
| Cinco minutos de GT público BAR–ATH | Evaluación diagnóstica histórica posterior a la generación, sin barrido de parámetros |

Los checkpoints derivan de [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data),
revisión `a715a38dfbaf5f58e431727c2b78d174101a703c`, declarada CC-BY-4.0 en
`configs/external_sources.json`. Se conserva esta atribución en el manifiesto
de artefactos. El código del repositorio mantiene su licencia MIT; esa licencia
no sustituye las condiciones de los datos ni las atribuciones de los modelos.

No se utiliza el GT privado ni otra fuente de posesión corregida de BAR–ATH.
No se crean etiquetas humanas de BAR–ATH para producir el CSV.
Vídeo, homografía y pose no forman parte de la inferencia final. La auditoría
SkillCorner no produjo una cohorte H/A/D validada que se incorporase a esta
receta. Los pilotos de coordinación de pases y continuidad de actor no
acreditaron una mejora suficiente y no se añaden a la entrega.
Los assets del partido y sus derivados restringidos permanecen locales.

## Reglas de posesión

- **H/A:** equipo local/visitante que controla el balón, según evidencia
  geométrica, táctica y temporal. La proximidad aislada no determina un cambio.
- **Balones sueltos, vuelos y disputas:** se reduce la confianza de la evidencia
  local y se favorece continuidad hasta reunir evidencia de cambio. El sistema
  no clasifica de forma independiente cada tipo de acción futbolística.
- **D:** parada apoyada por la geometría, incluidas situaciones de balón fuera
  y estacionario sin jugador próximo. Los reinicios rápidos del tagging
  protegen frente a falsas paradas geométricas.
- **X:** frames fuera de los bloques de vídeo correspondientes al juego. No se
  usa el GT para construir esta máscara.
- **Variantes:** `northmind_sequence` conserva la máscara D de la referencia
  geométrica y táctica. `northmind_sequence_bridged` permite unir huecos D de
  hasta 125 frames, cinco segundos a 25 fps, cuando ambos lados corresponden
  al mismo equipo y no existe protección geométrica de parada. La unión se
  aplica dentro de cada bloque, antes y después de decodificar.

Los parámetros efectivos están en `possession/inference.py` y
`possession/sequence_decoding.py`; no se infieren del CSV entregado.

## Evidencia y limitaciones

La referencia de test externo IDSSE publicada obtuvo **spellF1 0,51937 clean**
y **objetivo robusto 0,47719**. Son resultados externos de la referencia
registrada, no una puntuación BAR–ATH ni una evaluación independiente de la
transferencia final de cuatro vistas. Esta última mejoró ciertos comparadores
de validación, sin superar todos los criterios en ambos folds.

No se afirma aquí una mejora cuantificada de la candidata final en BAR–ATH.
La verificación de formato, procedencia y reproducción no acredita precisión.
La cobertura pública del partido es limitada; ajustarse repetidamente a ella
arriesga sobreajuste. Oclusiones, errores de tracking y tagging, disputas y
reinicios ambiguos pueden producir errores. La continuidad puede borrar
posesiones cortas; la variante con unión puede absorber paradas reales.

El cambio de dominio IDSSE → BAR–ATH continúa siendo una limitación. Las
cuatro vistas imponen consistencia frente a transformaciones; no prueban
comprensión de pases o control. Con más tiempo priorizaríamos evaluación
independiente y diversidad de partidos antes de añadir nuevas reglas.

## Reproducción

Seguir `README.md` del repositorio de entrega: instalar el entorno fijado,
restaurar la cápsula pública de modelos comprobando su manifiesto y SHA-256,
y preparar el paquete oficial de datos localmente. Después ejecutar:

```text
python -m scripts.run_local restore-submission-assets
python -m scripts.run_local prepare-final-submission
```

La entrada final genera `evaluation/reports/final-delivery/Northmind.csv` sin
cargar GT y devuelve únicamente OK/FAILED. `manifest.json`, en la misma
carpeta, identifica variante, configuración, procedencia y comprobaciones;
permanece privado. Se conserva además `Northmind-fallback.csv` como reserva.
La comprobación local del 16 de noviembre terminó OK: dos generaciones desde
ATD sin reutilizar caché de features produjeron bytes con SHA-256 coincidentes
y CSV válidos. Se verificaron los 66 archivos de la cápsula y ocho pruebas de
portabilidad. Una copia limpia de fuentes y modelos en otra raíz Windows,
sin caché target preexistente, repitió la inferencia completa: dos generaciones
válidas y CSV candidato/reserva idénticos byte a byte a los de la raíz original.
Después se repitió con un entorno nuevo aislado Python 3.13.17 en la nueva
raíz del mismo host Windows: dos generaciones válidas y ambos CSV idénticos
byte a byte a los originales. La instalación requirió completar ruedas
oficiales PyPI verificadas de numba, llvmlite y certifi después de fallar la
orden completa `pip install -r`; `pip check` terminó correctamente. No se
afirma que esa única orden funcionase, ni una ejecución Linux u otro host.
Los logs y predicciones no se publican.

Las cápsulas históricas contienen rutas absolutas Windows y fuentes selladas.
Su restauración resuelve las rutas sin alterar los registros ni aceptar
cambios de contenido. La prueba en una nueva raíz verifica ese traslado en
Windows, también con un entorno nuevo mediante el procedimiento documentado.
Otro sistema u otro host requiere comprobaciones adicionales. El
[repositorio público](https://github.com/ismahers/northmind-submission-2027)
y la [Release de modelos](https://github.com/ismahers/northmind-submission-2027/releases/tag/submission-v1)
se verificaron mediante acceso anónimo y comprobación SHA-256 del ZIP.
