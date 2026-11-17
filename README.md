# Northmind: reproducción de la entrega

17 de noviembre de 2026. **Documento técnico final: [Northmind.pdf](Northmind.pdf)**
(inglés, cuatro páginas A4). Fuente editable: [Northmind.md](Northmind.md).
La receta seleccionada sigue siendo `northmind_sequence`; los pilotos del
17 de noviembre son experimentales y no están seleccionados. Véase
[EXPERIMENTAL.md](EXPERIMENTAL.md).

Este paquete contiene las fuentes de la receta de inferencia BAR–ATH. La
publicación está en [ismahers/northmind-more-than-a-hack-2027](https://github.com/ismahers/northmind-more-than-a-hack-2027),
un repositorio nuevo con historia limpia. El fork de investigación permanece
separado. Se verificaron el acceso anónimo al código y al ZIP de modelos,
así como el SHA-256 del ZIP descargado públicamente.

## Preparación

1. Usar Python **3.13.17**, un entorno virtual nuevo y las versiones fijadas
   en `requirements-submission.txt`. La referencia de inferencia usa NumPy
   **2.2.6**, pandas **2.2.3**, python-dateutil **3.0.0** y PyTorch
   **2.10.0+cpu**. En Windows conviene clonar en una raíz corta, como `C:/nm`,
   o habilitar soporte de rutas largas. Se comprobó un entorno nuevo aislado
   en el mismo host Windows; véanse los detalles de instalación más abajo.
2. Obtener legalmente el paquete oficial de datos. Descomprimirlo localmente
   en `data/` o indicar su raíz mediante `MTAH_DATA_DIR`, conservando la
   estructura esperada por `games.py`. Se necesitan ATD y Smart Tagging.
   Los datos no se distribuyen en este repositorio.
3. Descargar [la cápsula de modelos](https://github.com/ismahers/northmind-submission-2027/releases/download/submission-v1/northmind-public-experiments-2026-10-28.zip)
   y guardar el ZIP, sin descomprimir, como
   `evaluation/reports/public-assets/northmind-public-experiments-2026-10-28.zip`.
   La Release es pública y su descarga anónima fue comprobada. El restaurador
   comprueba SHA-256 del ZIP y de cada archivo, y preserva registros,
   selecciones y sellos originales.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-submission.txt
.venv\Scripts\python -m scripts.run_local restore-submission-assets
.venv\Scripts\python -m scripts.run_local prepare-final-submission
```

Las dos últimas órdenes restauran los artefactos públicos y generan la
entrega. El comando `pip install -r` completo falló en el host de verificación;
no se afirma que esa única orden se haya probado con éxito. El entorno nuevo
se completó con ruedas oficiales de PyPI, verificadas mediante sus SHA-256,
de `numba==0.61.2`, `llvmlite==0.44.0` y `certifi==2025.10.5`, usando
`pip --isolated install --no-index --no-deps` sobre los archivos descargados.
Las dos primeras ruedas corresponden a `cp313-cp313-win_amd64`; certifi usa
`py3-none-any`. `pip check` terminó correctamente.

`scripts/clean_submission_install.py` implementa esta recuperación sin leer
logs: `complete_public_wheels()` obtiene las ruedas numba/llvmlite y
`complete_certifi_public_wheel()` obtiene certifi, comprobando los hashes del
índice PyPI antes de instalar. Estas funciones completan exclusivamente el
entorno independiente `evaluation/reports/final-clean-venv-v2`, creado por
`clean-submission-install`; no modifican automáticamente `.venv`. Para ese
flujo están registradas `complete-submission-public-wheels` y
`complete-submission-certifi` en `scripts.run_local`. La lista final fija
también las dependencias transitivas del entorno comprobado.

## Contrato de salida y comprobación

La candidata final es `evaluation/reports/final-delivery/Northmind.csv`
(`northmind_sequence`); la reserva es `Northmind-fallback.csv` en la misma
carpeta. `manifest.json` conserva configuración, procedencia y comprobaciones
privadas. El archivo para el formulario se llama **Northmind.csv**.
Debe contener exactamente `frame,label`,
199.319 filas, frames 0–199318 consecutivos y etiquetas H/A/D/X, sin celdas
vacías ni columna de índice.

La entrada final debe generar sin GT, validar el CSV mediante el contrato
oficial y capturar stdout/stderr desde la creación de cada proceso. La consola
solo muestra OK/FAILED. Los logs, manifiestos del target, cachés target y
predicciones se mantienen fuera de Git y no deben abrirse mediante agentes.

Antes de declarar reproducción exacta, ejecutar de nuevo desde las fuentes,
los modelos y el paquete oficial, y comparar **bytes completos** del CSV.
Un hash calculado tras normalizar saltos de línea no basta para esta garantía.
Validar la copia exacta que vaya a subirse. El validador oficial no requiere
GT, pero imprime estadísticas y errores: conservar también esa salida local.

Comprobación local del 16 de noviembre: generación final OK, candidato y
reserva válidos, con dos generaciones independientes desde ATD sin reutilizar
la caché de features y SHA-256 coincidentes. Restauración de 66 archivos con
hashes verificados y ocho pruebas de portabilidad correctas. Esto no acredita
una puntuación BAR–ATH. Una copia de fuentes y modelos en otra raíz Windows,
sin caché target preexistente y con un entorno nuevo aislado Python 3.13.17,
completó dos generaciones válidas y produjo CSV candidato/reserva idénticos
byte a byte a los originales. Esto acredita reproducción con entorno nuevo
y raíz nueva en el mismo host Windows, mediante el procedimiento de instalación
descrito. No se comprobó una ejecución Linux ni otro host físico.

## Artefactos y atribución

La receta utiliza ocho checkpoints: dos profesores, semillas 2027/2028, más
cabezas local y multiescala en cada fold 0/6. Necesita además registro,
selecciones, sello scale-selective y sus dependencias registradas.
La caché del partido objetivo se vuelve a generar; no se publica.

La cápsula histórica de partida se denomina
`northmind-public-experiments-2026-10-28.zip`, tiene 35.910.601 bytes y SHA-256:

```text
f09c894ad962364a2a16357749d9de304552532db58bb047f6fd2f9a1ca25162
```

Su manifiesto público registra 66 archivos, restaurados y verificados. La
[Release submission-v1](https://github.com/ismahers/northmind-submission-2027/releases/tag/submission-v1)
incluye el paquete y permite descargarlo sin autenticación. El manifiesto final
fija su URL y SHA-256; los artefactos permitieron regenerar ambos CSV.

Los modelos derivan de [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data),
revisión `a715a38dfbaf5f58e431727c2b78d174101a703c`, CC-BY-4.0 según
`configs/external_sources.json`. Se conserva la atribución del proveedor.
La licencia MIT de `LICENSE` cubre el código del repositorio, no los assets
del partido. No se redistribuyen datos BAR–ATH ni sus derivados restringidos.

## Publicación y límites

`configs/submission_source_allowlist.json` define los archivos de fuente
permitidos para construir la copia limpia. La copia incluye fuentes auxiliares
porque los sellos históricos verifican sus hashes, aunque no se ejecuten al
generar la entrega. No incluye la historia Git ni resultados de investigación.

Desde el repositorio de trabajo, `python -m scripts.run_local build-submission-source`
prepara `_pub/` para revisar y publicar
por separado. La carpeta debe estar ausente o vacía: el constructor no borra
una entrega anterior. `refresh-submission-source` actualiza exclusivamente las
fuentes permitidas y los manifiestos de esa copia; conserva los demás archivos
sin leerlos ni borrarlos. `SOURCE-MANIFEST.json` registra los bytes normalizados
a LF; `PUBLIC-MODEL-MANIFEST.json` identifica el ZIP y su atribución.
`PUBLIC-DELIVERY-MANIFEST.json` registra por separado los bytes exactos y el
SHA-256 del PDF final incluido en la raíz; no aplica normalización de texto
al binario. Este constructor no publica ni inicializa un repositorio.

Se comprobaron restauración, generación sin GT, igualdad binaria, validez del
CSV y acceso público al código y a los modelos. Mantener el CSV y los datos
restringidos fuera del repositorio público. El envío del formulario sigue
siendo una acción separada; este documento no afirma que se haya enviado.

English technical report: [download the final Northmind.pdf](https://github.com/ismahers/northmind-more-than-a-hack-2027/raw/refs/heads/main/Northmind.pdf). Editable English source: [Northmind.md](Northmind.md); PDF build instructions: [PDF-BUILD.md](PDF-BUILD.md). The PDF in the historical `submission-v1` Release remains a historical artifact.
