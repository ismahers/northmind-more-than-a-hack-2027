# Northmind: reproducción de la entrega

16 de noviembre de 2026. Documento técnico: [Northmind.md](Northmind.md).

Este paquete contiene las fuentes de la receta de inferencia BAR–ATH. La
publicación prevista es `ismahers/northmind-submission-2027`, en un repositorio
nuevo con historia limpia. El fork de investigación permanece separado.
La URL solo será válida para la entrega cuando el repositorio y sus artefactos
sean accesibles públicamente y hayan sido revisados.

## Preparación

1. Usar Python **3.13.17**, un entorno virtual nuevo y las versiones fijadas
   en `requirements-submission.txt`. La referencia de inferencia usa NumPy
   **2.2.6**, pandas **2.2.3**, python-dateutil **3.0.0** y PyTorch
   **2.10.0+cpu**. La instalación fría debe comprobarse;
   el éxito del entorno histórico no la garantiza.
2. Obtener legalmente el paquete oficial de datos. Descomprimirlo localmente
   en `data/` o indicar su raíz mediante `MTAH_DATA_DIR`, conservando la
   estructura esperada por `games.py`. Se necesitan ATD y Smart Tagging.
   Los datos no se distribuyen en este repositorio.
3. Descargar [la cápsula de modelos](https://github.com/ismahers/northmind-submission-2027/releases/download/submission-v1/northmind-public-experiments-2026-10-28.zip)
   y guardar el ZIP, sin descomprimir, como
   `evaluation/reports/public-assets/northmind-public-experiments-2026-10-28.zip`.
   La Release debe existir y ser pública antes de enviar. El restaurador
   comprueba SHA-256 del ZIP y de cada archivo, y preserva registros,
   selecciones y sellos originales.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-submission.txt
.venv\Scripts\python -m scripts.run_local restore-submission-assets
.venv\Scripts\python -m scripts.run_local prepare-final-submission
```

Las dos últimas órdenes restauran los artefactos públicos y generan la
entrega. Antes de publicar, comprobar que ambas terminan correctamente desde
una copia limpia. No se debe dar por terminada una entrega que aún requiera
recuperar archivos solo disponibles en el ordenador de sus autores.

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
una puntuación BAR–ATH. La disponibilidad de wheels de las cinco dependencias
se comprobó para Windows/Linux mediante metadatos; no equivale a una
instalación fría comprobada. Una copia de fuentes y modelos en otra raíz
Windows, sin caché target preexistente, completó dos generaciones válidas y
produjo CSV candidato/reserva idénticos byte a byte a los originales. Se usó
el mismo entorno Python 3.13.17: esta prueba verifica portabilidad de raíz,
sin acreditar una instalación fría o ejecución Linux.

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

Su manifiesto público registra 66 archivos. Este dato identifica el paquete;
no demuestra que todos los artefactos de la entrega final estén restaurados
ni que una Release privada sea accesible para los organizadores. El manifiesto
final debe fijar la URL pública y confirmar la cobertura de dependencias.

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
Este constructor no publica ni inicializa un repositorio.

Antes de publicar: verificar instalación, restauración en una raíz nueva,
generación sin GT, igualdad binaria de dos ejecuciones y validez del CSV;
comprobar acceso anónimo al repositorio y a los modelos. Mantener el CSV y
los datos restringidos fuera del repositorio público. La publicación y el
envío del formulario se realizan tras revisión; este documento no afirma que
esas acciones ya se hayan completado.
