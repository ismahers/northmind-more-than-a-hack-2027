"""Build a self-contained source-only notebook. Explicit allowlist excludes data and weights."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import textwrap
import zipfile
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def source_archive():
    names = ["possession/__init__.py", "scripts/__init__.py", "scripts/run_local.py",
        "scripts/external_data.py", "scripts/train_e0.py", "scripts/external_environment.py",
        "scripts/external_benchmark.py", "scripts/external_readiness.py", "scripts/smoke_e0.py", "scripts/external_reproduction.py",
        "scripts/build_colab_phase2.py", "requirements-external.txt", "evaluation/possession_metrics.py",
        "LICENSE", "docs/EXTERNAL_PHASE1.md", "docs/EXTERNAL_PHASE2.md", "notebooks/colab_phase1.ipynb"]
    for pattern in ("possession/external_data/*.py", "possession/learning/*.py", "configs/*.json", "tests/test_external_*.py"):
        names.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    result, hashes = BytesIO(), {}
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT) or name.startswith(("data/", "evaluation/reports/")):
                raise ValueError("Only source files in the explicit allowlist can enter Colab")
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return result.getvalue(), hashes


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(value).strip().splitlines(keepends=True)}


def code(value):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(value).strip().splitlines(keepends=True)}


def make_notebook():
    payload, hashes = source_archive()
    digest = hashlib.sha256(payload).hexdigest()
    cells = [markdown("""
        # Northmind — IDSSE real + E0 en GPU
        **Solo hay que subir este notebook a Colab, seleccionar GPU y ejecutar todas las celdas.**
        Autorizar Drive cuando Colab lo solicite (persistencia opcional). No subir datasets ni ZIP de código.
        El código va incorporado y verificado por SHA-256; no depende de un push ni de una rama cambiante.
        Descarga siete IDSSE públicos, audita, verifica leakage y entrena **solo el fold 0** por defecto.
        No contiene datos, GT del hackathon ni checkpoints. No ejecuta siete folds automáticamente.
        Los resultados de un fold son un primer benchmark, no evidencia de generalización sobre siete partidos.
        """
    ), code(textwrap.dedent("""
        from pathlib import Path
        import base64, hashlib, io, json, os, subprocess, sys, zipfile
        REPO = Path('/content/northmind-phase2')
        """) + "\nPAYLOAD_B64 = " + repr(base64.b64encode(payload).decode("ascii"))
        + "\nEXPECTED_BUNDLE_SHA256 = " + repr(digest)
        + "\nSOURCE_HASHES = " + repr(hashes) + "\n" + textwrap.dedent("""
        payload = base64.b64decode(PAYLOAD_B64)
        assert hashlib.sha256(payload).hexdigest() == EXPECTED_BUNDLE_SHA256
        REPO.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert set(archive.namelist()) == set(SOURCE_HASHES)
            for name in archive.namelist():
                target = (REPO / name).resolve()
                assert target.is_relative_to(REPO.resolve())
                content = archive.read(name)
                assert hashlib.sha256(content).hexdigest() == SOURCE_HASHES[name]
                if target.exists() and target.read_bytes() != content:
                    raise ValueError('Existe otra versión de código; conservar el run y usar un runtime limpio')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        os.chdir(REPO)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-external.txt'])
        def task(name, *args):
            subprocess.check_call([sys.executable, '-m', 'scripts.run_local', name, *map(str, args)], cwd=REPO)
        print('Código verificado:', EXPECTED_BUNDLE_SHA256)
        """)), code("""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Seleccionar Entorno de ejecución > Cambiar tipo de entorno > GPU y reconectar')
        print({'GPU': torch.cuda.get_device_name(), 'VRAM_libre_total': torch.cuda.mem_get_info(), 'torch': str(torch.__version__)})
        """), code("""
        USE_DRIVE = True  # False: almacenamiento efímero y exportación final; True: autorizar una vez.
        if USE_DRIVE:
            from google.colab import drive
            drive.mount('/content/drive')
            storage = Path('/content/drive/MyDrive/northmind-idsse-phase2')
            for local, persistent in [(REPO/'data/external', storage/'data'), (REPO/'evaluation/reports/external', storage/'reports')]:
                persistent.mkdir(parents=True, exist_ok=True)
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    local.symlink_to(persistent, target_is_directory=True)
                elif local.resolve() != persistent.resolve():
                    raise ValueError('La carpeta externa ya existe fuera de Drive; conservarla antes de cambiar persistencia')
        task('check-external-environment')
        task('test-external-phase1')
        reports = REPO/'evaluation/reports/external'
        (reports/'source_bundle.json').write_text(json.dumps({'sha256': EXPECTED_BUNDLE_SHA256, 'sources': SOURCE_HASHES}, indent=2))
        """), markdown("""
        ## Datos y comprobaciones automáticas
        Aproximadamente 2,6 GB de originales y 124 MiB procesados en la validación local.
        El primer acceso descarga; las repeticiones verifican hashes y reutilizan originales.
        No se leen eventos ni assets del hackathon. Si el chequeo semántico/temporal falla, no se entrena.
        """), code("""
        task('download-external-data', '--provider', 'idsse')
        task('prepare-external-data', '--provider', 'idsse')
        task('summarize-external-audits')
        audit = json.loads((reports/'idsse_audit_summary.json').read_text())
        if not audit['complete_labels_and_clocks']:
            raise ValueError('Auditoría incompleta: revisar JSON externos antes de entrenar')
        task('make-external-folds')
        task('verify-external-readiness')
        import pandas as pd
        display(pd.DataFrame([{k:r[k] for k in ('match','frames','joint_coverage','unknown_frames','possessions','ball_available')} for r in audit['matches']]))
        """), markdown("""
        ## Primer entrenamiento real, fold 0
        Configuración GPU fijada: E0 32/64, 10 Hz, batch 4, dos épocas, sin AMP ni límite de minibatches.
        Cinco partidos entrenan; J03WR9 valida y J03WMX es test. El mejor checkpoint usa pérdida de validación.
        La reanudación es automática si existe `last.pt`; no se prolonga un run que ya terminó.
        Cambiar configuración requiere otro nombre de run. FOLDS=[0] conserva los otros seis sin ejecutar.
        """), code("""
        RUN = 'e0-idsse-gpu-first'
        FOLDS = [0]
        config_path = REPO/'configs/e0_idsse_colab.json'
        config = json.loads(config_path.read_text())
        for fold in FOLDS:
            output = reports/'experiments'/RUN/f'fold-{fold}'
            last = output/'last.pt'
            complete = False
            if last.exists():
                state = torch.load(last, map_location='cpu', weights_only=False)
                complete = state['epoch'] >= config['epochs']
                del state
            if not complete:
                args = ['--fold',fold,'--run',RUN,'--config',config_path]
                if last.exists(): args.append('--resume')
                task('train-external-e0', *args)
            task('evaluate-external-e0','--fold',fold,'--run',RUN,'--split','validation')
            task('evaluate-external-e0','--fold',fold,'--run',RUN,'--split','test')
        task('summarize-external-benchmark','--run',RUN)
        display(pd.read_csv(reports/'benchmarks'/f'{RUN}.csv'))
        """), markdown("""
        ## Resultados y persistencia
        El CSV deja los folds no ejecutados explícitos. Probabilidades a 10 Hz y por frame nativo a 25 Hz
        quedan junto a las métricas para reutilizar HMM/HSMM. Las métricas por disponibilidad del balón
        describen coordenadas finitas, no visibilidad visual independiente. En IDSSE no hay frames sin XY.
        Los checkpoints y datos permanecen en la carpeta Drive dedicada si se activó.
        La exportación solo incluye métricas/configuraciones; no datasets ni checkpoints pesados.
        """), code("""
        archive_path = Path('/content')/f'{RUN}-metrics.zip'
        with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in (reports/'experiments'/RUN).rglob('*.json'):
                archive.write(p, str(p.relative_to(reports)))
            for p in (reports/'benchmarks').glob(f'{RUN}.*'):
                archive.write(p, str(p.relative_to(reports)))
            for name in ('environment.json','idsse_audit_summary.json','readiness.json','source_bundle.json'):
                archive.write(reports/name, name)
        from google.colab import files
        files.download(str(archive_path))
        """)]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "source_bundle_sha256": digest}, "cells": cells}


def main():
    notebook = make_notebook()
    target = ROOT / "notebooks" / "colab_phase2_idsse.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


def verify_main():
    """Run the extracted source snapshot offline, without Colab/account/data access."""
    notebook = json.loads((ROOT / "notebooks/colab_phase2_idsse.ipynb").read_text(encoding="utf-8"))
    import ast
    first = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse("".join(first["source"])).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("PAYLOAD_B64", "EXPECTED_BUNDLE_SHA256", "SOURCE_HASHES")}
    payload = base64.b64decode(assignments["PAYLOAD_B64"])
    if hashlib.sha256(payload).hexdigest() != assignments["EXPECTED_BUNDLE_SHA256"]:
        raise ValueError("Invalid notebook payload")
    reports = ROOT / "evaluation/reports/external"
    reports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colab-source-check-", dir=reports) as directory:
        directory = Path(directory)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            if set(archive.namelist()) != set(assignments["SOURCE_HASHES"]):
                raise ValueError("Unexpected notebook payload files")
            for name in archive.namelist():
                target = (directory / name).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise ValueError("Unsafe archive path")
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != assignments["SOURCE_HASHES"][name]:
                    raise ValueError("Source integrity failure")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        subprocess.run([sys.executable, "-m", "scripts.run_local", "test-external-phase1"],
                       cwd=directory, stdin=subprocess.DEVNULL, check=True)
    (reports / "colab_bundle_verification.json").write_text(json.dumps({
        "bundle_sha256": assignments["EXPECTED_BUNDLE_SHA256"], "source_files": len(assignments["SOURCE_HASHES"]),
        "standalone_source_tests_passed": True, "colab_gpu_executed": False,
        "contains_match_data_or_checkpoints": False}, indent=2), encoding="utf-8")
    return 0
