"""OOF development cases only. Detailed case content is for local human review."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile
import numpy as np
from possession.oof_case_review import analyze, aggregate, choose_pairs, viewer
from possession.external_data.schema import file_hash
from possession.external_data.audit import segments
from possession.external_data.features import nearest_indices
from possession.learning.study import ScoreContext
from possession.learning.temporal import fit_priors, TemporalDecoder
from possession.hsmm_pilot import save_context, load_context
from possession.football_mechanism import lift
from possession.conditional_fusion_budget import eligible

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"evaluation/reports/external/oof-case-review-v1"
PARENT = ROOT/"evaluation/reports/external/conditional-fusion-v1"
PUBLIC = ROOT/"docs/public-results/2026-10-29"
CONFIG = ROOT/"configs/oof_case_review_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def sources():
    names = ["possession/oof_case_review.py", "possession/oof_case_review_run.py", "scripts/oof_case_review.py",
             "scripts/run_local.py", "tests/test_oof_case_review.py", "configs/oof_case_review_v1.json",
             "possession/long_diagnosis.py", "possession/learning/temporal.py", "possession/learning/study.py",
             "possession/external_data/audit.py", "possession/external_data/schema.py",
             "possession/external_data/features.py", "possession/learning/metrics.py",
             "possession/conditional_fusion_budget.py", "possession/hsmm_pilot.py",
             "possession/football_mechanism.py", "evaluation/possession_metrics.py"]
    return {n: hashlib.sha256((ROOT/n).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for n in names}


def register():
    spec = read(CONFIG)
    if spec["scenario"] != "clean" or any(spec[k] for k in ("neural_training", "new_detector_fit", "validation_access", "test_access", "target_access")):
        raise ValueError("Clean OOF train review only")
    parent = read(PARENT/"registration.json"); seal = read(PARENT/"result-seal.json")
    if file_hash(PARENT/"registration.json") != read(PUBLIC/"conditional-fusion-v1.json")["registration_sha256"]:
        raise ValueError("Public parent identity mismatch")
    # Check only frozen OOF and expert files; NEVER open parent validation files.
    deps = {}
    for name, digest in seal["outputs"].items():
        if name.startswith(("oof/", "teachers/", "heads/", "inner-geometry/")):
            if file_hash(PARENT/name) != digest:
                raise ValueError("OOF capsule changed")
            deps[name] = digest
    stamp = {"protocol": spec, "sources": sources(), "units": parent["units"],
             "inputs": parent["inputs"], "parent_oof_files": deps,
             "parent_registration_sha256": file_hash(PARENT/"registration.json")}
    marker = DEST/"registration.json"
    if marker.exists() and read(marker) != stamp:
        raise ValueError("Immutable review changed")
    if not marker.exists(): write(marker, stamp)
    return stamp


def labels(key, allowed, stamp):
    if key not in allowed or key not in stamp["inputs"]:
        raise ValueError("Outside this inner train/heldout role")
    item = stamp["inputs"][key]; path = (ROOT/item["path"]).resolve()
    if not path.is_relative_to((ROOT/"data/external/processed").resolve()) or file_hash(path) != item["sha256"]:
        raise ValueError("Public input provenance mismatch")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in ("frame", "period", "time", "alive", "team", "confidence")}
        metadata = json.loads(str(z["metadata"]))
    if metadata["provider"]+":"+metadata["match_id"] != key:
        raise ValueError("Public identity mismatch")
    return SimpleNamespace(metadata=metadata, **data)


def view_labels(m, time_values=None, run_values=None):
    if time_values is None:
        time_values, run_values = [], []
        for r, (a, b) in enumerate(segments(m)):
            grid = m.time[a]+np.arange(int(np.floor((m.time[b-1]-m.time[a])*10+1e-5))+1)/10
            time_values.extend(grid); run_values.extend([r]*len(grid))
        time_values, run_values = np.array(time_values), np.array(run_values)
    native = np.empty(len(time_values), int)
    for r, (a, b) in enumerate(segments(m)):
        ix = np.flatnonzero(run_values == r)
        native[ix] = a+nearest_indices(m.time[a:b], time_values[ix])
    return SimpleNamespace(time=np.asarray(time_values), run=np.asarray(run_values),
                           alive=m.alive[native], team=m.team[native])


def decode(context, p, priors):
    alive = p[:, :2].sum(-1)
    raw_codes = np.where(alive >= .5, (p[:, 1] > p[:, 0]).astype(np.int8), 2)
    raw = context.decoded(np.eye(3)[raw_codes], TemporalDecoder({"kind": "argmax"}, None))
    hmm = context.decoded(p, TemporalDecoder({"kind": "hmm", "strength": .5}, priors))
    return raw, hmm


def clip(row, context, m, p, raw, hmm, stamp, spec):
    part_index, part = next((i, part) for i, part in enumerate(context.parts)
                           if part[2].start <= row["start"] and row["stop"] <= part[2].stop)
    selected, mapping, output = part
    origin = context.view.time[selected[0]]
    center_start = origin+(row["start"]-output.start)/25
    center_end = origin+(row["stop"]-output.start)/25
    first = max(origin, center_start-spec["clip_halo_seconds"])
    last = min(origin+(output.stop-output.start-1)/25, center_end+spec["clip_halo_seconds"])
    times = np.arange(first, last+1e-6, 1/spec["display_hz"])
    a, b = segments(m)[part_index]  # clocks can reset between periods
    native = a+nearest_indices(m.time[a:b], times)
    canonical = output.start+np.clip(np.round((times-origin)*25).astype(int), 0, output.stop-output.start-1)
    model_ix = selected[mapping[canonical-output.start]]
    path = ROOT/stamp["inputs"][row["key"]]["path"]
    # Geometry is read ONLY for these small, human-review windows. Never enters chat.
    with np.load(path, allow_pickle=False) as z:
        players = z["players"][native]; ball = z["ball"][native, :2]
    frames = []
    for j, t in enumerate(times):
        positions = [[round(float(x), 3), round(float(y), 3), side]
                     for side in (0, 1) for x, y in players[j, side] if np.isfinite([x, y]).all()]
        alive = float(p[model_ix[j], :2].sum())
        frames.append({"t": float(t-first), "players": positions,
            "ball": np.round(ball[j], 3).tolist() if np.isfinite(ball[j]).all() else None,
            "gt": "HADX"[int(context.gt[canonical[j]])], "raw": "HADX"[int(raw[canonical[j]])],
            "hmm": "HADX"[int(hmm[canonical[j]])], "home": float(p[model_ix[j], 0]/alive) if alive > 1e-12 else .5,
            "alive": alive})
    return {"id": row["id"], "role": "acierto" if row["recovered"] else "fallo",
            "family": row["family"], "duration_s": row["duration_s"],
            "central": [float(center_start-first), float(center_end-first)],
            "pitch": m.metadata["pitch_m"], "frames": frames}


def summarize_rows(catalog, spec):
    folds = {str(f): aggregate([r for r in catalog if r["fold"] == f]) for f in (0, 6)}
    chosen = {str(f): choose_pairs([r for r in catalog if r["fold"] == f], spec) for f in (0, 6)}
    return {"folds": folds, "review_pairs": {f: len(v) for f, v in chosen.items()},
        "neural_training": False, "new_detector_fitted": False, "validation_accessed": False,
        "test_accessed": False, "target_accessed": False,
        "interpretation": "Operational OOF development diagnosis, not causal attribution or independent confirmation."}, chosen


def run():
    started = time.perf_counter(); stamp = register(); spec = stamp["protocol"]
    catalog = []; cases = {}
    def check():
        if time.perf_counter()-started > spec["max_seconds"]:
            raise TimeoutError("Bounded case review budget exhausted")
    for u in stamp["units"]:
        check()
        train_views = [view_labels(labels(k, u["train"], stamp)) for k in u["train"]]
        prior = fit_priors(train_views)  # counts from INNER TRAIN only, no neural fitting
        folder = DEST/f"cases/fold-{u['outer_fold']}/inner-{u['inner_fold']}"
        folder.mkdir(parents=True, exist_ok=True); np.savez_compressed(folder/"priors.npz", **prior)
        for index, key in enumerate(u["heldout"]):
            check()
            source = PARENT/f"oof/fold-{u['outer_fold']}/inner-{u['inner_fold']}/match-{index}-clean.npz"
            prov = read(source.with_suffix(".provenance.json"))
            if (prov["sha256"] != file_hash(source) or prov["key"] != key
                or not eligible(prov["fit_keys"], prov["selection_keys"], u["heldout"], u["excluded"])):
                raise ValueError("OOF role/provenance failed")
            with np.load(source, allow_pickle=False) as z:
                p, active, t, runs = (z[n] for n in ("base", "active", "time", "run"))
            m = labels(key, u["heldout"], stamp); v = view_labels(m, t, runs)
            context = ScoreContext(m, v); raw, hmm = decode(context, p, prior)
            identity = f"fold-{u['outer_fold']}/inner-{u['inner_fold']}/match-{index}"
            base = folder/f"match-{index}"
            save_context(base.with_suffix(".context.npz"), context)
            np.savez_compressed(base.with_suffix(".inputs.npz"), base=p, active=active, time=t, run=runs)
            rows = analyze(context, p, active, raw, hmm, identity, spec)
            for row in rows: row.update(fold=u["outer_fold"], key=key)
            catalog.extend(rows); cases[identity] = (base, key)
    summary, pairs = summarize_rows(catalog, spec)
    payload = {"pairs": []}
    for fold, entries in pairs.items():
        for pair in entries:
            displayed = []
            for row in pair["cases"]:
                check(); base, key = cases[row["identity"]]
                with np.load(base.with_suffix(".inputs.npz"), allow_pickle=False) as z: p = z["base"]
                context = load_context(base.with_suffix(".context.npz"))
                with np.load(base.parent/"priors.npz", allow_pickle=False) as z: prior = dict(z)
                raw, hmm = decode(context, p, prior)
                m = labels(key, [key], stamp)
                displayed.append(clip(row, context, m, p, raw, hmm, stamp, spec))
            payload["pairs"].append({"fold": fold, "cases": displayed})
    write(DEST/"case-catalog.json", catalog); write(DEST/"review-clips.json", payload)
    (DEST/"review.html").write_text(viewer(payload), encoding="utf-8", newline="\n")
    summary.update(seconds=time.perf_counter()-started, registration_sha256=file_hash(DEST/"registration.json"))
    write(DEST/"summary.json", summary)
    check()
    if sources() != stamp["sources"]: raise ValueError("Review source changed")
    files = [p for p in DEST.rglob("*") if p.is_file() and p.name not in ("registration.json", "result-seal.json")]
    write(DEST/"result-seal.json", {"registration_sha256": file_hash(DEST/"registration.json"),
          "files": {p.relative_to(DEST).as_posix(): file_hash(p) for p in files}})
    return verify()


def verify():
    stamp, seal = read(DEST/"registration.json"), read(DEST/"result-seal.json")
    if stamp["sources"] != sources() or seal["registration_sha256"] != file_hash(DEST/"registration.json"):
        raise ValueError("Review provenance changed")
    for name, digest in seal["files"].items():
        if file_hash(DEST/name) != digest: raise ValueError("Review artifact changed")
    return 0


def replay():
    verify(); stamp = read(DEST/"registration.json"); spec = stamp["protocol"]; catalog = []
    expected = read(DEST/"case-catalog.json")
    by_identity = {r["identity"]: (r["fold"], r["key"]) for r in expected}
    for identity, (fold, key) in by_identity.items():
        base = DEST/"cases"/identity
        context = load_context(base.with_suffix(".context.npz"))
        with np.load(base.with_suffix(".inputs.npz"), allow_pickle=False) as z: p, active = z["base"], z["active"]
        with np.load(base.parent/"priors.npz", allow_pickle=False) as z: prior = dict(z)
        raw, hmm = decode(context, p, prior)
        rows = analyze(context, p, active, raw, hmm, identity, spec)
        for row in rows: row.update(fold=fold, key=key)
        catalog.extend(rows)
    if catalog != expected: raise ValueError("Case catalog replay differs")
    summary, _ = summarize_rows(catalog, spec)
    saved = read(DEST/"summary.json")
    for name, value in summary.items():
        if saved[name] != value: raise ValueError("Aggregate replay differs")
    if viewer(read(DEST/"review-clips.json")) != (DEST/"review.html").read_text(encoding="utf-8"):
        raise ValueError("Viewer replay differs")
    return 0


def export():
    verify(); names = sorted([*read(DEST/"result-seal.json")["files"], "registration.json", "result-seal.json"])
    archive = ROOT/"evaluation/reports/handoff/northmind-oof-case-review-v1.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    prefix = "evaluation/reports/external/oof-case-review-v1/"
    manifest = {n: {"sha256": file_hash(DEST/n), "bytes": (DEST/n).stat().st_size} for n in names}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names: z.write(DEST/n, prefix+n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any(hashlib.sha256(z.read(prefix+n)).hexdigest() != v["sha256"] for n, v in manifest.items()):
            raise ValueError("Review archive integrity failed")
    # Whitelisted aggregate only. Never export a case/coordinate table into chat.
    write(PUBLIC/"oof-case-review-v1.json", read(DEST/"summary.json"))
    write(PUBLIC/"oof-case-review-v1-package.json", {"archive": archive.name, "sha256": file_hash(archive),
        "bytes": archive.stat().st_size, "files": len(names), "verified": True,
        "scope": "Public IDSSE derived OOF context, local qualitative viewer and clips; no target, raw originals or logs"})
    return 0
