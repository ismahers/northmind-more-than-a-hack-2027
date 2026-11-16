"""FP32 CPU/MPS gate and paired timings of the unmodified SpatialE0.

Only aggregate public/synthetic diagnostics are persisted. No historical model
or registration is written. No fallback, AMP, compile or architecture changes.
"""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from possession.external_data.schema import file_hash
from possession.learning.model import masked_losses
from possession.learning.training import forward, memory_usage, seed_everything
from possession.phase5.model import boundary_loss
from possession.phase7.model import SpatialE0
from possession.phase8.packing import pack
from .data import ROOT, idsse, loader_for, synthetic, tensor_hash, weights_for


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def specification(path):
    spec = json.loads((ROOT / path).read_text())
    if "extends" in spec:
        base = json.loads((ROOT / spec["extends"]).read_text())
        base.update(spec)
        return base
    return spec


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def seed(seed_value, device):
    seed_everything(seed_value)
    if device == "mps":
        torch.mps.manual_seed(seed_value)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "mps": torch.mps.get_rng_state().clone() if device == "mps" else None}


def restore_rng(state, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device == "mps":
        torch.mps.set_rng_state(state["mps"])


def initial_model(stage, spec):
    seed(spec["seed"], "cpu")
    return SpatialE0("set_transformer", stage == "auxiliary")


def optimizer_for(model, stage, spec):
    return torch.optim.AdamW(model.parameters(), lr=spec[stage+"_learning_rate"],
                             weight_decay=spec["weight_decay"])


def loss_for(model, batch, weights, stage, spec):
    output = forward(model, batch)
    loss, stats = masked_losses(output, batch["alive"], batch["team"], batch["confidence"], batch["core"])
    auxiliary = torch.zeros((), device=loss.device)
    if stage == "auxiliary":
        auxiliary = boundary_loss(output["boundary_logit"], batch["transition_target"],
                                  (batch["transition_valid"] & batch["core"])[..., None],
                                  weights, "focal", spec["focal_gamma"])
        loss = loss + spec["boundary_weight"] * auxiliary
    return output, loss, stats, auxiliary


def clip(model, spec):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"], error_if_nonfinite=True)


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compare(a, b, tolerance):
    if set(a) != set(b):
        return {"passed": False, "reason": "Tensor keys differ"}
    largest, worst_ratio, squared_delta, squared_reference, passed = 0., 0., 0., 0., True
    for key in a:
        x, y = a[key].detach().cpu().double(), b[key].detach().cpu().double()
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            return {"passed": False, "reason": "Shape mismatch or nonfinite tensor"}
        delta = (x-y).abs()
        limit = tolerance["atol"] + tolerance["rtol"] * x.abs()
        largest = max(largest, float(delta.max()))
        worst_ratio = max(worst_ratio, float((delta / limit.clamp_min(1e-30)).max()))
        squared_delta += float(delta.square().sum())
        squared_reference += float(x.square().sum())
        passed = passed and bool((delta <= limit).all())
    return {"passed": passed, "max_absolute_error": largest, "max_tolerance_ratio": worst_ratio,
            "relative_l2": float(np.sqrt(squared_delta / max(squared_reference, 1e-30)))}


class Guard:
    def __init__(self, folder, spec):
        self.start = time.perf_counter(); self.folder = folder; self.spec = spec

    def check(self, **progress):
        if time.perf_counter()-self.start > self.spec["max_seconds"]:
            raise TimeoutError("Isolated benchmark time budget exceeded")
        if torch.backends.mps.is_available() and torch.mps.driver_allocated_memory() > self.spec["max_mps_driver_bytes"]:
            raise MemoryError("Isolated MPS memory guard exceeded")
        if progress:
            write(self.folder / "progress.json", {"elapsed_seconds": time.perf_counter()-self.start, **progress})


def memory(device):
    value = memory_usage(torch.device("cpu"))
    if device == "mps":
        value.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                     mps_driver_bytes=torch.mps.driver_allocated_memory(),
                     mps_recommended_bytes=torch.mps.recommended_max_memory())
    return value


def evaluate_step(template, batch, weights, stage, device, spec):
    model = copy.deepcopy(template).to(device).eval()
    optimizer = optimizer_for(model, stage, spec)
    data, weight = move(batch, device), weights.to(device)
    output, loss, _, auxiliary = loss_for(model, data, weight, stage, spec)
    loss.backward()
    result = {"outputs": {k: v.detach().cpu().clone() for k, v in output.items()},
              "loss": {"total": loss.detach().cpu().clone(), "auxiliary": auxiliary.detach().cpu().clone()},
              "gradients": {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}}
    clip(model, spec); optimizer.step(); sync(device)
    result["parameters"] = snapshot(model)
    result["memory"] = memory(device)
    return result


def stochastic_run(template, batch, weights, stage, device, spec):
    model = copy.deepcopy(template).to(device).train()
    optimizer = optimizer_for(model, stage, spec)
    data, weight = move(batch, device), weights.to(device)
    seed(spec["seed"]+77, device)
    loss_values = []
    for _ in range(spec["repeatability_updates"]):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = loss_for(model, data, weight, stage, spec)
        loss.backward(); clip(model, spec); optimizer.step()
        loss_values.append(loss.detach().cpu().clone())
    # An in-memory checkpoint verifies restoration of model, AdamW AND MPS RNG.
    state = snapshot(model)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    random_state = rng_state(device)
    def next_update():
        optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = loss_for(model, data, weight, stage, spec)
        loss.backward(); clip(model, spec); optimizer.step(); sync(device)
        return snapshot(model)
    continued = next_update()
    model.load_state_dict(state); optimizer.load_state_dict(optimizer_state)
    restore_rng(random_state, device)
    resumed = next_update()
    return state, {"losses": torch.stack(loss_values)}, compare(continued, resumed, spec["tolerances"]["repeatability"])


def compatibility_case(batch, weights, stage, spec, guard):
    guard.check(stage="compatibility", model_stage=stage)
    template = initial_model(stage, spec)
    before = torch.get_rng_state().clone()
    compact = pack(batch)
    packing_rng = bool(torch.equal(before, torch.get_rng_state()))
    unchanged = all(torch.equal(batch[k], compact[k]) for k in batch if k not in ("nodes", "node_mask"))
    # CPU produces the single shared state; both backends load that same state.
    cpu = evaluate_step(template, compact, weights, stage, "cpu", spec)
    mps = evaluate_step(template, compact, weights, stage, "mps", spec)
    comparisons = {k: compare(cpu[key], mps[key], spec["tolerances"][k])
                   for k, key in (("output", "outputs"), ("loss", "loss"),
                                  ("gradient", "gradients"), ("update", "parameters"))}
    optimizer_control = None
    if spec.get("gate_policy") == "operator_compatibility_with_same_gradient_adamw_control":
        from .diagnosis import assigned_gradient_step
        shared_cpu = assigned_gradient_step(template, mps["gradients"], stage, "cpu", spec)
        shared_mps = assigned_gradient_step(template, mps["gradients"], stage, "mps", spec)
        optimizer_control = compare(shared_cpu, shared_mps, spec["tolerances"]["update"])
        del shared_cpu, shared_mps
    packing = {}
    repeatability = {}
    for device, packed in (("cpu", cpu), ("mps", mps)):
        uncompressed = evaluate_step(template, batch, weights, stage, device, spec)
        packing[device] = {"outputs": compare(packed["outputs"], uncompressed["outputs"], spec["tolerances"]["output"]),
                           "gradients": compare(packed["gradients"], uncompressed["gradients"], spec["tolerances"]["gradient"])}
        first, losses, resumed = stochastic_run(template, compact, weights, stage, device, spec)
        second, losses_again, resumed_again = stochastic_run(template, compact, weights, stage, device, spec)
        repeatability[device] = {"parameters": compare(first, second, spec["tolerances"]["repeatability"]),
                                 "losses": compare(losses, losses_again, spec["tolerances"]["repeatability"]),
                                 "resume": resumed, "resume_second": resumed_again}
    strict_passed = all(v["passed"] for v in comparisons.values())
    numerical_passed = strict_passed if optimizer_control is None else (
        all(comparisons[k]["passed"] for k in ("output", "loss", "gradient")) and optimizer_control["passed"])
    passed = packing_rng and unchanged and numerical_passed
    passed &= all(v["passed"] for checks in packing.values() for v in checks.values())
    passed &= all(v["passed"] for checks in repeatability.values() for v in checks.values())
    result = {"stage": stage, "passed": bool(passed), "strict_cpu_mps_step_parity_passed": strict_passed,
              "same_gradient_optimizer_control": optimizer_control,
              "gate_policy": spec.get("gate_policy", "strict_cross_backend_step"),
              "initial_weights_sha256": tensor_hash(template.state_dict()),
              "parameters": sum(p.numel() for p in template.parameters()), "packing_preserves_rng": packing_rng,
              "packing_preserves_targets_masks": unchanged, "padded_shape": list(batch["nodes"].shape),
              "packed_shape": list(compact["nodes"].shape), "cpu_mps": comparisons,
              "packing": packing, "repeatability": repeatability, "mps_memory": mps["memory"]}
    del template, cpu, mps
    cleanup()
    return result


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"n": len(x), "median_s": float(np.median(x)), "q25_s": float(np.quantile(x, .25)),
            "q75_s": float(np.quantile(x, .75)), "min_s": float(x.min()), "max_s": float(x.max()),
            "values_s": x.tolist()}


def micro_profile(template, batch, weights, stage, device, spec, guard):
    model = copy.deepcopy(template).to(device).train()
    optimizer = optimizer_for(model, stage, spec)
    data, weight = move(batch, device), weights.to(device)
    seed(spec["seed"]+99, device)
    for _ in range(spec["warmup_steps"]):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = loss_for(model, data, weight, stage, spec)
        loss.backward(); clip(model, spec); optimizer.step()
    sync(device)
    samples = {k: [] for k in ("h2d", "d2h", "forward", "loss", "backward", "clip", "optimizer", "inference")}
    observed_memory = []
    for repetition in range(spec["profile_repetitions"]):
        guard.check(stage="micro", model_stage=stage, device=device, repetition=repetition)
        optimizer.zero_grad(set_to_none=True)
        sync(device); start = time.perf_counter(); transferred = move(batch, device); sync(device)
        samples["h2d"].append(time.perf_counter()-start)
        start = time.perf_counter(); returned = move(transferred, "cpu"); sync(device)
        samples["d2h"].append(time.perf_counter()-start)
        del transferred, returned
        sync(device); start = time.perf_counter(); output = forward(model, data); sync(device)
        samples["forward"].append(time.perf_counter()-start)
        observed_memory.append(memory(device))
        start = time.perf_counter()
        loss, _ = masked_losses(output, data["alive"], data["team"], data["confidence"], data["core"])
        if stage == "auxiliary":
            loss = loss + spec["boundary_weight"] * boundary_loss(output["boundary_logit"], data["transition_target"],
                (data["transition_valid"] & data["core"])[..., None], weight, "focal", spec["focal_gamma"])
        sync(device); samples["loss"].append(time.perf_counter()-start)
        start = time.perf_counter(); loss.backward(); sync(device)
        samples["backward"].append(time.perf_counter()-start)
        start = time.perf_counter(); clip(model, spec); sync(device)
        samples["clip"].append(time.perf_counter()-start)
        start = time.perf_counter(); optimizer.step(); sync(device)
        samples["optimizer"].append(time.perf_counter()-start)
        observed_memory.append(memory(device))
        del output, loss
    model.eval()
    with torch.no_grad():
        for j in range(spec["warmup_steps"]+spec["profile_repetitions"]):
            sync(device); start = time.perf_counter(); output = forward(model, data); sync(device)
            if j >= spec["warmup_steps"]:
                samples["inference"].append(time.perf_counter()-start)
            del output
    result = {"stage": stage, "device": device, "shape": list(data["nodes"].shape),
              "batch_sha256": tensor_hash(batch), "initial_weights_sha256": tensor_hash(template.state_dict()),
              "timings": {k: stats(v) for k, v in samples.items()},
              "memory_sampled_max": {k: max(row[k] for row in observed_memory) for k in observed_memory[0] if observed_memory[0][k] is not None},
              "note": "Synchronized components add barriers; do not sum them to estimate epoch speedup. CPU transfers are no-ops; inference is eval/no_grad; forward is training mode."}
    return result


def epoch(template, views, targets, weights, stage, device, spec, guard):
    model = copy.deepcopy(template).to(device).train()
    optimizer = optimizer_for(model, stage, spec)
    weight = weights.to(device)
    # Warmup is discarded, including optimizer moments and stochastic state.
    loader = loader_for(views, targets, spec, stage)
    sample = move(next(iter(loader)), device)
    for _ in range(spec["warmup_steps"]):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = loss_for(model, sample, weight, stage, spec)
        loss.backward(); clip(model, spec); optimizer.step()
    del loss, sample, loader
    model.load_state_dict(template.state_dict()); optimizer = optimizer_for(model, stage, spec)
    seed(spec["seed"]+6000*(stage == "auxiliary"), device)
    sync(device); start = time.perf_counter()
    loader = loader_for(views, targets, spec, stage)
    preparation_s = time.perf_counter()-start
    # Record actual sampler order with negligible Python append overhead. Hash it
    # after timing, so identical exposure is checked without hashing whole batches.
    order = []
    original_sampler = loader.batch_sampler.sampler
    class RecordingSampler:
        def __iter__(self):
            for index in original_sampler:
                order.append(index)
                yield index

        def __len__(self):
            return len(original_sampler)
    loader.batch_sampler.sampler = RecordingSampler()
    updates, core_frames, memory_samples, last_loss = 0, 0, [], None
    for batch in loader:
        guard.check()
        if time.perf_counter()-start > spec["max_epoch_seconds"]:
            raise TimeoutError("Single-epoch stability guard exceeded; no extrapolated epoch reported")
        data = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss, loss_stats, auxiliary = loss_for(model, data, weight, stage, spec)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite epoch loss")
        # Match historical per-minibatch telemetry/synchronization.
        loss_stats.cpu().numpy()
        loss.backward(); norm = clip(model, spec); float(norm)
        optimizer.step()
        if stage == "auxiliary":
            float(auxiliary.detach())
        updates += 1; core_frames += int(batch["core"].sum())
        if updates % 10 == 0:
            memory_samples.append(memory(device))
        last_loss = loss
    sync(device); elapsed = time.perf_counter()-start
    if not updates:
        raise ValueError("Empty benchmark epoch")
    memory_samples.append(memory(device))
    return {"device": device, "stage": stage, "seconds": elapsed, "preparation_s": preparation_s,
            "updates": updates, "windows": len(loader.dataset), "supervised_core_frames": core_frames,
            "window_order_sha256": hashlib.sha256(np.asarray(order, dtype="i8").tobytes()).hexdigest(),
            "initial_weights_sha256": tensor_hash(template.state_dict()),
            "final_loss_finite": bool(torch.isfinite(last_loss)),
            "memory_sampled_max": {k: max(row[k] for row in memory_samples) for k in memory_samples[0] if memory_samples[0][k] is not None}}


def summarize(folder):
    compat = json.loads((folder / "compatibility.json").read_text())
    rows_path = folder / "benchmark.json"
    benchmark = json.loads(rows_path.read_text()) if rows_path.exists() else {"micro": [], "epochs": []}
    rows = []
    for source in sorted({r["source"] for r in benchmark["epochs"]}):
        for stage in ("base", "auxiliary"):
            by_device = {d: [r for r in benchmark["epochs"] if r["source"] == source and r["stage"] == stage and r["device"] == d]
                         for d in ("cpu", "mps")}
            if not all(by_device.values()):
                continue
            paired = [a["seconds"]/b["seconds"] for a, b in zip(by_device["cpu"], by_device["mps"])]
            counts = {(r["updates"], r["windows"], r["supervised_core_frames"]) for group in by_device.values() for r in group}
            orders = {r["window_order_sha256"] for group in by_device.values() for r in group}
            initial = {r["initial_weights_sha256"] for group in by_device.values() for r in group}
            if len(counts) != 1 or len(orders) != 1 or len(initial) != 1:
                raise ValueError("Paired epoch exposure mismatch")
            rows.append({"source": source, "stage": stage,
                         "cpu": stats([r["seconds"] for r in by_device["cpu"]]),
                         "mps": stats([r["seconds"] for r in by_device["mps"]]),
                         "speedup_ratio_of_medians": float(np.median([r["seconds"] for r in by_device["cpu"]]) / np.median([r["seconds"] for r in by_device["mps"]])),
                         "paired_speedup_median": float(np.median(paired)), "paired_speedups": paired,
                         "updates_per_epoch": next(iter(counts))[0]})
    result = {"compatibility_passed": compat["passed"], "epoch_comparisons": rows,
              "scope": "Training-only fresh epochs on this machine; excludes initial public feature loading, validation/scoring and checkpoint I/O. No trained teacher or scientific metric produced.",
              "micro": benchmark["micro"], "compatibility": compat}
    write(folder / "summary.json", result)
    lines = ["# CPU/MPS isolated benchmark", "", result["scope"], "", f"Compatibility: {compat['passed']}", "",
             "| Source | Stage | CPU median s [Q25–Q75] | MPS median s [Q25–Q75] | Speedup | Updates |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        cells = [f"{row[d]['median_s']:.4f} [{row[d]['q25_s']:.4f}–{row[d]['q75_s']:.4f}]" for d in ("cpu", "mps")]
        lines.append(f"| {row['source']} | {row['stage']} | {cells[0]} | {cells[1]} | {row['speedup_ratio_of_medians']:.3f}x | {row['updates_per_epoch']} |")
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(folder, stage, environment, config_path="configs/mps_benchmark_v1.json"):
    spec = specification(config_path)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS fallback must be absent or zero before Python starts")
    if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
        raise ValueError("MPS fast math must remain disabled")
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    env_path = folder / "environment.json"
    if not env_path.exists():
        write(env_path, environment())
    paths = [ROOT / "configs/mps_benchmark_v1.json", ROOT / "scripts/mps_benchmark.py"]
    if config_path != "configs/mps_benchmark_v1.json":
        paths.append(ROOT / config_path)
    paths += sorted((ROOT / "possession/mps_benchmark").glob("*.py"))
    record = {"spec": spec, "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in paths}}
    registration = folder / "registration.json"
    if registration.exists() and json.loads(registration.read_text()) != record:
        raise ValueError("Isolated recipe changed; use a new run")
    write(registration, record)
    guard = Guard(folder, spec)
    try:
        if stage == "report":
            summarize(folder); return 0
        if not torch.backends.mps.is_available():
            write(folder / "diagnosis.json", {"status": "MPS unavailable", "benchmark_run": False})
            return 1
        if stage in ("compatibility", "all"):
            if (folder / "compatibility.json").exists():
                raise ValueError("Compatibility already recorded; use another run")
            views, targets, meta = synthetic(spec)
            loader = loader_for(views, targets, spec, "auxiliary", corrupt=False)
            # Use deliberately scattered padding and empty frames on all backends.
            batch = next(iter(loader))
            batch["node_mask"][0, 15:18] = False
            batch["node_mask"][0, 30:32, 0] = False
            batch["valid"][0, -4:] = False; batch["core"][0, -4:] = False
            # Reintroduce noncontiguous padding after the normal CPU collator.
            padded = dict(batch)
            p = batch["nodes"].shape[3]
            padded["nodes"] = torch.zeros(*batch["nodes"].shape[:3], p*2, 6)
            padded["node_mask"] = torch.zeros(*batch["node_mask"].shape[:3], p*2, dtype=torch.bool)
            padded["nodes"][..., ::2, :] = batch["nodes"]
            padded["node_mask"][..., ::2] = batch["node_mask"]
            weights = weights_for(targets, spec)
            results = []
            for model_stage in ("base", "auxiliary"):
                results.append(compatibility_case(padded, weights, model_stage, spec, guard))
                write(folder / "compatibility.json", {"passed": False, "complete": False, "synthetic": results})
            passed = all(r["passed"] for r in results)
            write(folder / "compatibility.json", {"passed": passed, "complete": True, "synthetic": results})
            if not passed:
                summarize(folder); return 1
            del views, targets, loader, batch, padded
            cleanup()
        compatibility = json.loads((folder / "compatibility.json").read_text())
        if not compatibility["passed"]:
            raise ValueError("Compatibility gate not passed")
        if stage in ("benchmark", "all"):
            if (folder / "benchmark.json").exists():
                raise ValueError("Benchmark exists; use another run")
            results = {"micro": [], "epochs": [], "data": []}
            sources = [("synthetic", synthetic(spec)), ("IDSSE", idsse(spec))]
            for source, data in sources:
                if data is None:
                    results["data"].append({"source": source, "available": False}); continue
                views, targets, meta = data
                results["data"].append(meta)
                weights = weights_for(targets, spec)
                for model_stage in ("base", "auxiliary"):
                    template = initial_model(model_stage, spec)
                    sample = next(iter(loader_for(views, targets, spec, model_stage)))
                    # Gate public inputs as well; avoids benchmarking an untested mask/shape.
                    if source == "IDSSE":
                        checked = compatibility_case(next(iter(loader_for(views, targets, spec, "auxiliary"))),
                                                     weights, model_stage, spec, guard)
                        compatibility.setdefault("IDSSE", []).append(checked)
                        compatibility["passed"] &= checked["passed"]
                        write(folder / "compatibility.json", compatibility)
                        if not checked["passed"]:
                            raise ValueError("Public-input compatibility failed")
                    for device in ("cpu", "mps"):
                        guard.check(stage="micro", source=source, model_stage=model_stage, device=device)
                        row = micro_profile(template, sample, weights, model_stage, device, spec, guard)
                        row["source"] = source; results["micro"].append(row)
                        cleanup(); write(folder / "benchmark.json", results)
                    for repetition in range(spec["epoch_repetitions"]):
                        # Alternating order reduces systematic thermal/order bias.
                        for device in (("cpu", "mps") if repetition % 2 == 0 else ("mps", "cpu")):
                            guard.check(stage="epoch", source=source, model_stage=model_stage, device=device, repetition=repetition)
                            row = epoch(template, views, targets, weights, model_stage, device, spec, guard)
                            row.update(source=source, repetition=repetition)
                            results["epochs"].append(row); write(folder / "benchmark.json", results)
                            cleanup()
                    del template, sample
                del views, targets
                cleanup()
            write(folder / "benchmark.json", results)
        summarize(folder)
        current = environment()
        original = json.loads(env_path.read_text())
        if current["protected_hashes"] != original["protected_hashes"]:
            raise ValueError("Protected reference sources changed")
        write(folder / "verification.json", {"protected_sources_unchanged": True,
             "fallback_disabled": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "cpu_threads": torch.get_num_threads(), "default_device": str(torch.get_default_device()),
             "no_historical_checkpoints_loaded_or_written": True, "completed": True})
        return 0
    except Exception as exc:
        # Sources are exclusively synthetic/public; raw traceback stays in captured log.
        write(folder / "diagnosis.json", {"exception_type": type(exc).__name__, "message": str(exc)[:2000],
             "progress": json.loads((folder / "progress.json").read_text()) if (folder / "progress.json").exists() else None})
        traceback.print_exc()
        return 1
