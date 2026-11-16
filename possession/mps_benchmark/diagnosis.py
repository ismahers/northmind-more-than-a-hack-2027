"""Separate AdamW sensitivity experiment; never relax the original gate."""
import copy
import json

import torch

from .data import ROOT, idsse, loader_for, weights_for
from .experiment import (cleanup, compare, evaluate_step, initial_model, optimizer_for,
                         snapshot, sync, write)


def assigned_gradient_step(template, gradients, stage, device, spec):
    model = copy.deepcopy(template).to(device)
    optimizer = optimizer_for(model, stage, spec)
    for name, param in model.named_parameters():
        if name in gradients:
            param.grad = gradients[name].to(device).clone()
    # Same clipping as the original step, now on identical gradient tensors.
    torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"], error_if_nonfinite=True)
    optimizer.step(); sync(device)
    return snapshot(model)


def diagnose(folder, env):
    if (folder / "adamw_diagnosis.json").exists():
        raise ValueError("Diagnosis exists; use another run")
    spec = json.loads((ROOT / "configs/mps_benchmark_v1.json").read_text())
    if env["mps_environment"].get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("Fallback prohibited")
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    write(folder / "environment.json", env)
    views, targets, meta = idsse(spec)
    batch = next(iter(loader_for(views, targets, spec, "auxiliary")))
    weights = weights_for(targets, spec)
    rows = []
    for stage in ("base", "auxiliary"):
        template = initial_model(stage, spec)
        cpu = evaluate_step(template, batch, weights, stage, "cpu", spec)
        mps = evaluate_step(template, batch, weights, stage, "mps", spec)
        # Evaluate the CPU AdamW response to the exact MPS gradient, not to the
        # slightly different CPU-autograd result. Keeps eps, lr and architecture.
        from_mps_gradient_cpu = assigned_gradient_step(template, mps["gradients"], stage, "cpu", spec)
        from_mps_gradient_mps = assigned_gradient_step(template, mps["gradients"], stage, "mps", spec)
        details = []
        for name in cpu["gradients"]:
            delta = (cpu["parameters"][name]-mps["parameters"][name]).abs()
            flat = int(delta.argmax())
            details.append({"parameter": name, "max_weight_difference": float(delta.flatten()[flat]),
                            "flat_index": flat,
                            "cpu_gradient_at_max": float(cpu["gradients"][name].flatten()[flat]),
                            "mps_gradient_at_max": float(mps["gradients"][name].flatten()[flat]),
                            "max_gradient_difference": float((cpu["gradients"][name]-mps["gradients"][name]).abs().max()),
                            "parameter_elements": delta.numel()})
        details.sort(key=lambda x: x["max_weight_difference"], reverse=True)
        row = {"stage": stage,
               "original_cross_backend_update": compare(cpu["parameters"], mps["parameters"], spec["tolerances"]["update"]),
               "same_gradient_optimizer_cpu_vs_mps": compare(from_mps_gradient_cpu, from_mps_gradient_mps, spec["tolerances"]["update"]),
               "cpu_with_mps_gradients_vs_actual_mps": compare(from_mps_gradient_cpu, mps["parameters"], spec["tolerances"]["update"]),
               "largest_parameter_differences": details[:8], "adamw_epsilon": 1e-8}
        rows.append(row)
        write(folder / "adamw_diagnosis.json", {"data": meta, "rows": rows, "original_tolerances_unchanged": True})
        del cpu, mps, template, from_mps_gradient_cpu, from_mps_gradient_mps
        cleanup()
    return 0
