"""Symmetric confidence-conditioned gate, fitted only on cross-fitted experts."""
import numpy as np
import torch


def gate_inputs(base_home, geometry_home, x):
    return np.column_stack((np.ones(len(x)), abs(2*base_home-1), abs(2*geometry_home-1),
                            abs(base_home-geometry_home), abs(x[:, 0]), abs(x[:, 2])))


def gate_weight(model, base_home, geometry_home, x):
    z = gate_inputs(base_home, geometry_home, x)/np.array(model["scale"])
    return model["maximum_weight"]/(1+np.exp(-np.clip(z@np.array(model["theta"]), -30, 30)))


def fit_gate(base_home, geometry_home, x, y, weights, spec, check=lambda: None):
    weights = np.asarray(weights)/np.sum(weights)
    z = gate_inputs(base_home, geometry_home, x)
    scale = np.maximum(np.sqrt(np.sum(weights[:, None]*z*z, axis=0)), .05)
    z, base, geo, target, w = [torch.tensor(v, dtype=torch.float64) for v in
                              (z/scale, base_home, geometry_home, y, weights)]
    theta = torch.zeros(6, dtype=torch.float64)
    fraction = spec["initial_geometry_weight"]/spec["maximum_geometry_weight"]
    theta[0] = np.log(fraction/(1-fraction)); theta.requires_grad_()
    optimizer = torch.optim.Adam([theta], lr=spec["gate_learning_rate"])
    history = []
    for step in range(spec["gate_steps"]):
        check(); optimizer.zero_grad()
        alpha = spec["maximum_geometry_weight"]*torch.sigmoid(z@theta)
        p = ((1-alpha)*base+alpha*geo).clamp(1e-7, 1-1e-7)
        loss = (torch.nn.functional.binary_cross_entropy(p, target, reduction="none")*w).sum()
        loss = loss+.5*spec["gate_l2"]*theta[1:].square().sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite crossfit gate loss")
        loss.backward(); optimizer.step()
        if step in (0, spec["gate_steps"]-1):
            history.append(float(loss.detach()))
    return {"theta": theta.detach().tolist(), "scale": scale.tolist(),
            "maximum_weight": spec["maximum_geometry_weight"], "steps": spec["gate_steps"],
            "loss_first_last": history, "samples": len(y), "fit_source": "strict inner crossfit experts"}


def blend(base, geometry_home, x, active, model):
    alive = base[:, :2].sum(-1)
    home = np.divide(base[:, 0], alive, out=np.full(len(base), .5), where=alive > 1e-12)
    alpha = gate_weight(model, home, geometry_home, x)
    alpha = np.where(active, alpha, 0)
    mixed = (1-alpha)*home+alpha*geometry_home
    out = np.array(base, dtype=np.float64, copy=True)
    out[active, 0], out[active, 1] = alive[active]*mixed[active], alive[active]*(1-mixed[active])
    return out, alpha
