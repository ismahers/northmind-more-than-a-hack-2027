"""Finite-pool minimum-risk ranking of COMPLETE sequences, without GT inputs."""
import numpy as np
import torch


def spans(y):
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1])+1, len(y)]
    return [(int(a), int(b), int(y[a])) for a, b in zip(edges[:-1], edges[1:])]


def render(y, boundaries):
    original = spans(y)
    boundaries = np.asarray(boundaries, dtype=np.int64)
    cuts = np.r_[0, np.maximum.accumulate(np.clip(boundaries, 0, len(y))), len(y)]
    out = y.copy()
    for (a, b), (_, _, state) in zip(zip(cuts[:-1], cuts[1:]), original):
        out[a:b] = state
    return out


def snap(y, p, radius, fraction=1.):
    runs = spans(y)
    edges = []
    for (_, b, old), (_, _, new) in zip(runs[:-1], runs[1:]):
        positions = np.arange(max(1, b-radius), min(len(y), b+radius+1))
        margin = p[:, new]-p[:, old]
        gradient = margin[positions]-margin[positions-1]
        # Equal evidence retains the closest boundary; no dependence on team name.
        best = np.lexsort((positions, abs(positions-b), -gradient))[0]
        target = positions[best] if gradient[best] > 0 else b
        edges.append(round(b+fraction*(target-b)))
    return render(y, np.asarray(edges, int))


def erase(y, p, cap, confidence):
    runs, out = spans(y), y.copy()
    for i, (a, b, state) in enumerate(runs[1:-1], 1):
        left, right = runs[i-1][2], runs[i+1][2]
        if left == right and b-a <= cap and float(p[a:b, state].mean()) < confidence:
            out[a:b] = left
    return out


def policies():
    specs = [("baseline", "base", 0, 0)]
    specs += [(f"shift_{d}", "shift", d, 0) for d in (-5, -2, -1, 1, 2, 5)]
    specs += [(f"snap_{r}_{f}", "snap", r, f) for r in (2, 5, 10) for f in (.5, 1.)]
    specs += [(f"erase_{d}_{q}", "erase", d, q) for d in (5, 10, 20) for q in (.55, .65)]
    specs += [(f"snap_erase_{d}", "combined", d, q) for d, q in ((10, .55), (20, .65))]
    return specs


def candidates(base, probabilities, run):
    if (probabilities.shape != (len(base), 3) or run.shape != base.shape or
            not np.isin(base, (0, 1, 2)).all() or not np.isfinite(probabilities).all()):
        raise ValueError("Invalid sequence candidate inputs")
    results = []
    for _, kind, a, b in policies():
        result = base.copy()
        for identity in np.unique(run):
            ix = np.flatnonzero(run == identity)
            if np.any(np.diff(ix) != 1):
                raise ValueError("Run must be contiguous")
            y, p = base[ix], probabilities[ix]
            if kind == "base":
                z = y
            elif kind == "shift":
                z = render(y, np.array([end for _, end, _ in spans(y)[:-1]])+a)
            elif kind == "snap":
                z = snap(y, p, a, b)
            elif kind == "erase":
                z = erase(y, p, a, b)
            else:
                z = erase(snap(y, p, 5), p, a, b)
            result[ix] = z
        results.append(result)
    result = np.asarray(results, np.int8)
    # Duplicate hypotheses must not gain extra softmax mass.
    seen, keep = set(), []
    for y in result:
        key = y.tobytes()
        keep.append(key not in seen)
        seen.add(key)
    return result, np.asarray(keep)


def features(hypotheses, probabilities, run):
    """Team-equivariant sequence summaries; no clock labels or GT consumed."""
    logp = np.log(np.clip(probabilities, 1e-7, 1))
    n = len(run)
    rows = []
    for i, y in enumerate(hypotheses):
        chosen = probabilities[np.arange(n), y]
        intervals = [span for identity in np.unique(run) for span in spans(y[run == identity])]
        alive_lengths = np.array([b-a for a, b, state in intervals if state != 2], float)
        dead_lengths = [b-a for a, b, state in intervals if state == 2]
        adjacent = (run[1:] == run[:-1]) & (y[1:] != y[:-1])
        boundaries = np.flatnonzero(adjacent)+1
        ha = adjacent & (y[1:] != 2) & (y[:-1] != 2)
        change_evidence = [probabilities[b, y[b]]-probabilities[b-1, y[b]] +
                           probabilities[b-1, y[b-1]]-probabilities[b, y[b-1]] for b in boundaries]
        rows.append([float(i == 0), float(logp[np.arange(n), y].mean()),
            float(chosen.mean()), float(np.quantile(chosen, .1)), float((chosen < .5).mean()),
            float((y == 2).mean()), len(boundaries)/n*600, float(ha.sum())/n*600,
            float((y != hypotheses[0]).mean()),
            sum(d < 10 for d in alive_lengths)/n*600,
            sum(d < 30 for d in alive_lengths)/n*600,
            sum(d >= 100 for d in alive_lengths)/n*600,
            float(np.log1p(alive_lengths).mean()) if len(alive_lengths) else 0.,
            float(np.log1p(dead_lengths).mean()) if dead_lengths else 0.,
            float(np.mean(change_evidence)) if change_evidence else 0.])
    result = np.asarray(rows, np.float64)
    return result-result[0]


def expected_risk(theta, x, losses, mask, weights):
    logits = torch.einsum("bkd,d->bk", x, theta).masked_fill(~mask, -torch.inf)
    return ((logits.softmax(-1)*losses).sum(-1)*weights).sum()/weights.sum()


def fit(x, losses, mask, weights, steps=400):
    # Normalize only on this fold's training cases. Baseline's centered vector is zero.
    scale = np.maximum(np.std(x[mask], axis=0), 1e-6)
    x = torch.tensor(x/scale, dtype=torch.float64)
    losses = torch.tensor(losses, dtype=torch.float64)
    mask = torch.tensor(mask, dtype=torch.bool)
    weights = torch.tensor(weights, dtype=torch.float64)
    theta = torch.zeros(x.shape[-1], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        theta[0] = 2.  # Positive baseline preference: alternative centered is -1.
    optimizer = torch.optim.Adam([theta], lr=.03)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        risk = expected_risk(theta, x, losses, mask, weights)
        objective = risk+.001*theta.square().mean()
        objective.backward()
        optimizer.step()
        if step in (0, steps-1):
            history.append({"step": step+1, "risk": float(risk.detach())})
    return {"theta": theta.detach().numpy().tolist(), "scale": scale.tolist(), "history": history}


def choose(model, x, keep):
    logits = x/np.asarray(model["scale"]) @ np.asarray(model["theta"])
    return int(np.argmax(np.where(keep, logits, -np.inf)))


def project(context, labels):
    """Canonical 25 Hz output; no access to the truth values."""
    result = np.full(len(context.gt), 3, np.int8)
    for selected, mapping, output in context.parts:
        result[output] = labels[selected][mapping]
    return result
