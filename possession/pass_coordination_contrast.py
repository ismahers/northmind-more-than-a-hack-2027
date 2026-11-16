"""Select one reproducible harm/correction contrast from frozen event decisions."""
import hashlib
import numpy as np
from possession.pass_coordination_view import viewer as explanation_viewer


def review_records(data, key, spec):
    live = data["truth"] < 2
    supported = live & np.isfinite(data["c"][:, :3]).all(axis=(1, 2)) & np.isfinite(data["base_home"])
    home = np.clip(data["base_home"], spec["probability_clip"], 1-spec["probability_clip"])
    logits = np.log(home/(1-home))
    e = data["g"]*data["c"][:, 0]
    shift = spec["conditional_logit_weight"]*(e[:, 0]-e[:, 1])
    after = logits+shift
    base, full = (logits < 0).astype(int), (after < 0).astype(int)
    rows = []
    for index in np.flatnonzero(supported & (base != full)):
        role = "harm" if base[index] == data["truth"][index] else "correction"
        token = hashlib.sha256(f"{key}:{index}".encode()).hexdigest()[:12]
        rows.append({"id": token, "key": key, "event_index": int(index), "role": role,
                     "inspection_side": int(full[index]), "strength": float(abs(shift[index])),
                     "base_margin": float(abs(logits[index])), "base": int(base[index]),
                     "full": int(full[index]), "truth": int(data["truth"][index])})
    return rows


def choose_pair(rows):
    harms = sorted([r for r in rows if r["role"] == "harm"], key=lambda r: (r["strength"], r["id"]))
    corrections = [r for r in rows if r["role"] == "correction"]
    if not harms or not corrections:
        return []
    median = float(np.median([r["strength"] for r in harms]))
    harm = min(harms, key=lambda r: (abs(r["strength"]-median), r["id"]))
    good = min(corrections, key=lambda r: (r["inspection_side"] != harm["inspection_side"],
               r["key"] != harm["key"], abs(r["strength"]-harm["strength"])+abs(r["base_margin"]-harm["base_margin"]), r["id"]))
    pair = [harm, good]
    if int(harm["id"][:2], 16) % 2:
        pair.reverse()
    return pair


def replace_once(html, before, after):
    if html.count(before) != 1:
        raise ValueError("Explanation template contract changed")
    return html.replace(before, after)


def viewer(payload):
    html = explanation_viewer(payload)
    replacements = [
        ("Northmind · qué está midiendo la fórmula", "Northmind · contraste de la señal"),
        ("<h1>Qué está midiendo la fórmula</h1>", "<h1>Comparar la misma señal en dos jugadas</h1>"),
        ("Son los <b>dos casos de la primera pareja</b>, en el mismo orden. Esto explica una fórmula geométrica; no muestra decisiones de posesión del modelo principal.",
         "Una pareja seleccionada de los resultados train ya cerrados: un cambio creó un error H/A y otro lo corrigió. El orden está mezclado y el resultado se oculta. No sabemos si hay rebote o disputa: eso no viene etiquetado."),
        ("<b>Comprueba si R es el compañero que viste desmarcarse.</b> Pulsa «Siguiente giro candidato», elige el equipo y avanza despacio.",
         "<b>Observa qué diferencia hay entre las jugadas en la relación jugador-balón.</b> Reproduce desde el principio; «Ir al instante analizado» señala el centro del cambio ya registrado. Revela el resultado después de observar ambos casos."),
        ("<button id=\"previousEvent\">", "<button id=\"focus\">Ir al instante analizado</button><button id=\"previousEvent\">"),
        ("<div id=\"panel\"></div>", "<div id=\"panel\"></div><button id=\"revealOutcome\">Mostrar/ocultar resultado registrado</button><div id=\"outcome\" style=\"white-space:pre-line;background:#1d2a38;padding:12px\" hidden></div>"),
        ("Observación: ¿A y R corresponden a los jugadores que describías? ¿La flecha previa representa el desmarque que viste?",
         "¿Qué diferencia observable ves respecto al otro caso? Si XY no permite distinguirlos, indícalo. No asignes nuevas etiquetas H/A/D."),
        ("frame=0;$('seek').max", "frame=0;$('side').value=String(current.inspection_side);$('outcome').hidden=true;$('outcome').textContent=current.outcome;$('seek').max"),
        ("const e=f.explanation;let lines=[];", "const e=f.explanation;let lines=['Instante analizado: '+current.frames[current.focus_frame].t.toFixed(2)+' s'+(frame===current.focus_frame?' · ESTÁS AQUÍ':'')];"),
        (" · misma primera pareja", " · contraste de la señal"),
        ("$('case').onchange=chooseCase;", "$('focus').onclick=()=>{stop();frame=current.focus_frame;draw()};$('revealOutcome').onclick=()=>{$('outcome').hidden=!$('outcome').hidden};$('case').onchange=chooseCase;"),
    ]
    for before, after in replacements:
        html = replace_once(html, before, after)
    return html
