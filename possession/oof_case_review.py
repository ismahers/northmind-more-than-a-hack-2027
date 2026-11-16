"""Operational error families and reproducible paired local review, no detector fitting."""
import hashlib
import json
import numpy as np
from possession.long_diagnosis import GlobalMatcher
from possession.football_mechanism import lift

FAMILIES = ("decoder_loss", "no_team_evidence", "weak_team_evidence", "low_alive", "extent_or_alignment")


def analyze(context, p, active, raw, hmm, identity, spec):
    matcher = GlobalMatcher(context.pm, context.gt)
    raw_match, final_match = matcher.score(raw, verify=True), matcher.score(hmm, verify=True)
    alive = p[:, :2].sum(-1)
    home = np.divide(p[:, 0], alive, out=np.full(len(p), .5), where=alive > 1e-12)
    native_home, native_alive, support = (lift(context, x) for x in (home, alive, active))
    rows = []
    for i, (a, b, team) in enumerate(matcher.truth):
        if b-a >= 25*spec["short_seconds"] or a == 0 or b >= len(context.gt):
            continue
        other = 1-team
        if context.gt[a-1] != other or context.gt[b] != other:
            continue  # only H-A-H / A-H-A, not dead-ball borders or censored cases
        correct = native_home[a:b] if team == 0 else 1-native_home[a:b]
        fraction = float((correct > .5).mean())
        alive_fraction = float((native_alive[a:b] >= .5).mean())
        recovered = bool(final_match.flags[2, i])
        raw_recovered = bool(raw_match.flags[2, i])
        family = ("recovered" if recovered else "decoder_loss" if raw_recovered
                  else "no_team_evidence" if not (correct > .5).any()
                  else "weak_team_evidence" if fraction < spec["majority"]
                  else "low_alive" if alive_fraction < spec["majority"] else "extent_or_alignment")
        swallowed = bool(not recovered and (hmm[a-1:b+1] == other).all())
        token = hashlib.sha256((identity+":"+str(i)).encode()).hexdigest()[:12]
        rows.append({"id": token, "identity": identity, "spell": i, "start": a, "stop": b,
            "team": team, "duration_s": (b-a)/25, "family": family, "recovered": recovered,
            "raw_recovered": raw_recovered, "fully_swallowed": swallowed,
            "correct_fraction": fraction, "alive_fraction": alive_fraction,
            "coverage": float((support[a:b] > .5).mean())})
    return rows


def aggregate(rows):
    missed = [r for r in rows if not r["recovered"]]
    return {"target_short_spells": len(rows), "recovered": len(rows)-len(missed),
        "missed": len(missed), "fully_swallowed": sum(r["fully_swallowed"] for r in rows),
        "missed_with_review_coverage": sum(r["coverage"] >= .8 for r in missed),
        "families": {f: {"missed": sum(r["family"] == f for r in missed),
            "fully_swallowed": sum(r["family"] == f and r["fully_swallowed"] for r in missed)} for f in FAMILIES}}


def choose_pairs(rows, spec):
    controls = [r for r in rows if r["recovered"] and r["coverage"] >= spec["review_coverage_min"]]
    used = set(); pairs = []
    ranked = sorted(FAMILIES, key=lambda f: (-sum(r["family"] == f and r["fully_swallowed"] for r in rows), FAMILIES.index(f)))
    for family in ranked:
        errors = [r for r in rows if r["family"] == family and r["fully_swallowed"]
                  and r["coverage"] >= spec["review_coverage_min"]]
        if not errors:
            continue
        median = np.median([r["duration_s"] for r in errors])
        error = min(errors, key=lambda r: (abs(r["duration_s"]-median), r["id"]))
        available = [r for r in controls if r["id"] not in used]
        if not available:
            break
        # Shared fold, same frontier type by construction; match duration and XY support.
        control = min(available, key=lambda r: (abs(r["duration_s"]-error["duration_s"])/3
                        +abs(r["coverage"]-error["coverage"]), r["id"]))
        used.add(control["id"])
        order = [error, control]
        if int(error["id"][:2], 16) % 2:
            order.reverse()
        pairs.append({"family": family, "cases": order})
        if len(pairs) >= spec["pairs_per_fold_max"]:
            break
    return pairs


def viewer(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return VIEWER.replace("__PAYLOAD__", encoded)


VIEWER = r'''<!doctype html><html lang="es"><meta charset="utf-8">
<title>Northmind · revisión futbolística OOF</title>
<style>body{font:16px system-ui;background:#101720;color:#eef3fa;max-width:1100px;margin:25px auto;padding:0 18px}button,select,input,textarea{font:inherit;margin:5px;padding:8px;border-radius:6px}canvas{width:100%;background:#153f32;border-radius:10px}textarea{box-sizing:border-box;width:100%;height:95px;background:#1c2937;color:white}small,p{line-height:1.5}#truth{white-space:pre-wrap;background:#1c2937;padding:12px}button{cursor:pointer}.dim{color:#b8c5d5}label{display:block;margin-top:12px}</style>
<h1>Northmind · casos OOF para conversar sobre fútbol</h1>
<p>Tracking público IDSSE. Cada pareja contiene un fallo y un acierto comparables.
Primero observa el movimiento; revela las etiquetas después. No estamos creando anotaciones H/A/D.</p>
<select id="pair"></select><select id="case"></select>
<canvas id="pitch" width="1050" height="680"></canvas>
<button id="prev">← frame</button><button id="play">Reproducir</button><button id="next">frame →</button>
<input id="seek" type="range" min="0" value="0" style="width:45%"><span id="clock"></span>
<p><b style="color:#ff8585">● H</b> &nbsp; <b style="color:#80bdff">● A</b> &nbsp; ⚪ balón.
Los puntos representan tracks, no identidades verificadas. No se conoce el sentido de ataque.</p>
<button id="reveal">Mostrar/ocultar etiquetas y diagnóstico</button><div id="truth" hidden></div>
<label>¿Qué comportamiento observable distingue el control de la persecución o disputa?</label><textarea id="observable"></textarea>
<label>¿Qué no puedes determinar con estas coordenadas?</label><textarea id="unknown"></textarea>
<label>¿Qué diferencia ves respecto al otro caso de la pareja?</label><textarea id="contrast"></textarea>
<button id="save">Guardar observaciones como JSON local</button>
<p class="dim">Vista discreta a 10 Hz, sin interpolación de huecos ni contactos detectados. XY disponible puede proceder de estimación del proveedor. No muestra postura, piernas, intención ni altura fiable. Las etiquetas son de IDSSE público; la definición del reto exige control, conserva el último poseedor en disputa y usa D solo para interrupciones. Tus observaciones no cambian etiquetas ni entrenan un modelo.</p>
<small>Fuente: PySport IDSSE, revisión a715a38dfbaf5f58e431727c2b78d174101a703c, CC-BY-4.0. Artefacto offline, sin peticiones de red. No contiene BAR-ATH.</small>
<script>const DATA=__PAYLOAD__;const $=id=>document.getElementById(id);let frame=0,timer=null,current=null;
const memory={};function getNote(k){try{return localStorage.getItem(k)||memory[k]||''}catch{return memory[k]||''}}
DATA.pairs.forEach((p,i)=>$('pair').add(new Option('Pareja '+(i+1)+' · fold '+p.fold,i)));
function choosePair(){stop();$('case').replaceChildren();DATA.pairs[$('pair').value].cases.forEach((c,i)=>$('case').add(new Option('Caso '+(i+1)+' · '+c.id,i)));chooseCase()}
function chooseCase(){stop();current=DATA.pairs[$('pair').value].cases[$('case').value];frame=0;$('seek').max=current.frames.length-1;$('truth').hidden=true;for(const k of ['observable','unknown','contrast'])$(k).value=getNote(current.id+k);draw()}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Reproducir'}
function draw(){const c=$('pitch'),g=c.getContext('2d'),f=current.frames[frame],L=current.pitch[0],W=current.pitch[1];const X=x=>(x/L+.5)*c.width,Y=y=>(y/W+.5)*c.height;
g.fillStyle='#153f32';g.fillRect(0,0,c.width,c.height);g.strokeStyle='#a9cbb5';g.lineWidth=2;g.strokeRect(12,12,c.width-24,c.height-24);g.beginPath();g.moveTo(c.width/2,12);g.lineTo(c.width/2,c.height-12);g.stroke();g.beginPath();g.ellipse(c.width/2,c.height/2,9.15/L*c.width,9.15/W*c.height,0,0,7);g.stroke();
for(const p of f.players){g.fillStyle=p[2]===0?'#ff8585':'#80bdff';g.beginPath();g.arc(X(p[0]),Y(p[1]),6,0,7);g.fill()}
g.strokeStyle='#f7e89f';g.beginPath();let connected=false;for(let j=Math.max(0,frame-10);j<=frame;j++){const b=current.frames[j].ball;if(!b){connected=false;continue}if(connected)g.lineTo(X(b[0]),Y(b[1]));else g.moveTo(X(b[0]),Y(b[1]));connected=true}g.stroke();if(f.ball){g.fillStyle='white';g.strokeStyle='#101720';g.beginPath();g.arc(X(f.ball[0]),Y(f.ball[1]),5,0,7);g.fill();g.stroke()}
$('seek').value=frame;$('clock').textContent='t = '+f.t.toFixed(2)+' s';$('truth').textContent='Caso '+current.id+' · '+current.role+'\nFamilia: '+current.family+'\nDuración central: '+current.duration_s.toFixed(2)+' s\nGT: '+f.gt+' · crudo: '+f.raw+' · HMM: '+f.hmm+'\nP(H|vivo): '+f.home.toFixed(3)+' · P(vivo): '+f.alive.toFixed(3)+'\nVentana central GT: '+current.central.map(x=>x.toFixed(2)).join(' a ')+' s';}
$('pair').onchange=choosePair;$('case').onchange=chooseCase;$('seek').oninput=()=>{frame=+$('seek').value;draw()};$('prev').onclick=()=>{stop();frame=Math.max(0,frame-1);draw()};$('next').onclick=()=>{stop();frame=Math.min(current.frames.length-1,frame+1);draw()};$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='Pausar';timer=setInterval(()=>{if(frame>=current.frames.length-1){stop();return}frame++;draw()},100)};$('reveal').onclick=()=>{$('truth').hidden=!$('truth').hidden};
for(const k of ['observable','unknown','contrast'])$(k).oninput=()=>{memory[current.id+k]=$(k).value;try{localStorage.setItem(current.id+k,$(k).value)}catch{}};
$('save').onclick=()=>{const notes=[];for(const p of DATA.pairs)for(const c of p.cases){const item={case_id:c.id};for(const k of ['observable','unknown','contrast'])item[k]=getNote(c.id+k);notes.push(item)}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({purpose:'observaciones cualitativas, sin nuevas etiquetas',notes},null,2)],{type:'application/json'}));a.download='northmind-observaciones-locales.json';a.click();URL.revokeObjectURL(a.href)};
if(DATA.pairs.length)choosePair();else document.body.append('No hay parejas que cumplan el soporte mínimo.');</script></html>'''
