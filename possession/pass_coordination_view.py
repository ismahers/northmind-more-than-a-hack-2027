"""Explain the frozen formula for human local review, without scoring possessions."""
import json
import numpy as np
from possession.pass_coordination import event_features, player_motion, receivers


def explain(m, run, i, incoming, outgoing, spec):
    """Return the exact maximising receiver and projection used by the old score."""
    speeds = np.linalg.norm([incoming, outgoing], axis=1)
    if not np.isfinite(speeds).all() or speeds.min() <= 1e-12:
        return {"valid": False, "reason": "Sin velocidad continua del balón para explicar este instante."}
    cosine = float(np.clip(incoming@outgoing/np.prod(speeds), -1, 1))
    g, c, counts = event_features(m, run, i, cosine, outgoing, spec)
    prior, valid = player_motion(m, run, i, spec["receiver_prior_interval_s"], spec)
    distance = np.where(m.player_available[i], np.linalg.norm(m.players[i]-m.ball[i], axis=-1), np.inf)
    sides = []
    for side in (0, 1):
        actor = int(distance[side].argmin())
        if not np.isfinite(distance[side, actor]):
            sides.append({"valid": False, "reason": "No hay actor próximo con coordenadas."}); continue
        keep = valid[side] & m.player_available[i, side] & (distance[side] <= spec["receiver_max_current_distance_m"])
        keep[actor] = False
        ids = np.flatnonzero(keep)
        base = {"actor_slot": actor, "actor": m.players[i, side, actor].tolist(),
                "actor_distance_m": float(distance[side, actor]), "receiver_count": int(counts[side]),
                "g": float(g[side])}
        if not len(ids):
            sides.append(dict(base, valid=False, reason="Sin compañero con movimiento previo válido en el radio de la fórmula.")); continue
        if not np.isfinite(c[0, side]):
            sides.append(dict(base, valid=False, reason="La fórmula no tiene soporte simultáneo de los dos equipos.")); continue
        scores = [receivers(m.ball[i], m.players[i, side, j:j+1], prior[side, j:j+1], outgoing, spec) for j in ids]
        best = int(np.argmax(scores)); j = int(ids[best]); score = float(scores[best])
        if not np.isclose(score, c[0, side], rtol=1e-12, atol=1e-12):
            raise ValueError("Explanation does not reproduce frozen maximum")
        position, u = m.players[i, side, j], prior[side, j]
        relative = outgoing-u; square = float(relative@relative)
        lo, hi = spec["intercept_horizon_s"]
        tau = float(np.clip((position-m.ball[i])@relative/square if square > 1e-12 else lo, lo, hi))
        ball_end, receiver_end = m.ball[i]+tau*outgoing, position+tau*u
        a, b = [i+round(t*spec["native_hz"]) for t in spec["receiver_prior_interval_s"]]
        sides.append(dict(base, valid=True, receiver_slot=j, receiver=position.tolist(),
                          prior_start=m.players[a, side, j].tolist(), prior_end=m.players[b, side, j].tolist(),
                          ball_end=ball_end.tolist(), receiver_end=receiver_end.tolist(),
                          prior_speed_mps=float(np.linalg.norm(u)), tau_s=tau,
                          miss_m=float(np.linalg.norm(ball_end-receiver_end)), c=score,
                          evidence=float(g[side]*score),
                          tied_receivers=int(np.isclose(scores, score, rtol=1e-12, atol=1e-12).sum())))
    eligible = bool(speeds.min() >= spec["candidate_min_ball_speed_mps"]
                    and cosine <= np.cos(np.deg2rad(spec["candidate_min_turn_degrees"]))
                    and distance.min() <= spec["candidate_max_nearest_actor_m"])
    return {"valid": True, "eligible": eligible, "turn_degrees": float(np.rad2deg(np.arccos(cosine))), "sides": sides}


def viewer(payload):
    return HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))


HTML = r'''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Northmind · qué está midiendo la fórmula</title>
<style>body{font:16px system-ui;max-width:1100px;margin:22px auto;padding:0 16px;background:#101720;color:#eef3fa}button,select,input,textarea{font:inherit;padding:8px;margin:5px;border-radius:6px}button{cursor:pointer}p{line-height:1.5}canvas{width:100%;background:#153f32;border-radius:8px}#panel{background:#1d2a38;padding:12px;white-space:pre-line;line-height:1.5}textarea{box-sizing:border-box;width:100%;min-height:85px;background:#1d2a38;color:white}.dim{color:#bac8d8}.legend{display:flex;gap:16px;flex-wrap:wrap}</style>
<h1>Qué está midiendo la fórmula</h1>
<p>Son los <b>dos casos de la primera pareja</b>, en el mismo orden. Esto explica una fórmula geométrica; no muestra decisiones de posesión del modelo principal.</p>
<p><b>Comprueba si R es el compañero que viste desmarcarse.</b> Pulsa «Siguiente giro candidato», elige el equipo y avanza despacio.</p>
<label>Caso <select id="case"></select></label>
<label>Equipo que inspeccionar <select id="side"><option value="1" selected>Away · azul</option><option value="0">Home · rojo</option></select></label>
<button id="previousEvent">Giro candidato anterior</button><button id="nextEvent">Siguiente giro candidato</button>
<canvas id="pitch" width="1050" height="680"></canvas>
<button id="prev">← paso</button><button id="play">Reproducir</button><button id="next">paso →</button>
<input id="seek" type="range" min="0" value="0" style="width:43%"><span id="clock"></span>
<p class="legend"><span style="color:#ffd36b">Ⓐ posible pasador</span><span style="color:#7ce7f5">Ⓡ posible receptor</span><span style="color:#7ce7f5">→ movimiento previo medido</span><span style="color:#ffd36b">⇢ proyección del balón</span><span style="color:#e3a0ff">⇢ proyección del receptor</span></p>
<div id="panel"></div>
<p class="dim">La flecha celeste usa de 0,6 a 0,2 s antes de este instante. Las líneas discontinuas son proyecciones rectas de la fórmula, no trayectorias observadas ni intención. La velocidad saliente del balón usa los siguientes 0,12 s. Los números de slot no son identidades verificadas; no conocemos el sentido de ataque.</p>
<label>Observación: ¿A y R corresponden a los jugadores que describías? ¿La flecha previa representa el desmarque que viste?</label>
<textarea id="note" placeholder="Comenta la selección y el movimiento, sin asignar H/A/D ni corregir posesiones."></textarea>
<button id="save">Guardar observaciones locales</button>
<p class="dim">No hemos cambiado el extractor, entrenado o evaluado otra receta. El giro y la coordinación no prueban control: un desvío no cambia por sí solo posesión; en disputa continúa el último poseedor. D solo significa interrupción. Vista a 10 Hz; posiciones disponibles pueden contener estimación y ruido. No muestra contacto, postura o altura fiable.</p>
<small>Fuente: PySport IDSSE a715a38dfbaf5f58e431727c2b78d174101a703c, CC-BY-4.0. Offline, sin peticiones de red ni BAR-ATH. Casos para revisión humana local; no introducir sus coordenadas, capturas o archivos en el chat del agente.</small>
<script>const DATA=__PAYLOAD__;const $=id=>document.getElementById(id);let current,frame=0,timer=null;const memory={};
function getNote(id){try{return localStorage.getItem('northmind-explain-'+id)||memory[id]||''}catch{return memory[id]||''}}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Reproducir'}
function chooseCase(){stop();current=DATA.cases[+$('case').value];frame=0;$('seek').max=current.frames.length-1;$('note').value=getNote(current.id);draw()}
function arrow(g,a,b,color,dashed,X,Y){g.strokeStyle=color;g.fillStyle=color;g.lineWidth=3;g.setLineDash(dashed?[7,5]:[]);const ax=X(a[0]),ay=Y(a[1]),bx=X(b[0]),by=Y(b[1]);g.beginPath();g.moveTo(ax,ay);g.lineTo(bx,by);g.stroke();g.setLineDash([]);if(Math.hypot(bx-ax,by-ay)<1)return;const t=Math.atan2(by-ay,bx-ax);g.beginPath();g.moveTo(bx,by);g.lineTo(bx-10*Math.cos(t-.4),by-10*Math.sin(t-.4));g.lineTo(bx-10*Math.cos(t+.4),by-10*Math.sin(t+.4));g.closePath();g.fill()}
function marker(g,p,text,color,X,Y){g.strokeStyle=color;g.lineWidth=3;g.beginPath();g.arc(X(p[0]),Y(p[1]),12,0,Math.PI*2);g.stroke();g.fillStyle=color;g.font='bold 17px system-ui';g.fillText(text,X(p[0])+14,Y(p[1])-12)}
function draw(){const canvas=$('pitch'),g=canvas.getContext('2d'),f=current.frames[frame],L=current.pitch[0],W=current.pitch[1],side=+$('side').value;const X=x=>(x/L+.5)*canvas.width,Y=y=>(y/W+.5)*canvas.height;
g.fillStyle='#153f32';g.fillRect(0,0,canvas.width,canvas.height);g.strokeStyle='#9bbfa8';g.lineWidth=2;g.setLineDash([]);g.strokeRect(12,12,canvas.width-24,canvas.height-24);g.beginPath();g.moveTo(canvas.width/2,12);g.lineTo(canvas.width/2,canvas.height-12);g.stroke();g.beginPath();g.ellipse(canvas.width/2,canvas.height/2,9.15/L*canvas.width,9.15/W*canvas.height,0,0,Math.PI*2);g.stroke();
for(const p of f.players){g.fillStyle=p[2]===0?'#ff8585':'#80bdff';g.beginPath();g.arc(X(p[0]),Y(p[1]),6,0,Math.PI*2);g.fill()}
g.strokeStyle='#f7e89f';g.lineWidth=1.5;g.beginPath();let connected=false;for(let j=Math.max(0,frame-10);j<=frame;j++){const b=current.frames[j].ball;if(!b){connected=false;continue}if(connected)g.lineTo(X(b[0]),Y(b[1]));else g.moveTo(X(b[0]),Y(b[1]));connected=true}g.stroke();if(f.ball){g.fillStyle='white';g.beginPath();g.arc(X(f.ball[0]),Y(f.ball[1]),5,0,Math.PI*2);g.fill()}
const e=f.explanation;let lines=[];
if(!e.valid){lines.push(e.reason)}else{lines.push('Giro estimado: '+e.turn_degrees.toFixed(1)+'°');lines.push(e.eligible?'Este instante pasa el filtro geométrico de giro y proximidad.':'Este instante NO pasa el filtro geométrico; dibujar una proyección no implica que se detectase un pase.');
lines.push(current.original_diagnostic_scope?(f.selected?'Este centro entró en la muestra del diagnóstico cerrado.':'Este centro NO entró en la muestra del diagnóstico cerrado (filtro, separación de giros o límite de muestra).'):'Este caso no pertenece a los tres partidos del diagnóstico cerrado; aquí solo mostramos la misma fórmula.');
const s=e.sides[side];if(s.actor){marker(g,s.actor,'A · slot '+s.actor_slot,'#ffd36b',X,Y)}
if(!s.valid){lines.push(s.reason)}else{marker(g,s.receiver,'R · slot '+s.receiver_slot,'#7ce7f5',X,Y);arrow(g,s.prior_start,s.prior_end,'#7ce7f5',false,X,Y);arrow(g,f.ball,s.ball_end,'#ffd36b',true,X,Y);arrow(g,s.receiver,s.receiver_end,'#e3a0ff',true,X,Y);
g.strokeStyle='#e3a0ff';g.setLineDash([3,4]);g.beginPath();g.moveTo(X(s.ball_end[0]),Y(s.ball_end[1]));g.lineTo(X(s.receiver_end[0]),Y(s.receiver_end[1]));g.stroke();g.setLineDash([]);
lines.push('A es el más cercano de este equipo. R maximiza la fórmula entre '+s.receiver_count+' compañeros válidos; NO es un receptor confirmado.');if(s.tied_receivers>1)lines.push('Hay '+s.tied_receivers+' receptores empatados; se muestra el primer slot.');lines.push('Rapidez previa de R: '+s.prior_speed_mps.toFixed(2)+' m/s · proyección: '+s.tau_s.toFixed(2)+' s · separación prevista: '+s.miss_m.toFixed(2)+' m');lines.push('Giro/proximidad/espacio G: '+s.g.toFixed(3)+' · coordinación C: '+s.c.toFixed(3)+' · G×C: '+s.evidence.toFixed(3)+' (no es probabilidad de posesión).');}}
$('panel').textContent=lines.join('\n');$('clock').textContent='t = '+f.t.toFixed(2)+' s';$('seek').value=frame;
const eventFrames=current.frames.map((f,i)=>f.explanation.eligible?i:-1).filter(i=>i>=0);$('previousEvent').disabled=!eventFrames.some(i=>i<frame);$('nextEvent').disabled=!eventFrames.some(i=>i>frame);}
function eventStep(direction){stop();const indices=current.frames.map((f,i)=>f.explanation.eligible?i:-1).filter(i=>i>=0);const next=direction>0?indices.find(i=>i>frame):indices.filter(i=>i<frame).pop();if(next!==undefined){frame=next;draw()}}
DATA.cases.forEach((c,i)=>$('case').add(new Option('Caso '+(i+1)+' · misma primera pareja',i)));$('case').value='0';$('side').value='1';
$('case').onchange=chooseCase;$('side').onchange=draw;$('seek').oninput=()=>{stop();frame=+$('seek').value;draw()};$('prev').onclick=()=>{stop();frame=Math.max(0,frame-1);draw()};$('next').onclick=()=>{stop();frame=Math.min(current.frames.length-1,frame+1);draw()};$('nextEvent').onclick=()=>eventStep(1);$('previousEvent').onclick=()=>eventStep(-1);
$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='Pausar';timer=setInterval(()=>{if(frame>=current.frames.length-1){stop();return}frame++;draw()},100)};
$('note').oninput=()=>{memory[current.id]=$('note').value;try{localStorage.setItem('northmind-explain-'+current.id,$('note').value)}catch{}};
$('save').onclick=()=>{const notes=DATA.cases.map(c=>({case_id:c.id,observation:getNote(c.id)}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({purpose:'revisión de selección geométrica, sin etiquetas de posesión',notes},null,2)],{type:'application/json'}));a.download='northmind-revision-extractor.json';a.click();URL.revokeObjectURL(a.href)};
if(DATA.cases.length)chooseCase();else $('panel').textContent='No hay casos disponibles.';</script></html>'''
