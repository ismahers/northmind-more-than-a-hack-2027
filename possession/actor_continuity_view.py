"""Cached-clip visual measurement only. No possession classifier or thresholds."""
from possession.pass_coordination_contrast import viewer as contrast_viewer
from possession.pass_coordination_contrast import replace_once


def viewer(payload):
    html = contrast_viewer(payload)
    html = replace_once(html, "<h1>Comparar la misma señal en dos jugadas</h1>", "<h1>¿El balón sigue con el mismo jugador?</h1>")
    html = replace_once(html, '<title>Northmind · contraste de la señal</title>', '<title>Northmind · continuidad jugador–balón</title>')
    html = replace_once(html, '<h1>¿El balón sigue con el mismo jugador?</h1>', '<style>#panel,#revealOutcome,#outcome,#focus,.legend{display:none!important}</style><h1>¿El balón sigue con el mismo jugador?</h1>')
    html = replace_once(html,
        "Una pareja seleccionada de los resultados train ya cerrados: un cambio creó un error H/A y otro lo corrigió. El orden está mezclado y el resultado se oculta. No sabemos si hay rebote o disputa: eso no viene etiquetado.",
        "Tres recortes ya disponibles: el ejemplo anterior y los dos contrastes. Vamos a medir distancia al mismo slot, sin decidir si hay pase, conducción o posesión.")
    html = replace_once(html,
        "<b>Observa qué diferencia hay entre las jugadas en la relación jugador-balón.</b> Reproduce desde el principio; «Ir al instante analizado» señala el centro del cambio ya registrado. Revela el resultado después de observar ambos casos.",
        "Elige el equipo del jugador que quieres seguir, pausa en el giro o golpeo y pulsa <b>Fijar jugador de referencia aquí</b>. La gráfica seguirá a ese mismo slot después. La referencia inicial es automática: comprueba que coincide con la acción que quieres mirar.")
    html = replace_once(html, '<div id="panel"></div>', '<div id="panel"></div>'+SECTION)
    html = replace_once(html, "new Option('Caso '+(i+1)+' · contraste de la señal',i)", "new Option(c.label,i)")
    html = replace_once(html, "if(DATA.cases.length)chooseCase();", INSTRUMENT+"\nif(DATA.cases.length)chooseCase();")
    return html


SECTION = '''<section style="background:#1d2a38;padding:12px;margin-top:12px;border-radius:8px">
<h2>Distancia al jugador fijado</h2><button id="anchorActor">Fijar jugador de referencia aquí</button>
<button id="returnAnchor">Volver a la referencia</button>
<p id="referenceInfo"></p><canvas id="distancePlot" width="1050" height="250" style="background:#14222f"></canvas>
<p style="color:#80efbd">Línea verde: distancia del balón al mismo slot. Línea vertical: instante de referencia; cursor blanco: reproducción actual. Los huecos no se unen.</p>
<p>Se mide de −0,8 a +1,5 s alrededor de la referencia, sin afirmar un límite de control. Una conducción rápida puede separar el balón momentáneamente; un pase fallado puede haber sido controlado. La curva no cambia ninguna etiqueta. El slot fijado no certifica una identidad de persona.</p></section>'''


INSTRUMENT = r'''
let actorReference=null;
function distanceToFixed(f,reference){if(!f.ball)return null;const p=f.players.find(p=>p[2]===reference.side&&p[3]===reference.slot);return p?Math.hypot(p[0]-f.ball[0],p[1]-f.ball[1]):null}
function setActorReference(at){const f=current.frames[at],side=+$('side').value;if(!f.ball){actorReference=null;return}
const players=f.players.filter(p=>p[2]===side);if(!players.length){actorReference=null;return}
let best=players[0],bestDistance=Math.hypot(best[0]-f.ball[0],best[1]-f.ball[1]);
for(const p of players.slice(1)){const d=Math.hypot(p[0]-f.ball[0],p[1]-f.ball[1]);if(d<bestDistance){best=p;bestDistance=d}}
actorReference={side,slot:best[3],index:at,time:f.t};}
function drawFixedPitch(){const c=$('pitch'),g=c.getContext('2d'),f=current.frames[frame],L=current.pitch[0],W=current.pitch[1],X=x=>(x/L+.5)*c.width,Y=y=>(y/W+.5)*c.height;
g.fillStyle='#153f32';g.fillRect(0,0,c.width,c.height);g.strokeStyle='#9bbfa8';g.lineWidth=2;g.setLineDash([]);g.strokeRect(12,12,c.width-24,c.height-24);g.beginPath();g.moveTo(c.width/2,12);g.lineTo(c.width/2,c.height-12);g.stroke();
for(const p of f.players){g.fillStyle=p[2]===0?'#ff8585':'#80bdff';g.beginPath();g.arc(X(p[0]),Y(p[1]),6,0,Math.PI*2);g.fill()}
g.strokeStyle='#f7e89f';g.lineWidth=1.5;g.beginPath();let previous=null;for(let j=Math.max(0,frame-10);j<=frame;j++){const q=current.frames[j];if(!q.ball){previous=null;continue}if(previous&&q.t-previous.t<=.150001)g.lineTo(X(q.ball[0]),Y(q.ball[1]));else g.moveTo(X(q.ball[0]),Y(q.ball[1]));previous=q}g.stroke();if(f.ball){g.fillStyle='white';g.beginPath();g.arc(X(f.ball[0]),Y(f.ball[1]),5,0,Math.PI*2);g.fill()}}
function drawContinuity(){drawFixedPitch();const plot=$('distancePlot'),g=plot.getContext('2d');g.fillStyle='#14222f';g.fillRect(0,0,plot.width,plot.height);
if(!actorReference){$('referenceInfo').textContent='Sin referencia válida: elige un instante con balón y jugador disponibles.';return}
const ref=actorReference,rows=current.frames.map(f=>({t:f.t-ref.time,d:distanceToFixed(f,ref)})).filter(r=>r.t>=-.80001&&r.t<=1.50001);
const values=rows.filter(r=>r.d!==null).map(r=>r.d),max=Math.max(1,...values)*1.12;
const X=t=>45+(t+.8)/2.3*(plot.width-65),Y=d=>plot.height-35-d/max*(plot.height-60);
g.strokeStyle='#72859a';g.lineWidth=1;g.setLineDash([]);g.beginPath();g.moveTo(45,20);g.lineTo(45,plot.height-35);g.lineTo(plot.width-20,plot.height-35);g.stroke();g.font='14px system-ui';g.fillStyle='#c6d4e4';
g.fillText(max.toFixed(1)+' m',4,18);g.fillText('0 m',8,plot.height-30);for(const t of [-.8,0,.5,1,1.5])g.fillText(t.toFixed(1)+' s',X(t)-15,plot.height-10);
g.strokeStyle='#80efbd';g.lineWidth=3;g.beginPath();let previous=null;for(const r of rows){if(r.d===null){previous=null;continue}if(previous&&r.t-previous.t<=.150001)g.lineTo(X(r.t),Y(r.d));else g.moveTo(X(r.t),Y(r.d));previous=r}g.stroke();
g.strokeStyle='#80efbd';g.setLineDash([5,4]);g.beginPath();g.moveTo(X(0),20);g.lineTo(X(0),plot.height-35);g.stroke();g.setLineDash([]);
const now=current.frames[frame].t-ref.time;if(now>=-.8&&now<=1.5){g.strokeStyle='white';g.beginPath();g.moveTo(X(now),20);g.lineTo(X(now),plot.height-35);g.stroke()}
const currentDistance=distanceToFixed(current.frames[frame],ref);
$('referenceInfo').textContent='Referencia: '+(ref.side===0?'Home':'Away')+' · slot '+ref.slot+' · t = '+ref.time.toFixed(2)+' s. Distancia actual: '+(currentDistance===null?'sin observación':currentDistance.toFixed(2)+' m')+'. Se conserva el slot aunque otro sea más cercano.';
const c=$('pitch'),f=current.frames[frame],p=f.players.find(p=>p[2]===ref.side&&p[3]===ref.slot);if(p){const h=c.getContext('2d'),L=current.pitch[0],W=current.pitch[1];marker(h,p,'A0 · referencia','#80efbd',x=>(x/L+.5)*c.width,y=>(y/W+.5)*c.height)}
}
const originalDraw=draw;draw=function(){originalDraw();drawContinuity()};
const originalChooseCase=chooseCase;chooseCase=function(){actorReference=null;originalChooseCase();setActorReference(current.focus_frame);draw()};
$('case').onchange=()=>chooseCase();$('side').onchange=()=>{setActorReference(frame);draw()};
$('anchorActor').onclick=()=>{stop();setActorReference(frame);draw()};$('returnAnchor').onclick=()=>{if(actorReference){stop();frame=actorReference.index;draw()}};
'''
