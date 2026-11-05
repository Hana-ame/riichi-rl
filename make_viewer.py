"""把 sample_game.json 嵌进单文件 HTML 牌谱游览器。

python make_viewer.py --log rules/sample_game.json --out rules/game_viewer.html
"""
import argparse, json, os

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>牌谱游览器 · __TITLE__</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--txt:#e8eaf0;--muted:#9aa3b2;--hl:#f5c542}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}
#meta{color:var(--muted);font-size:12px}
#result{margin-left:auto;font-weight:600;color:var(--hl)}
.wrap{display:grid;grid-template-columns:1fr 320px;gap:12px;padding:12px 18px 24px}
.board{display:grid;grid-template-areas:"top top top" "left center right" "bottom bottom bottom";gap:10px;background:#11331f;border:2px solid #1d5c37;border-radius:10px;padding:12px;min-height:520px}
.seat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
.seat h3{margin:0 0 4px;font-size:13px;display:flex;gap:8px;align-items:center}
.seat .meta{color:var(--muted);font-size:11px}
.hand{display:flex;flex-wrap:wrap;gap:2px;margin:6px 0}
.tile{display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:32px;padding:0 2px;border-radius:4px;background:#f4f1e8;color:#222;font-weight:700;font-size:12px;border:1px solid #cfc8b8}
.tile.man{color:#1b5e20}.tile.pin{color:#0d47a1}.tile.sou{color:#b71c1c}.tile.honor{color:#4a148c}
.tile.small{min-width:18px;height:24px;font-size:10px}
.tile.last{outline:2px solid var(--hl);box-shadow:0 0 6px var(--hl)}
.river{display:flex;flex-wrap:wrap;gap:1px;margin:4px 0;min-height:26px}
.melds{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}
.meld{display:flex;gap:1px;border:1px dashed #4a5568;padding:1px;border-radius:4px}
.center{grid-area:center;background:#0d2a1a;border:1px solid #1d5c37;border-radius:8px;padding:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.center .big{font-size:22px;font-weight:700;color:var(--hl)}
.center .row{display:flex;gap:14px;color:var(--muted);font-size:12px;flex-wrap:wrap;justify-content:center}
.side{display:flex;flex-direction:column;gap:10px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
.controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
button{background:#26303f;color:var(--txt);border:1px solid #3a4757;border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{background:#324055}
button:disabled{opacity:.4;cursor:default}
input[type=range]{width:100%}
#event{font-size:13px;min-height:44px}
#event b{color:var(--hl)}
.log{max-height:300px;overflow:auto;font-size:12px;border-top:1px solid var(--line);margin-top:8px;padding-top:6px}
.log .row{padding:3px 4px;border-radius:4px;cursor:pointer;display:flex;gap:6px}
.log .row:hover{background:#232b38}
.log .row.cur{background:#3a3418;outline:1px solid var(--hl)}
.log .idx{color:var(--muted);min-width:28px}
.tag{font-size:10px;padding:1px 5px;border-radius:8px;background:#2b3648;color:#b9c4d6}
.tag.ron{background:#4a1d1d;color:#ff8a8a}.tag.tsumo{background:#1d4a2a;color:#8affb0}.tag.ryu{background:#333;color:#bbb}
.toggle{font-size:12px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>🀄 牌谱游览器</h1>
  <div id="meta"></div>
  <div id="result"></div>
  <a id="jsonlink" style="color:#9aa3b2;font-size:12px" download>下载牌谱 JSON</a>
</header>
<div class="wrap">
  <div class="board" id="board">
    <div class="seat" id="seat-top" style="grid-area:top"></div>
    <div class="seat" id="seat-left" style="grid-area:left"></div>
    <div class="center" id="center"></div>
    <div class="seat" id="seat-right" style="grid-area:right"></div>
    <div class="seat" id="seat-bottom" style="grid-area:bottom"></div>
  </div>
  <div class="side">
    <div class="panel">
      <div class="controls">
        <button id="first">⏮</button><button id="prev">◀</button>
        <button id="play">▶ 播放</button><button id="next">▶</button><button id="last">⏭</button>
      </div>
      <div style="margin:8px 0 2px;display:flex;justify-content:space-between;font-size:12px;color:var(--muted)">
        <span id="stepinfo">—</span><span class="toggle"><label><input type="checkbox" id="reveal" checked> 显示四家手牌</label></span>
      </div>
      <input type="range" id="slider" min="0" max="1" value="0">
      <div id="event">—</div>
      <div class="log" id="log"></div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const SEAT_POS = ["bottom","right","top","left"]; // 座位0..3 顺时针：下/右/上/左
const TILES = {m:["m1","m2","m3","m4","m5","m6","m7","m8","m9"],
               p:["p1","p2","p3","p4","p5","p6","p7","p8","p9"],
               s:["s1","s2","s3","s4","s5","s6","s7","s8","s9"],
               z:["z1","z2","z3","z4","z5","z6","z7"]};
let idx=0, timer=null, reveal=true;
const maxIdx = DATA.steps.length;
const $ = id=>document.getElementById(id);

const HONOR={E:"東",S:"南",W:"西",N:"北",P:"白",F:"發",C:"中"};
function tileHTML(name, small, last){
  if(!name) return "";
  const suit=name[0];
  const cls = suit==="m"?"man":suit==="p"?"pin":suit==="s"?"sou":"honor";
  const label = HONOR[name] || name;
  return `<span class="tile ${cls} ${small?"small":""} ${last?"last":""}">${label}</span>`;
}
function handHTML(hand, small, lastTile){
  let out="";
  hand.forEach((t,i)=> out+=tileHTML(t, small, lastTile && t===lastTile && i===hand.length-1));
  return out;
}
function riverHTML(river, lastTile){ return river.map((t,i)=>tileHTML(t,true, (i===river.length-1)&&t===lastTile)).join(""); }
function meldHTML(melds){
  return melds.map(m=>`<span class="meld">${m.tiles.map(t=>tileHTML(t,true)).join("")}</span>`).join("");
}
function snapAt(i){ return i===0 ? DATA.initial : DATA.steps[i-1].snap; }
function eventsAt(i){ return i===0 ? [] : DATA.steps[i-1].events; }

function render(){
  const snap = snapAt(idx);
  const evs = eventsAt(idx);
  $("slider").value = idx;
  $("stepinfo").textContent = `第 ${idx} / ${maxIdx} 步`;
  $("event").innerHTML = evs.length
    ? evs.map(e=>`<b>座位${e.seat}</b> ${e.kind}${e.tile? " "+e.tile:""}${e.is_call?" <span class=tag>叫牌</span>":""}`).join("；")
    : "开局（发牌完成）";
  // center
  const dora = snap.dora.map(t=>tileHTML(t,false)).join("");
  $("center").innerHTML = `<div class="big">${snap.round.wind} ${snap.round.kyoku+1} 局 · 第${snap.turn}巡</div>
    <div class="row"><span>宝牌指示 <b>${dora}</b></span><span>余牌 ${snap.wall}</span><span>本场 ${snap.round.honba}</span></div>`;
  // seats
  [0,1,2,3].forEach(seat=>{
    const pos = SEAT_POS[seat];
    const p = snap.players[seat];
    const el = $("seat-"+pos);
    if(!el) return;
    const lastTile = (evs.find(e=>e.seat===seat && e.kind==="DISCARD")||{}).tile;
    const hideHand = (!reveal && seat!==0);
    el.innerHTML = `<h3>座位${seat} · ${p.seat}${p.dealer?" · 庄":""}${p.riichi?' <span class="tag ron">立直</span>':""}
      <span class="meta">${p.points} 点</span></h3>
      <div class="hand">${hideHand? `<span class="meta">手牌隐藏（${p.hand.length} 张）</span>` : handHTML(p.hand,false,lastTile)}</div>
      <div class="melds">${meldHTML(p.melds)}</div>
      <div class="river">${riverHTML(p.river,lastTile)}</div>`;
  });
  // log highlight
  document.querySelectorAll(".log .row").forEach(r=>r.classList.toggle("cur", +r.dataset.i===idx));
  const cur = document.querySelector(`.log .row[data-i="${idx}"]`);
  if(cur) cur.scrollIntoView({block:"nearest"});
  $("prev").disabled = idx===0; $("first").disabled = idx===0;
  $("next").disabled = idx===maxIdx; $("last").disabled = idx===maxIdx;
}
function buildLog(){
  const log=$("log");
  log.innerHTML = "";
  const add=(i,evs)=>{
    const row=document.createElement("div");
    row.className="row"; row.dataset.i=i;
    const txt = evs.length? evs.map(e=>`座${e.seat} ${e.kind}${e.tile?" "+e.tile:""}`).join("；") : "开局";
    row.innerHTML=`<span class="idx">${i}</span><span>${txt}</span>`;
    row.onclick=()=>{ stop(); idx=i; render(); };
    log.appendChild(row);
  };
  add(0, []);
  DATA.steps.forEach((s,i)=> add(i+1, s.events));
}
function play(){ if(timer) return; $("play").textContent="⏸ 暂停";
  timer=setInterval(()=>{ if(idx>=maxIdx){stop();return;} idx++; render(); }, 700); }
function stop(){ if(timer){clearInterval(timer); timer=null;} $("play").textContent="▶ 播放"; }
$("first").onclick=()=>{stop();idx=0;render();};
$("prev").onclick=()=>{stop();idx=Math.max(0,idx-1);render();};
$("next").onclick=()=>{stop();idx=Math.min(maxIdx,idx+1);render();};
$("last").onclick=()=>{stop();idx=maxIdx;render();};
$("play").onclick=()=> timer?stop():play();
$("slider").max=maxIdx; $("slider").oninput=e=>{stop();idx=+e.target.value;render();};
$("reveal").onchange=e=>{reveal=e.target.checked;render();};
document.addEventListener("keydown",e=>{ if(e.key==="ArrowLeft") $("prev").onclick(); if(e.key==="ArrowRight") $("next").onclick(); });
$("meta").textContent = `签：${DATA.meta.ckpt} · seed ${DATA.meta.seed} · ${DATA.steps.length} 步`;
$("jsonlink").href = "__LOGFILE__";
const r = DATA.result || {};
const names={ron:"荣和",tsumo:"自摸",ryuukyoku:"流局"};
$("result").innerHTML = r.type ? `${names[r.type]||r.type}${r.winner!=null?" · 座位"+r.winner+" 和牌":""}${r.type==="ron"&&r.from!=null?"（放铳：座位"+r.from+"）":""}` : "—";
buildLog(); render();
</script>
</body>
</html>
"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log", default="rules/sample_game.json")
    ap.add_argument("--out", default="rules/game_viewer.html")
    a=ap.parse_args()
    data=json.load(open(a.log, encoding="utf-8"))
    html=HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__TITLE__", os.path.basename(a.log)).replace("__LOGFILE__", os.path.basename(a.log))
    open(a.out,"w",encoding="utf-8").write(html)
    print("wrote", a.out, "size", os.path.getsize(a.out))

if __name__=="__main__":
    main()
