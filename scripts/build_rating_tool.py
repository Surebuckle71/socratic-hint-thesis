"""
Build annotation/rate.html: a self-contained page for rating the 60 pilot items.

It reads annotation/annotation_sheet.csv (problem, reference answer, dialogue so far),
never the key file. Ratings autosave in the browser (localStorage); "Export CSV" downloads
a file in exactly the format scripts/analyze_annotation.py expects
(save it as annotation/annotation_sheet_A.csv).

    python scripts/build_rating_tool.py
"""

import csv
import json
from pathlib import Path

ANN = Path(__file__).resolve().parent.parent / "annotation"
SUB = ["problem_comprehension", "arithmetic_execution", "step_sequencing", "self_correction"]
DESC = {
    "problem_comprehension": "Does the student understand what the problem asks and which quantities matter?",
    "arithmetic_execution": "Does the student carry out calculations correctly?",
    "step_sequencing": "Does the student order the solution steps sensibly?",
    "self_correction": "Does the student notice and repair their own mistakes, or respond well when a mistake is pointed out?",
}

rows = list(csv.DictReader(open(ANN / "annotation_sheet.csv", encoding="utf-8-sig", newline="")))
items = [{"id": r["item_id"], "problem": r["problem"], "answer": r["reference_answer"].strip(), "dialogue": r["dialogue_so_far"]} for r in rows]
data = json.dumps({"items": items, "sub": SUB, "desc": DESC}).replace("</", "<\\/")

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Mastery rating tool</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --mute:#5b6577; --line:#dfe3ea; --acc:#2f5bd8; --ok:#1f8a4c; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, Segoe UI, sans-serif; }
  header { position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line); padding:10px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:5; }
  header h1 { font-size:16px; margin:0; }
  .bar { flex:1; min-width:160px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:var(--ok); width:0; transition:width .2s; }
  main { max-width:1100px; margin:18px auto; padding:0 16px; display:grid; grid-template-columns:1fr 340px; gap:16px; }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--mute); margin:0 0 6px; }
  .problem { margin-bottom:12px; } .answer { color:var(--mute); font-size:14px; white-space:pre-wrap; }
  .turn { padding:8px 10px; border-radius:8px; margin:6px 0; white-space:pre-wrap; }
  .tutor { background:#eef0f4; } .student { background:#e9f0ff; }
  .who { font-weight:600; font-size:12px; letter-spacing:.04em; display:block; color:var(--mute); }
  .row { padding:10px 0; border-top:1px solid var(--line); } .row:first-of-type { border-top:0; }
  .row.cur { background:#f3f6ff; margin:0 -10px; padding:10px; border-radius:8px; }
  .name { font-weight:600; } .d { color:var(--mute); font-size:13px; margin:2px 0 8px; }
  .btns { display:flex; gap:6px; flex-wrap:wrap; }
  button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:6px 10px; cursor:pointer; }
  button:hover { border-color:var(--acc); }
  button.sel { background:var(--acc); color:#fff; border-color:var(--acc); }
  button.na.sel { background:#6b7280; border-color:#6b7280; }
  .nav { display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; align-items:center; }
  .primary { background:var(--acc); color:#fff; border-color:var(--acc); }
  .hint { color:var(--mute); font-size:13px; }
  select { font:inherit; padding:5px; border-radius:8px; border:1px solid var(--line); }
  textarea { width:100%; box-sizing:border-box; font:inherit; border:1px solid var(--line); border-radius:8px; padding:6px; margin-top:8px; }
</style></head>
<body>
<header>
  <h1>Mastery rating</h1><span id="prog">0 of 60 complete</span><div class="bar"><div id="fill"></div></div>
  <select id="jump"></select>
  <button id="export" class="primary">Export CSV</button>
</header>
<main>
  <section class="card" id="left">
    <h2 id="itemtitle"></h2>
    <div class="problem"><h2>Problem</h2><div id="problem"></div></div>
    <div class="problem"><h2>Reference answer</h2><div class="answer" id="answer"></div></div>
    <h2>Dialogue so far (the tutor's next message is hidden)</h2>
    <div id="dialogue"></div>
  </section>
  <aside class="card" id="right">
    <h2>Rate the student's mastery at this point</h2>
    <div class="hint">1 = low (clear problem) &nbsp; 2 = medium (mixed) &nbsp; 3 = high (clear evidence) &nbsp; No evidence = leave blank. Judge the student, not the tutor.</div>
    <div id="rows"></div>
    <textarea id="notes" rows="2" placeholder="Notes (optional)"></textarea>
    <div class="nav"><button id="prev">&larr; Previous</button><button id="next" class="primary">Next &rarr;</button></div>
    <p class="hint">Keys: <b>1</b> <b>2</b> <b>3</b> rate the highlighted row, <b>0</b> = no evidence, <b>&uarr;</b>/<b>&darr;</b> move between rows, <b>&larr;</b>/<b>&rarr;</b> change item. Progress saves automatically in this browser.</p>
  </aside>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('payload').textContent);
  const KEY = 'mastery_rating_tool_v1';
  const S = JSON.parse(localStorage.getItem(KEY) || '{"r":{},"n":{},"i":0}');
  let cur = Math.min(S.i || 0, D.items.length - 1), row = 0;
  const save = () => { S.i = cur; localStorage.setItem(KEY, JSON.stringify(S)); };
  const get = (id, s) => (S.r[id] || {})[s];
  const complete = it => D.sub.every(s => get(it.id, s) !== undefined);
  function set(id, s, v) { (S.r[id] = S.r[id] || {})[s] = v; save(); draw(); }
  function drawDialogue(text) {
    const box = document.getElementById('dialogue'); box.textContent = '';
    text.split('\\n').forEach(line => {
      const m = line.match(/^\\[(TUTOR|STUDENT)\\]\\s*(.*)$/s);
      const div = document.createElement('div');
      if (m) { div.className = 'turn ' + m[1].toLowerCase(); const w = document.createElement('span'); w.className = 'who'; w.textContent = m[1]; div.append(w, document.createTextNode(m[2])); }
      else if (line.trim()) { div.className = 'turn'; div.textContent = line; } else return;
      box.append(div);
    });
  }
  function draw() {
    const it = D.items[cur];
    document.getElementById('itemtitle').textContent = 'Item ' + (cur + 1) + ' of ' + D.items.length;
    document.getElementById('problem').textContent = it.problem;
    document.getElementById('answer').textContent = it.answer;
    drawDialogue(it.dialogue);
    const rows = document.getElementById('rows'); rows.textContent = '';
    D.sub.forEach((s, k) => {
      const wrap = document.createElement('div'); wrap.className = 'row' + (k === row ? ' cur' : '');
      const n = document.createElement('div'); n.className = 'name'; n.textContent = s;
      const d = document.createElement('div'); d.className = 'd'; d.textContent = D.desc[s];
      const b = document.createElement('div'); b.className = 'btns';
      [[1, '1 Low'], [2, '2 Medium'], [3, '3 High'], ['', 'No evidence']].forEach(([v, label]) => {
        const x = document.createElement('button'); x.textContent = label; if (v === '') x.classList.add('na');
        if (get(it.id, s) === v) x.classList.add('sel');
        x.onclick = () => { row = k; set(it.id, s, v); };
        b.append(x);
      });
      wrap.append(n, d, b); wrap.onclick = () => { row = k; draw(); }; rows.append(wrap);
    });
    const notes = document.getElementById('notes'); notes.value = S.n[it.id] || '';
    notes.oninput = () => { S.n[it.id] = notes.value; save(); };
    const done = D.items.filter(complete).length;
    document.getElementById('prog').textContent = done + ' of ' + D.items.length + ' complete';
    document.getElementById('fill').style.width = (100 * done / D.items.length) + '%';
    const j = document.getElementById('jump');
    if (!j.options.length) D.items.forEach((x, i) => j.add(new Option(x.id, i)));
    D.items.forEach((x, i) => { j.options[i].text = x.id + (complete(x) ? ' \\u2713' : ''); });
    j.value = cur;
  }
  const go = i => { cur = Math.max(0, Math.min(D.items.length - 1, i)); row = 0; save(); draw(); window.scrollTo(0, 0); };
  document.getElementById('prev').onclick = () => go(cur - 1);
  document.getElementById('next').onclick = () => go(cur + 1);
  document.getElementById('jump').onchange = e => go(+e.target.value);
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    const it = D.items[cur];
    if (['1', '2', '3'].includes(e.key)) { set(it.id, D.sub[row], +e.key); row = Math.min(3, row + 1); draw(); }
    else if (e.key === '0') { set(it.id, D.sub[row], ''); row = Math.min(3, row + 1); draw(); }
    else if (e.key === 'ArrowDown') { row = Math.min(3, row + 1); draw(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { row = Math.max(0, row - 1); draw(); e.preventDefault(); }
    else if (e.key === 'ArrowRight') go(cur + 1);
    else if (e.key === 'ArrowLeft') go(cur - 1);
  });
  const q = v => '"' + String(v).replace(/"/g, '""') + '"';
  document.getElementById('export').onclick = () => {
    const undecided = D.items.reduce((n, it) => n + D.sub.filter(s => get(it.id, s) === undefined).length, 0);
    if (undecided && !confirm(undecided + ' rating cell(s) have not been decided yet and will be exported blank. Export anyway?')) return;
    const head = ['item_id', 'problem', 'reference_answer', 'dialogue_so_far', ...D.sub.map(s => s + '_rating'), 'notes'];
    const lines = [head.join(',')];
    D.items.forEach(it => lines.push([it.id, it.problem, it.answer, it.dialogue, ...D.sub.map(s => { const v = get(it.id, s); return v === undefined ? '' : v; }), S.n[it.id] || ''].map(q).join(',')));
    const blob = new Blob(['\\ufeff' + lines.join('\\r\\n')], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'annotation_sheet_A.csv'; a.click();
  };
  draw();
})();
</script></body></html>
"""

out = ANN / "rate.html"
out.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
print("wrote", out, "with", len(items), "items")
