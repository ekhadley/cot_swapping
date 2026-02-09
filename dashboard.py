#!./.venv/bin/python
import glob
import json
import os
import re

from flask import Flask, jsonify

from tracker import Tracker
from utils import load_jsonl

app = Flask(__name__)

MIN_ACCURACY = 0.1
MAX_ACCURACY = 0.8


def _get_tracker():
    db_path = "data/eval.db"
    if os.path.exists(db_path):
        return Tracker(db_path)
    return None


def _load_jsonl_fallback():
    """Load data from JSONL files (backward compat)."""
    results = {}
    for path in sorted(glob.glob("data/raw_results_*.jsonl")):
        match = re.search(r"raw_results_(\w+)\.jsonl", path)
        if not match:
            continue
        label = match.group(1)
        results[label] = load_jsonl(path)
    return results


@app.route("/api/data")
def api_data():
    tracker = _get_tracker()
    if tracker:
        data = tracker.get_all_data()
        counts = tracker.get_live_counts()
        # Merge JSONL data for labels not in SQLite
        jsonl_data = _load_jsonl_fallback()
        for label, problems in jsonl_data.items():
            if label not in data:
                data[label] = problems
        return jsonify({"problems": data, "counts": counts})
    # Pure JSONL fallback
    return jsonify({"problems": _load_jsonl_fallback(), "counts": {}})


@app.route("/")
def index():
    return PAGE_HTML


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CoT Swap Eval</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a0c;
  --surface: #111114;
  --surface2: #18181c;
  --border: #252529;
  --border-hi: #35353b;
  --text: #c8c8d0;
  --text-dim: #6e6e7a;
  --text-bright: #e8e8f0;
  --accent: #5ce0d8;
  --accent-dim: #2a6b66;
  --green: #4ade80;
  --green-bg: rgba(74, 222, 128, 0.08);
  --red: #f87171;
  --red-bg: rgba(248, 113, 113, 0.08);
  --orange: #fb923c;
  --orange-bg: rgba(251, 146, 60, 0.08);
  --blue: #60a5fa;
  --blue-bg: rgba(96, 165, 250, 0.08);
  --gray-badge: #6e6e7a;
  --gray-badge-bg: rgba(110, 110, 122, 0.08);
  --mono: 'IBM Plex Mono', 'JetBrains Mono', monospace;
}

html { font-size: 13px; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  min-height: 100vh;
  overflow-x: hidden;
}

/* grain overlay */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
}

.shell {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}

/* header */
.hdr {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
}
.hdr h1 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.hdr .sub {
  font-size: 0.85rem;
  color: var(--text-dim);
  font-weight: 300;
}
.hdr .pulse {
  margin-left: auto;
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }
}

/* live counters */
.live-bar {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  font-size: 0.75rem;
  color: var(--text-dim);
  letter-spacing: 0.04em;
}
.live-bar .cnt {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
.live-bar .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  display: inline-block;
}
.dot.in-flight { background: var(--blue); animation: blink 1s ease-in-out infinite; }
.dot.pending { background: var(--gray-badge); }
.dot.done { background: var(--green); }
.dot.err { background: var(--red); }

/* tabs */
.tabs {
  display: flex;
  gap: 0;
  margin-bottom: 1.5rem;
}
.tab {
  padding: 0.5rem 1.2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text-dim);
  cursor: pointer;
  font-family: var(--mono);
  font-size: 0.85rem;
  font-weight: 400;
  transition: all 0.15s;
  letter-spacing: 0.04em;
}
.tab:first-child { border-radius: 4px 0 0 4px; }
.tab:last-child { border-radius: 0 4px 4px 0; }
.tab + .tab { border-left: none; }
.tab:hover { color: var(--text); background: var(--surface2); }
.tab.active {
  background: var(--accent-dim);
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 500;
}

/* stats row */
.stats {
  display: flex;
  gap: 1px;
  margin-bottom: 1.5rem;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
}
.stat {
  flex: 1;
  background: var(--surface);
  padding: 0.9rem 1rem;
}
.stat .label {
  font-size: 0.7rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.3rem;
}
.stat .val {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-bright);
}
.stat .val.acc { color: var(--accent); }

/* table */
.tbl {
  width: 100%;
  border-collapse: collapse;
}
.tbl th {
  text-align: left;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  padding: 0.5rem 0.7rem;
  border-bottom: 1px solid var(--border);
  font-weight: 400;
}
.tbl th.r, .tbl td.r { text-align: right; }

.row-main {
  cursor: pointer;
  transition: background 0.1s;
}
.row-main:hover { background: var(--surface2); }
.row-main td {
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
  vertical-align: middle;
}
.row-main .idx {
  color: var(--text-dim);
  font-weight: 300;
  min-width: 3rem;
}
.row-main .typ {
  color: var(--text);
  font-weight: 400;
}

/* accuracy bar */
.bar-wrap {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.bar-track {
  flex: 1;
  height: 6px;
  background: var(--surface);
  border-radius: 3px;
  overflow: hidden;
  border: 1px solid var(--border);
  min-width: 80px;
  max-width: 160px;
}
.bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
}
.bar-pct {
  font-size: 0.85rem;
  font-weight: 500;
  min-width: 3.5rem;
  text-align: right;
}
.bar-fill.hi { background: var(--green); }
.bar-fill.mid { background: var(--accent); }
.bar-fill.lo { background: var(--red); }

.score {
  font-size: 0.85rem;
  color: var(--text-dim);
}
.err-rate {
  font-size: 0.8rem;
  color: var(--red);
}
.err-rate.clean { color: var(--text-dim); }

/* expand arrow */
.arrow {
  color: var(--text-dim);
  font-size: 0.7rem;
  transition: transform 0.15s;
  display: inline-block;
}
.row-main.open .arrow { transform: rotate(90deg); }

/* detail panel */
.row-detail { display: none; }
.row-detail.open { display: table-row; }
.row-detail td {
  padding: 0;
  border-bottom: 1px solid var(--border);
}
.detail-inner {
  background: var(--surface);
  padding: 0.7rem;
  border-left: 2px solid var(--accent-dim);
  margin: 0 0.5rem 0 2rem;
}
.sample {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 0.3rem 0.8rem;
  padding: 0.35rem 0.4rem;
  border-radius: 3px;
  align-items: baseline;
  font-size: 0.8rem;
}
.sample:nth-child(odd) { background: rgba(255,255,255,0.015); }
.sample .badge {
  display: inline-block;
  padding: 0.1rem 0.4rem;
  border-radius: 2px;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  min-width: 3rem;
  text-align: center;
}
.badge.ok { background: var(--green-bg); color: var(--green); border: 1px solid rgba(74,222,128,0.15); }
.badge.fail { background: var(--red-bg); color: var(--red); border: 1px solid rgba(248,113,113,0.15); }
.badge.none { background: var(--orange-bg); color: var(--orange); border: 1px solid rgba(251,146,60,0.15); }
.badge.active { background: var(--blue-bg); color: var(--blue); border: 1px solid rgba(96,165,250,0.15); animation: blink 1.5s ease-in-out infinite; }
.badge.wait { background: var(--gray-badge-bg); color: var(--gray-badge); border: 1px solid rgba(110,110,122,0.15); }
.badge.err { background: var(--red-bg); color: var(--red); border: 1px solid rgba(248,113,113,0.15); }

.sample .ans {
  color: var(--text);
  word-break: break-all;
}
.sample .ts {
  color: var(--text-dim);
  font-size: 0.7rem;
  white-space: nowrap;
}
.preview {
  color: var(--text-dim);
  font-size: 0.75rem;
  opacity: 0.5;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 40ch;
  display: inline;
  margin-left: 0.5rem;
}

/* empty state */
.empty {
  text-align: center;
  padding: 4rem 1rem;
  color: var(--text-dim);
}
.empty h2 {
  font-size: 1rem;
  font-weight: 400;
  margin-bottom: 0.5rem;
  color: var(--text);
}
.empty p { font-size: 0.85rem; }

/* fade-in */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
.fade { animation: fadeIn 0.2s ease both; }
</style>
</head>
<body>
<div class="shell">
  <div class="hdr">
    <h1>CoT Swap Eval</h1>
    <span class="sub">qwen3 reasoning &middot; AIME</span>
    <div class="pulse"></div>
  </div>
  <div class="live-bar" id="livebar" style="display:none"></div>
  <div class="tabs" id="tabs"></div>
  <div id="stats" class="stats" style="display:none"></div>
  <div id="content"></div>
</div>

<script>
let DATA = {};
let COUNTS = {};
let activeTab = null;
let expanded = new Set();

// Compute per-problem stats from per_sample
function probStats(p) {
  const samps = p.per_sample || [];
  const nCorrect = samps.filter(s => s.status === 'correct' || (!s.status && s.correct)).length;
  const nIncorrect = samps.filter(s => s.status === 'incorrect' || (!s.status && s.extracted_answer != null && !s.correct)).length;
  const nValid = nCorrect + nIncorrect;
  const nError = samps.filter(s => s.status === 'error' || s.status === 'no_answer' || (!s.status && s.extracted_answer == null)).length;
  const nPending = samps.filter(s => s.status === 'pending' || s.status === 'in_progress').length;
  const complete = samps.length > 0 && nPending === 0;
  const accuracy = nValid > 0 ? nCorrect / nValid : 0;
  return { nCorrect, nIncorrect, nValid, nError, nPending, complete, accuracy, total: samps.length };
}

function render() {
  const liveEl = document.getElementById('livebar');
  const tabsEl = document.getElementById('tabs');
  const statsEl = document.getElementById('stats');
  const contentEl = document.getElementById('content');
  const labels = Object.keys(DATA).sort();

  // Live counters
  const hasLive = Object.keys(COUNTS).length > 0;
  if (hasLive) {
    const inf = COUNTS.in_progress || 0;
    const pend = COUNTS.pending || 0;
    const done = (COUNTS.correct || 0) + (COUNTS.incorrect || 0) + (COUNTS.no_answer || 0);
    const errs = COUNTS.error || 0;
    liveEl.style.display = 'flex';
    liveEl.innerHTML = `
      <span class="cnt"><span class="dot in-flight"></span> ${inf} in flight</span>
      <span class="cnt"><span class="dot pending"></span> ${pend} pending</span>
      <span class="cnt"><span class="dot done"></span> ${done} done</span>
      ${errs ? `<span class="cnt"><span class="dot err"></span> ${errs} errors</span>` : ''}
    `;
  } else {
    liveEl.style.display = 'none';
  }

  if (!labels.length) {
    tabsEl.innerHTML = '';
    statsEl.style.display = 'none';
    contentEl.innerHTML = '<div class="empty fade"><h2>No data yet</h2><p>Run ./eval.py --model strong to generate results</p></div>';
    return;
  }

  if (!activeTab || !labels.includes(activeTab)) activeTab = labels[0];

  // tabs
  tabsEl.innerHTML = labels.map(l => {
    const n = DATA[l].length;
    return `<div class="tab ${l===activeTab?'active':''}" onclick="switchTab('${l}')">${l} (${n})</div>`;
  }).join('');

  const problems = DATA[activeTab] || [];
  const allStats = problems.map(p => probStats(p));

  // stats — only count complete problems for accuracy and plausible
  const totalWithRequests = problems.length;
  const completedProblems = allStats.filter(s => s.complete && s.nValid > 0);
  const nComplete = completedProblems.length;
  const avgAcc = nComplete ? (completedProblems.reduce((sum, s) => sum + s.accuracy, 0) / nComplete) : 0;
  const plausible = completedProblems.filter(s => s.accuracy >= 0.1 && s.accuracy <= 0.8).length;
  statsEl.style.display = 'flex';
  statsEl.innerHTML = `
    <div class="stat"><div class="label">Complete / Total</div><div class="val">${nComplete} / ${totalWithRequests}</div></div>
    <div class="stat"><div class="label">Avg Accuracy</div><div class="val acc">${(avgAcc*100).toFixed(1)}%</div></div>
    <div class="stat"><div class="label">Plausible</div><div class="val">${plausible}</div></div>
  `;

  if (!totalWithRequests) {
    contentEl.innerHTML = '<div class="empty fade"><p>No problems evaluated yet</p></div>';
    return;
  }

  let html = '<table class="tbl"><thead><tr>';
  html += '<th style="width:2rem"></th><th>Idx</th><th>Type</th><th>Accuracy</th><th class="r">Score</th><th class="r">Errors</th>';
  html += '</tr></thead><tbody>';

  problems.forEach((p, pi) => {
    const key = activeTab + ':' + p.idx;
    const isOpen = expanded.has(key);
    const ps = allStats[pi];
    const pct = (ps.accuracy * 100).toFixed(0);
    const barClass = ps.accuracy >= 0.6 ? 'hi' : ps.accuracy >= 0.3 ? 'mid' : 'lo';
    const errText = ps.nError > 0 ? `${ps.nError}/${ps.total}` : '0';
    const errCls = ps.nError > 0 ? 'err-rate' : 'err-rate clean';

    html += `<tr class="row-main ${isOpen?'open':''}" onclick="toggle('${key}')">`;
    html += `<td><span class="arrow">&#9654;</span></td>`;
    html += `<td class="idx">${p.idx}</td>`;
    html += `<td class="typ">${p.type || ''}</td>`;
    html += `<td><div class="bar-wrap"><div class="bar-track"><div class="bar-fill ${barClass}" style="width:${pct}%"></div></div><span class="bar-pct">${pct}%</span></div></td>`;
    html += `<td class="r score">${ps.nCorrect}/${ps.nValid}</td>`;
    html += `<td class="r ${errCls}">${errText}</td>`;
    html += '</tr>';

    html += `<tr class="row-detail ${isOpen?'open':''}"><td colspan="6"><div class="detail-inner">`;
    if (p.per_sample) {
      let nRunning = 0, nPending = 0, nError = 0;
      const finished = [];
      p.per_sample.forEach(s => {
        const st = s.status;
        if (st === 'in_progress') nRunning++;
        else if (st === 'pending') nPending++;
        else if (st === 'error') nError++;
        else finished.push(s);
      });
      // Summary badges for non-finished
      if (nRunning || nPending || nError) {
        html += `<div class="sample">`;
        if (nRunning) html += `<span class="badge active">${nRunning} running</span> `;
        if (nPending) html += `<span class="badge wait">${nPending} pending</span> `;
        if (nError) html += `<span class="badge err">${nError} errors</span> `;
        html += `<span class="ans"></span><span class="ts"></span></div>`;
      }
      finished.forEach(s => {
        let badge, cls;
        if (s.status === 'no_answer' || s.extracted_answer === null || s.extracted_answer === undefined) {
          badge = 'none'; cls = 'none';
        } else if (s.correct) {
          badge = 'pass'; cls = 'ok';
        } else {
          badge = 'fail'; cls = 'fail';
        }
        const ans = s.extracted_answer ?? '—';
        const ts = s.finished_at ? new Date(s.finished_at * 1000).toLocaleTimeString()
                 : (s.created ? new Date(s.created * 1000).toLocaleTimeString() : '');
        const preview = (s.raw_response || '').replace(/<[^>]*>/g, '').slice(0, 80);

        html += `<div class="sample">`;
        html += `<span class="badge ${cls}">${badge}</span>`;
        html += `<span class="ans">${escHtml(String(ans))}${preview ? `<span class="preview">${escHtml(preview)}</span>` : ''}</span>`;
        html += `<span class="ts">${ts}</span>`;
        html += '</div>';
      });
    }
    html += '</div></td></tr>';
  });

  html += '</tbody></table>';
  contentEl.innerHTML = html;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function switchTab(label) {
  activeTab = label;
  render();
}

function toggle(key) {
  expanded.has(key) ? expanded.delete(key) : expanded.add(key);
  render();
}

async function poll() {
  try {
    const resp = await fetch('/api/data');
    const json = await resp.json();
    DATA = json.problems || json;
    COUNTS = json.counts || {};
    render();
  } catch(e) {}
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
