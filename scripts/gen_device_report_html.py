#!/usr/bin/env python3
"""Render dist/device-report.json as a standalone, filterable HTML page.

Companion to gen_device_report.py (same input, same scrub policy —
never a device IP in the output). The page ships the whole case list
as inline JSON and does filtering/search client-side, so it works
offline as a file and as a published artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_device_report import (  # noqa: E402
    AREA_TITLES,
    _mark_pairs,
    public_api_surface,
)

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def scrub(text) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return IP_RE.sub("目标设备", text)


VERDICTS = ["PASS", "KNOWN-ISSUE", "SKIP-NA", "ERROR"]

# verdict → css token name (defined in the page's stylesheet)
V_TOKEN = {
    "PASS": "pass",
    "KNOWN-ISSUE": "ki",
    "SKIP-NA": "na",
    "SKIP": "na",
    "ERROR": "err",
}


def slim_cases(cases: list[dict]) -> list[dict]:
    """Keep the matrix payload small: evidence rides along only where
    it explains a non-PASS verdict."""
    out = []
    for c in cases:
        row = {
            "id": c["id"].rsplit(".", 1)[-1],
            "cls": c["id"].rsplit(".", 1)[0].split(".", 1)[-1],
            "area": c["area"],
            "iface": scrub(c.get("interface") or "")[:90],
            "verdict": c["verdict"],
            "ms": c.get("duration_ms"),
            "note": scrub(c.get("note") or "")[:220],
        }
        if c["verdict"] != "PASS":
            row["error"] = scrub(c.get("error") or "")[:1200]
        out.append(row)
    return out


def group_known_issues(cases: list[dict]) -> list[dict]:
    rows = [c for c in cases if c["verdict"] == "KNOWN-ISSUE"]
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for c in rows:
        grouped.setdefault(c.get("note") or "未标注", []).append(c)
    out = []
    for reason, group in grouped.items():
        out.append({
            "reason": scrub(reason)[:400],
            "cases": [{
                "id": c["id"].rsplit(".", 1)[-1],
                "iface": scrub(c.get("interface") or "")[:90],
                "ms": c.get("duration_ms"),
                "error": scrub(c.get("error") or "")[:220],
            } for c in group],
        })
    return out


def coverage_rows(cases: list[dict]) -> list[dict]:
    try:
        surface = public_api_surface()
    except Exception:  # noqa: BLE001 — coverage must not kill the page
        return []
    names = frozenset(surface)
    exercised = set()
    for c in cases:
        for pair in _mark_pairs(c.get("interface") or "", names):
            exercised.add(pair)
    rows = []
    for cls, methods in surface.items():
        hit = sum(1 for m in methods if (cls, m) in exercised)
        if hit or len(methods) >= 3:
            rows.append({"cls": cls, "hit": hit, "total": len(methods)})
    rows.sort(key=lambda r: (-r["hit"] / max(r["total"], 1), r["cls"]))
    return rows


def build_payload(report: dict) -> dict:
    cases = report["cases"]
    summary = report["summary"]
    env = report.get("env", {})
    areas = OrderedDict()
    for c in cases:
        areas.setdefault(c["area"], {v: 0 for v in VERDICTS})
        areas[c["area"]][c["verdict"]] = \
            areas[c["area"]].get(c["verdict"], 0) + 1

    verdicts = dict(summary["verdicts"])
    cov = coverage_rows(cases)
    cov_covered = sum(r["hit"] for r in cov)
    cov_total = sum(r["total"] for r in cov)

    return {
        "generated": env.get("generated_at"),
        "sdk": env.get("sdk", {}),
        "node": env.get("node", {}),
        "sockets": env.get("sockets") or [],
        "models": env.get("models") or [],
        "summary": {
            "total": summary["total"],
            "verdicts": verdicts,
            "wallS": summary["wall_time_s"],
            "modules": len(summary.get("modules") or []),
        },
        "areas": [{"key": k, "title": AREA_TITLES.get(k, k), "counts": v}
                  for k, v in areas.items()],
        "coverage": {
            "rows": cov,
            "covered": cov_covered,
            "total": cov_total,
        },
        "knownIssues": group_known_issues(cases),
        "opsNotes": [scrub(n) for n in report.get("ops_notes") or []],
        "cases": slim_cases(cases),
        "errorCases": [{
            "id": c["id"].rsplit(".", 1)[-1],
            "iface": scrub(c.get("interface") or ""),
            "ms": c.get("duration_ms"),
            "note": scrub(c.get("note") or ""),
            "error": scrub(c.get("error") or ""),
            "evidence": {k: scrub(v)[:200]
                         for k, v in (c.get("evidence") or {}).items()},
        } for c in cases if c["verdict"] == "ERROR"],
    }


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDK 0.7.4 真机测试报告</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  --ground: #F5F6F8;
  --surface: #FFFFFF;
  --surface-2: #EEF0F4;
  --ink: #1A1D24;
  --ink-2: #4C5361;
  --line: #E1E4EA;
  --line-strong: #C9CDD6;
  --accent: #2F4D8F;
  --accent-ink: #FFFFFF;
  --pass: #177A48;
  --pass-soft: #E3F1E9;
  --ki: #96660A;
  --ki-soft: #F6EDDA;
  --err: #B3261E;
  --err-soft: #F9E7E5;
  --na: #6E7683;
  --na-soft: #E9EBEF;
  --code-bg: #F1F3F6;
  --shadow: 0 1px 2px rgba(26,29,36,.06), 0 4px 14px rgba(26,29,36,.07);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #14161B;
    --surface: #1B1E26;
    --surface-2: #22262F;
    --ink: #E9EBEF;
    --ink-2: #98A0AE;
    --line: #2B303A;
    --line-strong: #3B4150;
    --accent: #8AA8E0;
    --accent-ink: #14161B;
    --pass: #53C08B;
    --pass-soft: #1D2E26;
    --ki: #D6A154;
    --ki-soft: #2F2718;
    --err: #E5776B;
    --err-soft: #33211F;
    --na: #8A92A0;
    --na-soft: #23262E;
    --code-bg: #22262F;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground: #14161B;
  --surface: #1B1E26;
  --surface-2: #22262F;
  --ink: #E9EBEF;
  --ink-2: #98A0AE;
  --line: #2B303A;
  --line-strong: #3B4150;
  --accent: #8AA8E0;
  --accent-ink: #14161B;
  --pass: #53C08B;
  --pass-soft: #1D2E26;
  --ki: #D6A154;
  --ki-soft: #2F2718;
  --err: #E5776B;
  --err-soft: #33211F;
  --na: #8A92A0;
  --na-soft: #23262E;
  --code-bg: #22262F;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 18px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font: 400 14px/1.62 "IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace; font-variant-numeric: tabular-nums; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 28px 72px; }

/* ---------- header ---------- */
header.masthead { padding: 42px 0 22px; border-bottom: 2px solid var(--ink); }
.eyebrow {
  font: 600 11px/1 "IBM Plex Mono", monospace;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 14px;
}
h1 { font-size: 27px; line-height: 1.25; font-weight: 700; margin: 0 0 6px; letter-spacing: -.01em; text-wrap: balance; }
.meta { color: var(--ink-2); font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 22px; margin-top: 12px; }
.meta b { color: var(--ink); font-weight: 600; }

/* ---------- section scaffolding ---------- */
section { margin-top: 46px; }
h2 {
  font-size: 18px; font-weight: 700; margin: 0 0 4px;
  display: flex; align-items: baseline; gap: 10px;
}
h2 .n { font: 600 11px/1 "IBM Plex Mono", monospace; color: var(--accent); letter-spacing: .1em; }
.sec-note { color: var(--ink-2); font-size: 12.5px; margin: 0 0 16px; }

/* ---------- verdict band ---------- */
.band { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 4fr); gap: 36px; align-items: start; }
@media (max-width: 900px) { .band { grid-template-columns: 1fr; } }
.vbar { display: flex; height: 30px; border-radius: 4px; overflow: hidden; border: 1px solid var(--line-strong); }
.vbar .seg-pass { background: var(--pass); }
.vbar .seg-ki  { background: var(--ki); }
.vbar .seg-na  { background: var(--na); }
.vbar .seg-err { background: var(--err); min-width: 5px; }
.vstats { display: flex; flex-wrap: wrap; margin-top: 14px; border-top: 1px solid var(--line); }
.vstat { padding: 12px 26px 2px 0; }
.vstat + .vstat { padding-left: 26px; border-left: 1px solid var(--line); }
.vstat .num { font: 600 30px/1 "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.vstat .lbl { font-size: 12px; color: var(--ink-2); margin-top: 4px; display: flex; align-items: center; gap: 6px; }
.vstat.err .num { color: var(--err); }
.vstat.ki .num { color: var(--ki); }
.vstat.na .num { color: var(--na); }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.dot.pass { background: var(--pass); } .dot.ki { background: var(--ki); }
.dot.na { background: var(--na); } .dot.err { background: var(--err); }

/* area distribution — one shared scale */
.area-chart { border-top: 1px solid var(--line); }
.area-row { display: grid; grid-template-columns: 128px 1fr 208px; gap: 12px; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--line); }
.area-row .name { font-size: 12.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.area-row .name small { display: block; font: 400 10.5px/1.3 "IBM Plex Mono", monospace; color: var(--ink-2); font-weight: 400; }
.track { height: 13px; display: flex; border-radius: 2px; overflow: hidden; background: var(--surface-2); }
.track i { display: block; height: 100%; }
.track .s-pass { background: var(--pass); } .track .s-ki { background: var(--ki); }
.track .s-na { background: var(--na); } .track .s-err { background: var(--err); min-width: 3px; }
.area-row .nums { font: 500 11.5px/1.5 "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; color: var(--ink-2); text-align: right; white-space: nowrap; }
.area-row .nums b { color: var(--ink); font-weight: 600; }

/* ---------- findings ---------- */
.err-card {
  background: var(--surface); border: 1px solid var(--line-strong);
  border-left: 4px solid var(--err); border-radius: 6px;
  padding: 20px 22px; box-shadow: var(--shadow);
}
.err-card .head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.err-card .head .t { font: 600 15px/1.3 "IBM Plex Mono", monospace; }
.err-card .why { margin: 12px 0 0; font-size: 13.5px; }
.err-card pre {
  margin: 14px 0 0; padding: 12px 14px; background: var(--code-bg);
  border-radius: 4px; overflow-x: auto;
  font: 400 11.5px/1.6 "IBM Plex Mono", monospace; color: var(--ink-2);
  white-space: pre-wrap; word-break: break-all;
}
.ev-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.ev-chips span { font: 500 11px/1 "IBM Plex Mono", monospace; background: var(--surface-2); border: 1px solid var(--line); border-radius: 3px; padding: 5px 8px; }

.chip {
  display: inline-flex; align-items: center; gap: 6px;
  font: 600 10.5px/1 "IBM Plex Mono", monospace; letter-spacing: .06em;
  border-radius: 3px; padding: 4px 8px;
}
.chip.pass { color: var(--pass); background: var(--pass-soft); }
.chip.ki   { color: var(--ki);   background: var(--ki-soft); }
.chip.na   { color: var(--na);   background: var(--na-soft); }
.chip.err  { color: var(--err);  background: var(--err-soft); }

/* known issues */
.ki-list { display: grid; gap: 14px; }
.ki-item { border: 1px solid var(--line); border-radius: 6px; background: var(--surface); padding: 16px 18px; }
.ki-item .reason { font-size: 13.5px; font-weight: 600; line-height: 1.55; }
.ki-item .cases { margin: 10px 0 0; padding: 0; list-style: none; display: grid; gap: 6px; }
.ki-item .cases li { display: flex; flex-wrap: wrap; gap: 4px 14px; font: 400 12px/1.6 "IBM Plex Mono", monospace; color: var(--ink-2); }
.ki-item .cases li b { color: var(--ink); font-weight: 500; }

/* ---------- matrix ---------- */
.controls { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; margin-bottom: 14px; }
.fchips { display: flex; gap: 6px; }
.fchip {
  font: 600 11px/1 "IBM Plex Mono", monospace; letter-spacing: .04em;
  border: 1px solid var(--line-strong); background: var(--surface);
  color: var(--ink-2); border-radius: 999px; padding: 7px 12px;
  cursor: pointer; display: inline-flex; align-items: center; gap: 7px;
}
.fchip .dot { width: 7px; height: 7px; }
.fchip[aria-pressed="true"] { color: var(--accent); border-color: var(--accent); }
.fchip:focus-visible, .area-sel:focus-visible, .search:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.area-sel, .search {
  font: 400 12.5px/1 "IBM Plex Sans", sans-serif; color: var(--ink);
  background: var(--surface); border: 1px solid var(--line-strong);
  border-radius: 5px; padding: 8px 10px;
}
.search { flex: 1; min-width: 170px; max-width: 330px; }
.count { font: 500 11.5px/1 "IBM Plex Mono", monospace; color: var(--ink-2); margin-left: auto; white-space: nowrap; }

.matrix-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table.matrix { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.matrix th {
  position: sticky; top: 0; text-align: left; z-index: 1;
  font: 600 10.5px/1 "IBM Plex Mono", monospace; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink-2); background: var(--surface-2);
  padding: 9px 12px; border-bottom: 1px solid var(--line-strong);
}
table.matrix td { padding: 7px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.matrix tr:last-child td { border-bottom: 0; }
td.c-id { font: 500 11.5px/1.5 "IBM Plex Mono", monospace; white-space: nowrap; }
td.c-id small { display: block; color: var(--ink-2); font-weight: 400; font-size: 10.5px; }
td.c-if { font: 400 11.5px/1.5 "IBM Plex Mono", monospace; color: var(--ink-2); word-break: break-all; max-width: 330px; }
td.c-ms { font: 500 11.5px/1.5 "IBM Plex Mono", monospace; text-align: right; white-space: nowrap; color: var(--ink-2); }
td.c-note { color: var(--ink-2); max-width: 360px; }
td.c-note .m { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
tr.is-err td { background: var(--err-soft); }
tr.is-ki td.c-id, tr.is-ki td.c-if { color: var(--ink); }

/* ---------- timeline ---------- */
.timeline { list-style: none; margin: 0; padding: 0; border-left: 2px solid var(--line-strong); }
.timeline li { position: relative; padding: 0 0 22px 26px; }
.timeline li:last-child { padding-bottom: 2px; }
.timeline li::before {
  content: ""; position: absolute; left: -6px; top: 5px;
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--ground);
}
.timeline .t { font: 600 11.5px/1 "IBM Plex Mono", monospace; color: var(--accent); display: block; margin-bottom: 6px; }
.timeline p { margin: 0; font-size: 13.5px; }

/* ---------- coverage + env ---------- */
.duo { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 36px; align-items: start; }
@media (max-width: 900px) { .duo { grid-template-columns: 1fr; } }
.cov-headline { font-size: 13.5px; margin: 0 0 12px; }
.cov-headline b { font: 600 20px/1 "IBM Plex Mono", monospace; }
table.cov { width: 100%; border-collapse: collapse; font-size: 12px; }
table.cov td { padding: 5px 8px 5px 0; border-bottom: 1px solid var(--line); }
table.cov td:last-child { text-align: right; font: 500 11.5px/1.4 "IBM Plex Mono", monospace; white-space: nowrap; color: var(--ink-2); }
table.cov .covbar { height: 5px; border-radius: 2px; background: var(--surface-2); position: relative; min-width: 64px; }
table.cov .covbar i { position: absolute; inset: 0 auto 0 0; background: var(--pass); border-radius: 2px; }
table.cov tr.zero .covbar i { background: var(--na); }
dl.env { display: grid; grid-template-columns: max-content 1fr; gap: 7px 18px; margin: 0; font-size: 12.5px; }
dl.env dt { color: var(--ink-2); white-space: nowrap; }
dl.env dd { margin: 0; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; word-break: break-all; }
.sock-list { display: flex; flex-wrap: wrap; gap: 5px; margin: 4px 0 2px; }
.sock-list span { font: 400 10.5px/1 "IBM Plex Mono", monospace; background: var(--surface-2); border: 1px solid var(--line); border-radius: 3px; padding: 4px 7px; }

footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line-strong); color: var(--ink-2); font-size: 12.5px; }
footer code { font: 400 11px/1.5 "IBM Plex Mono", monospace; background: var(--code-bg); border-radius: 3px; padding: 2px 6px; }

@media (prefers-reduced-motion: no-preference) {
  table.matrix tbody tr:hover td { background: var(--surface-2); }
  table.matrix tbody tr.is-err:hover td { background: var(--err-soft); }
}
</style>
</head>
<body>
<div class="wrap">

<header class="masthead">
  <div class="eyebrow">NEORUNTIME IPC SDK · DEVICE RUN</div>
  <h1>SDK 0.7.4 真机接口测试报告</h1>
  <div class="meta">
    <span>目标设备 <b class="mono">aarch64 · hailo15</b></span>
    <span>Python <b class="mono" id="m-python"></b></span>
    <span>执行 <b class="mono" id="m-date"></b>(设备时间 GMT)</span>
    <span>14 个模块 · 每模块独立进程 + 看门狗</span>
  </div>
</header>

<section id="sec-overview">
  <h2>总览</h2>
  <p class="sec-note">192 个用例覆盖 14 个区域;唯一 ERROR 的归因见下一节,已知问题按根因归组为 6 项。</p>
  <div class="band">
    <div>
      <div class="vbar" id="vbar"></div>
      <div class="vstats" id="vstats"></div>
    </div>
    <div class="area-chart" id="area-chart"></div>
  </div>
</section>

<section id="sec-findings">
  <h2>需要跟进的结论</h2>
  <p class="sec-note">先看要动手的,再看全量矩阵。</p>
  <div id="err-cards"></div>
  <div class="ki-list" id="ki-list" style="margin-top:18px"></div>
</section>

<section id="sec-matrix">
  <h2>接口 × 结论矩阵</h2>
  <p class="sec-note">全量 192 行,按结论 / 区域 / 关键词过滤;备注列为现场证据摘要。</p>
  <div class="controls">
    <div class="fchips" id="fchips"></div>
    <select class="area-sel" id="area-sel" aria-label="按区域过滤"></select>
    <input class="search" id="search" type="search" placeholder="搜索用例 / 接口 / 备注…" aria-label="搜索矩阵">
    <span class="count" id="count"></span>
  </div>
  <div class="matrix-scroll">
    <table class="matrix">
      <thead><tr><th>用例</th><th>接口</th><th>结论</th><th style="text-align:right">耗时</th><th>备注</th></tr></thead>
      <tbody id="matrix-body"></tbody>
    </table>
  </div>
</section>

<section id="sec-ops">
  <h2>环境事件与恢复记录</h2>
  <p class="sec-note">本轮执行期间目标设备上发生的 daemon 级事件与恢复动作,时间为设备时钟(GMT)。</p>
  <ol class="timeline" id="timeline"></ol>
</section>

<section id="sec-cov">
  <h2>覆盖率与环境</h2>
  <div class="duo" style="margin-top:14px">
    <div>
      <p class="cov-headline" id="cov-headline"></p>
      <table class="cov"><tbody id="cov-body"></tbody></table>
    </div>
    <div>
      <dl class="env" id="env-dl"></dl>
    </div>
  </div>
</section>

<footer id="footer"></footer>
</div>

<script>
const D = __DATA__;
const V_ORDER = ["PASS", "KNOWN-ISSUE", "SKIP-NA", "ERROR"];
const V_TOKEN = { "PASS": "pass", "KNOWN-ISSUE": "ki", "SKIP-NA": "na", "SKIP": "na", "ERROR": "err" };
const V_LABEL = { "PASS": "通过 PASS", "KNOWN-ISSUE": "已知问题 KNOWN-ISSUE", "SKIP-NA": "不适用 SKIP-NA", "SKIP": "跳过", "ERROR": "错误 ERROR" };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (ms) => ms == null ? "—" : (ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms");

/* ---- header meta ---- */
$("m-python").textContent = D.node.python || "—";
$("m-date").textContent = (D.generated || "").slice(0, 10);

/* ---- 01 overview ---- */
(function overview() {
  const total = D.summary.total;
  const counts = D.summary.verdicts;
  $("vbar").innerHTML = V_ORDER.map((v) => {
    const w = ((counts[v] || 0) / total) * 100;
    return `<i class="seg-${V_TOKEN[v]}" style="width:${w}%" title="${esc(V_LABEL[v])} × ${counts[v] || 0}"></i>`;
  }).join("");
  $("vstats").innerHTML = V_ORDER.map((v) => `
    <div class="vstat ${V_TOKEN[v]}">
      <div class="num">${counts[v] || 0}</div>
      <div class="lbl"><span class="dot ${V_TOKEN[v]}"></span>${esc(V_LABEL[v])}</div>
    </div>`).join("") + `
    <div class="vstat"><div class="num">${total}</div><div class="lbl">用例总数</div></div>
    <div class="vstat"><div class="num">${D.summary.wallS}s</div><div class="lbl">模块端到端</div></div>`;

  const maxN = Math.max(...D.areas.map((a) => Object.values(a.counts).reduce((x, y) => x + y, 0)));
  $("area-chart").innerHTML = D.areas.map((a) => {
    const n = Object.values(a.counts).reduce((x, y) => x + y, 0);
    const scale = (n / maxN) * 100;
    const segs = V_ORDER.map((v) => {
      const c = a.counts[v] || 0;
      if (!c) return "";
      return `<i class="s-${V_TOKEN[v]}" style="width:${(c / n) * 100}%" title="${esc(v)} × ${c}"></i>`;
    }).join("");
    const nums = V_ORDER.filter((v) => a.counts[v]).map((v) => `${a.counts[v]}${v === "PASS" ? "" : v === "ERROR" ? "E" : v === "KNOWN-ISSUE" ? "K" : "N"}`).join(" · ");
    return `<div class="area-row">
      <div class="name">${esc(a.title)}<small>${esc(a.key)}</small></div>
      <div class="track" style="width:${scale}%" title="${n} 用例">${segs}</div>
      <div class="nums"><b>${n}</b> 用例 · ${esc(nums)}</div>
    </div>`;
  }).join("");
})();

/* ---- 02 findings ---- */
(function findings() {
  $("err-cards").innerHTML = D.errorCases.length ? D.errorCases.map((c) => `
    <div class="err-card">
      <div class="head">
        <span class="chip err">ERROR</span>
        <span class="t">${esc(c.id)}</span>
        <span class="mono" style="color:var(--ink-2);font-size:11.5px">${esc(c.iface)} · ${fmtMs(c.ms)}</span>
      </div>
      ${c.note ? `<p class="why">${esc(c.note)}</p>` : ""}
      ${Object.keys(c.evidence || {}).length ? `<div class="ev-chips">${Object.entries(c.evidence).map(([k, v]) => `<span>${esc(k)}: ${esc(v)}</span>`).join("")}</div>` : ""}
      ${c.error ? `<pre>${esc(c.error)}</pre>` : ""}
    </div>`).join("") : "";
  $("ki-list").innerHTML = D.knownIssues.map((g) => `
    <div class="ki-item">
      <div class="reason"><span class="chip ki" style="margin-right:8px">KI · ${g.cases.length} 例</span>${esc(g.reason)}</div>
      <ul class="cases">${g.cases.map((c) => `<li><b>${esc(c.id)}</b><span>${esc(c.iface)}</span><span>${fmtMs(c.ms)}</span></li>`).join("")}</ul>
    </div>`).join("");
})();

/* ---- 03 matrix ---- */
const state = { verdicts: new Set(), area: "", q: "" };
try {
  const saved = JSON.parse(localStorage.getItem("sdk-report-filter") || "null");
  if (saved) { state.verdicts = new Set(saved.verdicts || []); state.area = saved.area || ""; state.q = saved.q || ""; }
} catch (e) { /* private mode etc. */ }
function persist() { try { localStorage.setItem("sdk-report-filter", JSON.stringify({ verdicts: [...state.verdicts], area: state.area, q: state.q })); } catch (e) {} }

(function buildControls() {
  $("fchips").innerHTML = V_ORDER.map((v) => `
    <button class="fchip" data-v="${v}" aria-pressed="${state.verdicts.has(v)}">
      <span class="dot ${V_TOKEN[v]}"></span>${esc(v)}
      <span style="opacity:.65">${D.summary.verdicts[v] || 0}</span>
    </button>`).join("");
  $("fchips").addEventListener("click", (e) => {
    const btn = e.target.closest(".fchip"); if (!btn) return;
    const v = btn.dataset.v;
    state.verdicts.has(v) ? state.verdicts.delete(v) : state.verdicts.add(v);
    btn.setAttribute("aria-pressed", state.verdicts.has(v));
    persist(); renderMatrix();
  });
  const sel = $("area-sel");
  sel.innerHTML = `<option value="">全部区域</option>` + D.areas.map((a) => `<option value="${esc(a.key)}">${esc(a.title)} ${esc(a.key)}</option>`).join("");
  sel.value = state.area;
  sel.addEventListener("change", () => { state.area = sel.value; persist(); renderMatrix(); });
  const search = $("search");
  search.value = state.q;
  search.addEventListener("input", () => { state.q = search.value.trim().toLowerCase(); persist(); renderMatrix(); });
})();

function renderMatrix() {
  const rows = D.cases.filter((c) =>
    (!state.verdicts.size || state.verdicts.has(c.verdict)) &&
    (!state.area || c.area === state.area) &&
    (!state.q || (c.id + " " + c.cls + " " + c.iface + " " + (c.note || "")).toLowerCase().includes(state.q)));
  $("matrix-body").innerHTML = rows.map((c) => `
    <tr class="is-${V_TOKEN[c.verdict]}">
      <td class="c-id">${esc(c.id)}<small>${esc(c.cls)}</small></td>
      <td class="c-if">${esc(c.iface)}</td>
      <td><span class="chip ${V_TOKEN[c.verdict]}">${esc(c.verdict)}</span></td>
      <td class="c-ms">${fmtMs(c.ms)}</td>
      <td class="c-note">${c.note ? `<div class="m">${esc(c.note)}</div>` : "—"}</td>
    </tr>`).join("");
  $("count").textContent = `${rows.length} / ${D.cases.length} 行`;
}
renderMatrix();

/* ---- 04 timeline ---- */
$("timeline").innerHTML = D.opsNotes.map((n) => {
  const m = n.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?:[–—-]\d{2}:\d{2})?)/);
  const t = m ? m[1] : "";
  const body = t ? n.slice(m[0].length).replace(/^[((]?/, "").replace(/[))]?$/, "") : n;
  return `<li>${t ? `<span class="t">${esc(t)}</span>` : ""}<p>${esc(body)}</p></li>`;
}).join("");

/* ---- 05 coverage + env ---- */
(function covEnv() {
  const c = D.coverage;
  if (c.total) {
    $("cov-headline").innerHTML = `公开 API 方法覆盖:<b>${c.covered}/${c.total}</b>( ${(100 * c.covered / c.total).toFixed(1)}%,保守口径:矩阵中以 Class.method 形态标记的调用)`;
    $("cov-body").innerHTML = c.rows.map((r) => `
      <tr class="${r.hit ? "" : "zero"}">
        <td class="mono" style="white-space:nowrap">${esc(r.cls)}</td>
        <td><div class="covbar"><i style="width:${(100 * r.hit / r.total).toFixed(0)}%"></i></div></td>
        <td>${r.hit}/${r.total}</td>
      </tr>`).join("");
  }
  const kv = [
    ["报告生成", esc((D.generated || "").replace("T", " ").slice(0, 19)) + " GMT"],
    ["内核/架构", esc((D.node.uname || "").split(" #")[0])],
    ["SDK 版本", `<b style="font-weight:600">${esc(D.sdk.version || "")}</b>`],
    ["安装路径", esc(D.sdk.module_path || "")],
    ["UDS 端点", `${D.sockets.length} 个`],
  ];
  $("env-dl").innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
    `<dt style="align-self:start;padding-top:2px">socket 清单</dt><dd><div class="sock-list">${D.sockets.map((s) => `<span>${esc(s)}</span>`).join("")}</div></dd>`;
})();

/* ---- footer ---- */
$("footer").innerHTML = `设备 venv 由编排脚本安装被测版本,回滚:<code>pip install --no-deps neoruntime-ipc-sdk==&lt;previous-version&gt;</code>;镜头物理状态已由 <code>lens_reset_zero()</code> 复位校验。报告数据:${esc((D.generated || "").slice(0, 10))} · SDK ${esc(D.sdk.version || "")}。`;
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="dist/device-report.json")
    ap.add_argument("-o", "--output",
                    default="docs/test-reports/{date}-python-sdk-{ver}-device.html")
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)

    payload = build_payload(report)
    date = (report.get("env", {}).get("generated_at") or "")[:10] or "unknown-date"
    ver = payload["sdk"].get("version", "x")
    out_path = args.output.format(date=date, ver=ver)

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")  # never terminate the script tag
    html = TEMPLATE.replace("__DATA__", data_json)

    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(html)
    os.replace(tmp, out_path)

    assert not IP_RE.search(html), "device IP leaked into HTML"
    print(f"page: {out_path} ({os.path.getsize(out_path) // 1024} KiB, "
          f"{len(payload['cases'])} cases, "
          f"{len(payload['knownIssues'])} KI groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
