#!/usr/bin/env python3
"""Render the on-device SDK performance report as Markdown.

Usage::

    python3 scripts/gen_perf_report.py dist/device-perf-report.json \
        [--output docs/test-reports/<date>-python-sdk-perf.md]

Input is the JSON produced by the perf modules via
``python/tests/device/run_device_tests.py`` (schema
``neoruntime-device-test-report/1``; measurements live under each
case's ``evidence["perf:<label>"]``). Output layout:

* 结论(采样方法、异常阈值、设备环境)
* P1 推理 — 控制面 RPC / 端到端 infer / 批量 / 流到达
* P2 媒体与加速层 — 帧路径 + 路由决策 + A/B 对照
* P3 事件 — publish RPC / 批量吞吐 / 端到端投递 / 到达抖动
* P4 设备面 — 只读状态 RPC(GPIO 按排除策略不出现在数据里)
* P5 长稳 — 分桶 RSS 归因(客户端 vs 各 daemon)+ 速率衰减
* NA / KNOWN-ISSUE 明细

异常标注(不做硬性 PASS/FAIL,报告呈现事实):
``p99/p50 > 8`` 长尾、``err_pct > 0.5`` 错误率、soak 速率衰减 > 10%。

脱敏规约(仓库 b373aad):一切渲染文本做 IPv4 擦除,意外出现的设备
地址一律替换为「目标设备」。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

# Mirrors perf_common.ANOMALY_* — keep in sync (a single source would
# need the device tree importable here; not worth it for 3 constants).
ANOMALY_TAIL_RATIO = 8.0
ANOMALY_ERR_PCT = 0.5
ANOMALY_DECAY_PCT = 10.0

AREA_TITLES = OrderedDict([
    ("perf-inference", "P1 推理"),
    ("perf-media", "P2 媒体与加速层"),
    ("perf-events", "P3 事件"),
    ("perf-device", "P4 设备面"),
    ("perf-soak", "P5 长稳"),
])


def scrub(value) -> str:
    """Any value → one line of Markdown-safe, IP-free text."""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return IP_RE.sub("目标设备", text)


def as_obj(value):
    """Evidence dict/list values arrive str()'d by the device runner —
    scalars keep their type, containers come back as repr text. Parse
    those back; anything unparsable or scalar becomes None."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, (dict, list)) else None
    return None


def fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000.0:.2f}s" if ms >= 1000 else f"{ms:.1f}ms"


def fmt_kb(kb) -> str:
    if not isinstance(kb, (int, float)):
        return "—"
    sign = "-" if kb < 0 else ""
    kb = abs(kb)
    if kb >= 1024:
        return f"{sign}{kb / 1024.0:.1f}MiB"
    return f"{sign}{kb:.0f}KiB"


def perf_records(cases):
    """Yield (case, label, record) for every perf:* evidence entry."""
    for case in cases:
        for key, raw in (case.get("evidence") or {}).items():
            if not key.startswith("perf:"):
                continue
            record = as_obj(raw)
            if isinstance(record, dict):
                yield case, key[len("perf:"):], record


def latency_flags(rec) -> list[str]:
    flags = []
    p50, p99 = rec.get("p50"), rec.get("p99")
    if p50 and p99 and p99 / p50 > ANOMALY_TAIL_RATIO:
        flags.append(f"长尾(p99/p50={p99 / p50:.0f}×)")
    err_pct = rec.get("err_pct")
    if err_pct is not None and err_pct > ANOMALY_ERR_PCT:
        flags.append(f"错误率{err_pct}%")
    return flags


def spread_note(rec) -> str:
    spread = rec.get("round_spread")
    if not spread or not spread.get("spread_pct"):
        return ""
    return f"(轮间{spread['spread_pct']}%)"


def latency_table(rows) -> list[str]:
    out = ["| 接口/操作 | n | ok/err | p50 | p90 | p95 | p99 | max | 标注 |",
           "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for label, rec in rows:
        flags = "、".join(latency_flags(rec))
        out.append(
            f"| `{label}` {spread_note(rec)} | {rec.get('n', '—')} "
            f"| {rec.get('ok', '—')}/{rec.get('err', '—')} "
            f"| {fmt_ms(rec.get('p50'))} | {fmt_ms(rec.get('p90'))} "
            f"| {fmt_ms(rec.get('p95'))} | {fmt_ms(rec.get('p99'))} "
            f"| {fmt_ms(rec.get('max'))} | {flags or '—'} |")
    return out


def stream_table(rows) -> list[str]:
    out = ["| 流 | 帧 | 时长 | fps | 丢帧 | 丢帧率 | 间隔p50 | 间隔p95 | 间隔max |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, rec in rows:
        drops = "—" if rec.get("drops") is None else rec.get("drops")
        drop_pct = ("—" if rec.get("drop_pct") is None
                    else f"{rec.get('drop_pct')}%")
        out.append(
            f"| `{label}` | {rec.get('frames', '—')} "
            f"| {rec.get('duration_s', '—')}s | {rec.get('fps', '—')} "
            f"| {drops} | {drop_pct} | {fmt_ms(rec.get('gap_p50'))} "
            f"| {fmt_ms(rec.get('gap_p95'))} | {fmt_ms(rec.get('gap_max'))} |")
    return out


def split_records(cases, area: str):
    """(latency, stream) rows for one perf area."""
    latency, stream = [], []
    for case, label, rec in perf_records(cases):
        if case.get("area") != area:
            continue
        if rec.get("unit") == "ms":
            latency.append((label, rec))
        elif "frames" in rec:
            stream.append((label, rec))
    return latency, stream


def ab_section(cases) -> list[str]:
    """A/B 对照表 + 路由决策证据(来自 P2 的 evidence)。"""
    out = []
    ratios = health = probes = routes = None
    for case in cases:
        ev = case.get("evidence") or {}
        if "p50_ratio_default_over_sw" in ev:
            ratios = as_obj(ev["p50_ratio_default_over_sw"])
        health = health or as_obj(ev.get("health"))
        probes = probes or as_obj(ev.get("probes"))
        routes = routes or as_obj(ev.get("routes"))

    default, swonly = {}, {}
    for _case, label, rec in perf_records(cases):
        if label.startswith("ab_") and label.endswith("_default"):
            default[label[3:-8]] = rec
        elif label.startswith("ab_") and label.endswith("_swonly"):
            swonly[label[3:-7]] = rec

    if not default and not ratios:
        return out
    out += ["", "### accel 路由 A/B(默认策略 vs SOFTWARE_ONLY)", "",
            "| 操作 | 默认 p50 | 仅软件 p50 | 默认/软件 |",
            "|---|---:|---:|---:|"]
    for op in sorted(set(default) | set(swonly) | set(ratios or {})):
        a = default.get(op, {}).get("p50")
        b = swonly.get(op, {}).get("p50")
        ratio = (ratios or {}).get(op)
        out.append(f"| `{op}` | {fmt_ms(a)} | {fmt_ms(b)} "
                   f"| {ratio if ratio is not None else '—'} |")
    if routes or health or probes:
        out += ["", "路由决策证据(同一次运行):",
                f"- probes: `{scrub(probes)}`",
                f"- health: `{scrub(health)}`"]
        for op, decision in sorted((routes or {}).items()):
            out.append(f"- `{op}` → {scrub(decision)}")
        out.append("注:比值≈1.0 表示两侧执行同一软件腿(记录的是路由决策"
                   "而非加速比);比值>1 表示硬件路由腿真实参与且比软件腿"
                   "更慢。硬件腿是 DSP daemon 的 UDS 调用,零拷贝 dma-buf "
                   "导入仅对 keep-fd 帧源生效——本 A/B 输入为普通数组,"
                   "比值主要计价每次调用的像素 socket 传输,属路由质量"
                   "发现(数组输入场景),非 DSP 算力结论。")
    return out


def soak_section(cases) -> list[str]:
    out = []
    for case in cases:
        if case.get("area") != "perf-soak":
            continue
        ev = case.get("evidence") or {}
        if "duration_s" not in ev:
            continue
        out += ["", "### 长稳结果", "",
                f"- 模式: **{scrub(ev.get('mode'))}**"
                + (f"(infer 冒烟失败:{scrub(ev.get('infer_smoke_error'))})"
                   if ev.get("infer_smoke_error") else ""),
                f"- 时长 {ev.get('duration_s')}s / 迭代 {ev.get('iterations')}"
                f" / 错误 {ev.get('errors')}",
                f"- 速率: 首 {ev.get('iter_rate_first_per_s')}/s → "
                f"末 {ev.get('iter_rate_last_per_s')}/s"
                f"(衰减 {ev.get('rate_decay_pct')}%,"
                f"阈值 {ev.get('anomaly_threshold_decay_pct')}%)",
                f"- 客户端 RSS: {fmt_kb(ev.get('client_rss_first_kb'))} → "
                f"{fmt_kb(ev.get('client_rss_last_kb'))}"
                f"(Δ {fmt_kb(ev.get('client_rss_delta_kb'))});"
                f" fd {ev.get('client_fds_first')} → "
                f"{ev.get('client_fds_last')}",
                "", "| daemon | RSS Δ |", "|---|---:|"]
        for name, delta in sorted((as_obj(ev.get("daemon_rss_delta_kb"))
                                   or {}).items()):
            out.append(f"| {scrub(name)} | {fmt_kb(delta)} |")
        buckets = as_obj(ev.get("buckets")) or []
        if len(buckets) > 1:
            out += ["", "分桶序列(客户端 RSS KiB / 迭代数):", "```"]
            for row in buckets:
                rss = (row.get("client") or {}).get("rss_kb")
                out.append(f"t={row.get('t_s'):>7}s  rss={rss or 0:>9}  "
                           f"iters={row.get('iters')}")
            out += ["```"]
    return out


def notes_section(cases) -> list[str]:
    out = []
    buckets = (
        ("NA(前置缺失/路径不可用)",
         [c for c in cases if c.get("verdict") == "SKIP-NA"]),
        ("KNOWN-ISSUE(已知缺陷,非本次新失败)",
         [c for c in cases if c.get("verdict") == "KNOWN-ISSUE"]),
        ("FAIL/ERROR", [c for c in cases
                        if c.get("verdict") in ("FAIL", "ERROR")]),
    )
    for title, rows in buckets:
        if not rows:
            continue
        out += ["", f"### {title}", ""]
        for row in rows:
            err_lines = (row.get("error") or "").strip().splitlines()
            last = err_lines[-1] if err_lines else ""
            out.append(f"- `{row.get('id', '?')}`: "
                       f"{scrub(row.get('note') or '')}"
                       + (f" — `{scrub(last)}`" if last else ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", help="device-perf-report.json 路径")
    parser.add_argument("--output", default=None,
                        help="输出 Markdown(默认 docs/test-reports/"
                             "<yyyymmdd>-python-sdk-perf.md)")
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)
    cases = report.get("cases", [])
    env = report.get("env") or {}
    node = env.get("node") or {}
    sdk = env.get("sdk") or {}
    verdicts = report.get("summary", {}).get("verdicts", {})

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = args.output or str(
        Path(__file__).resolve().parent.parent / "docs" / "test-reports"
        / f"{stamp}-python-sdk-perf.md")

    lines = [
        "# Python SDK 接口性能报告(设备实测)",
        "",
        f"- 日期: {when}",
        f"- 设备: {scrub(node.get('uname'))}",
        f"- SDK: {scrub(sdk.get('version'))}"
        f"(模块路径 `{scrub(sdk.get('module_path'))}`)",
        f"- Python: {scrub(node.get('python'))}",
        f"- 用例结论分布: {scrub(verdicts)}",
        "",
        "## 方法",
        "",
        "- 每项操作:预热丢弃(10%)后采样,默认 300 样本 × 3 轮,"
        "取 p50 中位轮为首数,附轮间离散度;",
        "- 流式接口:固定时长消费,按 frame_seq/payload seq 连续性计丢帧,"
        "到达间隔分布反映抖动;",
        "- 错误只计数不计时;错误率 >50% 的路径提前终止,避免烧预算;",
        f"- 异常标注阈值:p99/p50 > {ANOMALY_TAIL_RATIO:g}×(长尾)、"
        f"err > {ANOMALY_ERR_PCT:g}%(错误率)、soak 速率衰减 > "
        f"{ANOMALY_DECAY_PCT:g}%;标注是提示,不是 PASS/FAIL;",
        "- 环境注:perf 运行窗口内设备不应叠加其他负载"
        "(测量窗口由操作者确认)。",
    ]

    for area, title in AREA_TITLES.items():
        latency, stream = split_records(cases, area)
        extra = []
        if area == "perf-media":
            extra = ab_section(cases)
        elif area == "perf-soak":
            extra = soak_section(cases)
        if not (latency or stream or extra):
            continue
        lines += ["", f"## {title}", ""]
        if latency:
            lines += latency_table(latency) + [""]
        if stream:
            lines += stream_table(stream) + [""]
        lines += extra

    lines += notes_section(cases)
    lines += ["", "---",
              "回滚:测试用 SDK 经 wheel 强装到设备 venv,恢复用",
              "`/data/venv-sdk/bin/pip install --no-deps "
              "neoruntime-ipc-sdk==0.6.0`(见 scripts/run-device-tests.sh 头注)。"]

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
