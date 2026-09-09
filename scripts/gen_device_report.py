#!/usr/bin/env python3
"""Render the on-device SDK interface-test report as Markdown.

Usage::

    python3 scripts/gen_device_report.py dist/device-report.json \
        [--output docs/test-reports/<date>-python-sdk-<ver>-device.md]

Input is the JSON produced by ``python/tests/device/run_device_tests.py``
(schema ``neoruntime-device-test-report/1``). Output layout:

* 结论 + per-area verdict 汇总表
* 接口 × 结论矩阵(含耗时)
* 失败/错误明细(完整错误栈)
* KNOWN-ISSUE / N/A 独立区
* 公开 API 覆盖率(按导出类的公开方法)
* 环境块 + 回滚说明

脱敏规约(仓库 b373aad):本工具对一切渲染文本做 IPv4 擦除,
意外出现的设备地址一律替换为「目标设备」。
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
VERDICTS = ["PASS", "FAIL", "ERROR", "KNOWN-ISSUE", "SKIP-NA", "SKIP"]
HARD = ("FAIL", "ERROR")
AREA_TITLES = OrderedDict([
    ("env", "环境勘探"),
    ("inference", "推理"),
    ("genai", "推理 GenAI"),
    ("media", "媒体 fd"),
    ("events", "事件"),
    ("device", "设备控制"),
    ("camera", "相机"),
    ("dsp", "DSP"),
    ("audio", "音频"),
    ("overlay", "叠加"),
    ("app", "应用"),
    ("plugin", "插件"),
    ("toolkit", "工具箱"),
    ("recording_web", "录制/Web"),
])


def scrub(value) -> str:
    """Any value → one line of Markdown-safe, IP-free text."""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return IP_RE.sub("目标设备", text)


def fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000.0:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def fmt_bytes(n) -> str:
    if not isinstance(n, (int, float)) or n < 0:
        return str(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GiB"


# --------------------------------------------------------------------------
# Coverage: public methods of every exported SDK class vs exercised marks.
# --------------------------------------------------------------------------

def public_api_surface() -> "OrderedDict[str, list[str]]":
    """Exported class → its public callable names (sorted)."""
    import neoruntime_ipc_sdk as sdk

    surface: OrderedDict[str, list[str]] = OrderedDict()
    for name in getattr(sdk, "__all__", dir(sdk)):
        if name.startswith("_"):
            continue
        obj = getattr(sdk, name)
        if not inspect.isclass(obj):
            continue
        methods = [m for m in dir(obj)
                   if not m.startswith("_") and callable(getattr(obj, m, None))]
        if methods:
            surface[name] = sorted(methods)
    return surface


def _mark_pairs(interface: str, class_names):
    """(class, method) pairs from a mark string.

    Marks come in two shapes: 'FdMediaClient.get_frame(keep_fd=True)' and
    area-prefixed 'recording.TsWriter.write' or
    'web.MjpegStream.push_jpeg/latest/wait_new'. Segments are split on
    dots (and spaces); any segment that names an exported class claims
    the NEXT segment as its method list, where 'a/b/c' counts as three
    methods. Non-existent methods simply never match the surface, so
    free-text segments are harmless.
    """
    token = interface.split("(")[0].strip()
    segs = token.replace(" ", ".").split(".")
    for i, seg in enumerate(segs[:-1]):
        if seg not in class_names:
            continue
        for method in segs[i + 1].split("/"):
            method = re.sub(r"[^0-9a-zA-Z_].*$", "", method)
            if method:
                yield seg, method


def coverage_section(cases: list[dict]) -> list[str]:
    lines = ["## 公开 API 覆盖率", ""]
    try:
        surface = public_api_surface()
    except Exception as exc:  # noqa: BLE001 — coverage must not kill the report
        return lines + [f"无法枚举本地 SDK 公开面:{scrub(exc)}", ""]

    class_names = frozenset(surface)
    exercised = set()
    for case in cases:
        for pair in _mark_pairs(case.get("interface") or "", class_names):
            exercised.add(pair)

    total = covered = 0
    rows = []
    for cls, methods in surface.items():
        hit = [m for m in methods if (cls, m) in exercised]
        total += len(methods)
        covered += len(hit)
        rows.append((cls, len(hit), len(methods)))

    pct = (100.0 * covered / total) if total else 0.0
    lines += [
        f"按导出类公开方法统计:**{covered}/{total}({pct:.1f}%)** "
        "在真机用例中被直接调用(矩阵中 owner.method 形态的接口标记);",
        "以自由文本标记的用例(如 HTTP round-trip)不计入分子,属保守口径。",
        "",
        "| 导出类 | 实测方法 | 公开方法 | 覆盖率 |",
        "|---|---:|---:|---:|",
    ]
    for cls, hit, tot in sorted(rows, key=lambda r: (-r[1], r[0])):
        pct = (100.0 * hit / tot) if tot else 0.0
        lines.append(f"| `{cls}` | {hit} | {tot} | {pct:.0f}% |")
    return lines + [""]


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------

def ops_notes_section(notes) -> list[str]:
    """Operational events during the run (daemon wedges, reboots, manual
    recoveries) — injected by the merge step as report['ops_notes']."""
    if not notes:
        return []
    lines = ["## 环境事件与恢复记录", ""]
    for note in notes:
        lines.append(f"- {scrub(note)}")
    return lines + [""]


def summary_tables(cases: list[dict]) -> list[str]:
    areas = OrderedDict()
    for case in cases:
        areas.setdefault(case["area"], []).append(case)

    header = ("| 区域 | 用例 | " + " | ".join(VERDICTS) + " | 耗时 |",
              "|---|---:" + "|---:" * len(VERDICTS) + "|---:|")
    lines = ["## 汇总", "", *header]
    totals = {v: 0 for v in VERDICTS}
    for area, rows in areas.items():
        counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
        for v, n in counts.items():
            totals[v] += n
        wall = sum(r.get("duration_ms") or 0 for r in rows)
        title = AREA_TITLES.get(area, area)
        lines.append(
            f"| {title} `{area}` | {len(rows)} | "
            + " | ".join(str(counts[v] or "·") for v in VERDICTS)
            + f" | {fmt_ms(wall)} |")
    wall_all = sum(r.get("duration_ms") or 0 for r in cases)
    lines.append(
        f"| **合计** | **{len(cases)}** | "
        + " | ".join(f"**{totals[v]}**" if totals[v] else "·"
                    for v in VERDICTS)
        + f" | {fmt_ms(wall_all)} |")
    return lines + [""]


def matrix_section(cases: list[dict]) -> list[str]:
    areas = OrderedDict()
    for case in cases:
        areas.setdefault(case["area"], []).append(case)

    lines = ["## 接口 × 结论矩阵", ""]
    for area, rows in areas.items():
        title = AREA_TITLES.get(area, area)
        lines += [f"### {title} `{area}`", "",
                  "| 用例 | 接口 | 结论 | 耗时 | 备注 |",
                  "|---|---|---|---:|---|"]
        for r in rows:
            note = scrub(r.get("note") or "")[:80]
            lines.append(
                f"| `{r['id'].rsplit('.', 1)[-1]}` "
                f"| `{scrub(r['interface'])[:70]}` "
                f"| **{r['verdict']}** | {fmt_ms(r.get('duration_ms'))} "
                f"| {note} |")
        lines.append("")
    return lines


def failure_section(cases: list[dict], verdict_key: str, title: str) -> list[str]:
    rows = [c for c in cases if c["verdict"] == verdict_key]
    if not rows:
        return [f"## {title}", "", "无。", ""]
    lines = [f"## {title}", ""]
    for r in rows:
        lines += [f"### {r['id']}", "",
                  f"- 接口:`{scrub(r['interface'])}`",
                  f"- 结论:**{r['verdict']}**,耗时 {fmt_ms(r.get('duration_ms'))}"]
        if r.get("note"):
            lines.append(f"- 备注:{scrub(r['note'])}")
        if r.get("evidence"):
            lines.append("- 证据:")
            for key in sorted(r["evidence"]):
                lines.append(f"  - `{key}`: {scrub(r['evidence'][key])[:200]}")
        if r.get("error"):
            lines += ["- 错误栈:", "", "```python",
                      scrub(r["error"]).strip() or "(空)", "```"]
        lines.append("")
    return lines


def _ki_reason(case: dict) -> str:
    """KNOWN-ISSUE attribution: the decorator stores the reason in the
    outcome note as 'known issue reproduced: <reason>'."""
    note = case.get("note") or ""
    prefix = "known issue reproduced: "
    at = note.find(prefix)
    return note[at + len(prefix):].strip() if at >= 0 else (note or "未标注")


def known_issue_section(cases: list[dict]) -> list[str]:
    """KNOWN-ISSUE grouped by root cause: one attribution table first,
    per-case reproduction detail after — 7 cases sharing one daemon
    defect should read as one finding, not seven."""
    rows = [c for c in cases if c["verdict"] == "KNOWN-ISSUE"]
    if not rows:
        return ["## 已知问题(KNOWN-ISSUE)", "", "无。", ""]
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for case in rows:
        grouped.setdefault(_ki_reason(case), []).append(case)

    lines = ["## 已知问题(KNOWN-ISSUE,按根因归组)", ""]
    for reason, group in grouped.items():
        ifaces = ", ".join(sorted({(c.get("interface") or "").split("(")[0]
                                    for c in group}))
        lines.append(f"### {scrub(reason)[:400]}")
        lines += ["",
                  f"影响 {len(group)} 个用例:{scrub(ifaces)[:500]}", ""]
        for r in group:
            lines.append(f"- `{r['id'].rsplit('.', 1)[-1]}` — {fmt_ms(r.get('duration_ms'))}"
                         f",复现错误:`{scrub(r.get('error') or '')[:200]}`")
        lines.append("")
    return lines


def na_section(cases: list[dict]) -> list[str]:
    rows = [c for c in cases if c["verdict"] in ("SKIP-NA", "SKIP")]
    if not rows:
        return ["## 不适用 / 未执行(SKIP-NA / SKIP)", "", "无。", ""]
    lines = ["## 不适用 / 未执行(SKIP-NA / SKIP)", "",
             "| 用例 | 接口 | 原因 |",
             "|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['id'].rsplit('.', 1)[-1]}` "
                     f"| `{scrub(r['interface'])[:60]}` "
                     f"| {scrub(r.get('note') or '')[:120]} |")
    return lines + [""]


def env_section(env: dict) -> list[str]:
    node = env.get("node", {})
    sdk = env.get("sdk", {})
    lines = ["## 环境", "",
             "| 项 | 值 |",
             "|---|---|",
             f"| 报告生成 | {scrub(env.get('generated_at'))} |",
             f"| 内核/架构 | {scrub(node.get('uname'))} |",
             f"| Python | {scrub(node.get('python'))} |",
             f"| 系统 | {scrub(node.get('os_release'))} |",
             f"| SDK 版本 | `{scrub(sdk.get('version'))}` |",
             f"| SDK 安装路径 | `{scrub(sdk.get('module_path'))}` |", ""]

    daemons = env.get("daemons", {})
    if daemons:
        lines += ["### 守护进程", ""]
        for daemon, procs in daemons.items():
            for proc in procs:
                lines.append(f"- **{daemon}** pid {scrub(proc.get('pid'))} — "
                             f"`{scrub(proc.get('args'))[:160]}`")
        lines.append("")

    sockets = env.get("sockets", {})
    if sockets:
        lines += ["### UDS 端点", ""]
        for sock in sorted(sockets):
            lines.append(f"- `{sock}`")
        lines.append("")

    models = env.get("models", [])
    if models:
        lines += ["### 模型", "",
                  "| 路径 | 大小 |", "|---|---:|"]
        for m in models:
            lines.append(f"| `{scrub(m.get('path'))}` "
                         f"| {fmt_bytes(m.get('size_bytes'))} |")
        lines.append("")
    return lines


def render(report: dict, output_hint_version: str) -> str:
    cases = report.get("cases", [])
    summary = report.get("summary", {})
    verdicts = summary.get("verdicts", {})
    env = report.get("env", {})
    hard = sum(verdicts.get(v, 0) for v in HARD)
    today = datetime.now().strftime("%Y-%m-%d")
    sdk_ver = env.get("sdk", {}).get("version", output_hint_version)

    if hard == 0:
        conclusion = ("**通过** — 无 FAIL/ERROR。"
                      "SKIP-NA 为环境无承载(如实记录),KNOWN-ISSUE 见独立区。")
    else:
        conclusion = (f"**存在 {hard} 个 FAIL/ERROR** — 明细见下,需按归因跟进。")

    out = [
        f"# Python SDK 真机接口测试报告({today},SDK {scrub(sdk_ver)})",
        "",
        f"> 目标设备端到端执行 `python/tests/device/` 全套件;"
        f"报告数据源 `device-report.json`(schema {scrub(report.get('schema'))})。",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        f"- 用例总数:**{summary.get('total', len(cases))}**"
        f"(PASS {verdicts.get('PASS', 0)} / FAIL {verdicts.get('FAIL', 0)}"
        f" / ERROR {verdicts.get('ERROR', 0)} / KNOWN-ISSUE "
        f"{verdicts.get('KNOWN-ISSUE', 0)} / SKIP-NA {verdicts.get('SKIP-NA', 0)}"
        f" / SKIP {verdicts.get('SKIP', 0)})",
        f"- 端到端耗时:{fmt_ms((summary.get('wall_time_s') or 0) * 1000.0)}",
    ]
    modules = summary.get("modules") or []
    if modules:
        out.append(f"- 执行模块:{len(modules)} 个(每模块独立进程 + 看门狗)")
        missing = summary.get("modules_without_cases") or []
        if missing:
            out.append(
                "- ⚠️ 以下模块没有记录到任何用例(进程被看门狗杀掉或崩溃,"
                "该区域实际未测):" + ", ".join(f"`{scrub(m)}`" for m in missing))
    out += [
        *matrix_section(cases),
        *ops_notes_section(report.get("ops_notes")),
        *failure_section(cases, "FAIL", "失败明细(FAIL)"),
        *failure_section(cases, "ERROR", "错误明细(ERROR)"),
        *known_issue_section(cases),
        *na_section(cases),
        *coverage_section(cases),
        *env_section(env),
        "## 回滚说明",
        "",
        "设备 venv 已由编排脚本安装被测版本;如需恢复:",
        "",
        "```bash",
        "ssh <target> '/data/venv-sdk/bin/pip install --no-deps "
        "neoruntime-ipc-sdk==<previous-version>'",
        "```",
        "",
        "镜头物理状态:设备控制区用例结束后已执行 `lens_reset_zero()` "
        "并轮询校验归位。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="device-report.json 路径")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 Markdown 路径(默认按日期+版本命名)")
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as fh:
        report = json.load(fh)
    if report.get("schema") != "neoruntime-device-test-report/1":
        print(f"警告:未知 schema {report.get('schema')!r},继续渲染", file=sys.stderr)

    today = datetime.now().strftime("%Y-%m-%d")
    version = report.get("env", {}).get("sdk", {}).get("version", "unknown")
    output = args.output or (
        PROJECT_ROOT / "docs" / "test-reports"
        / f"{today}-python-sdk-{version}-device.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report, version), encoding="utf-8")

    verdicts = report.get("summary", {}).get("verdicts", {})
    print(f"report: {output}")
    print(f"summary: {json.dumps(verdicts)} "
          f"total={report.get('summary', {}).get('total')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
