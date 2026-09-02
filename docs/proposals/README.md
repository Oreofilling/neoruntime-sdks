# Daemon Contract Proposals

Design proposals for daemon-side capabilities the SDK cannot add on its
own. None of these are implemented; each captures the app-developer
need, a proto draft, hardware/code evidence from the platform repo and
device 192.168.93.72, and a phased rollout so the daemon team can size
the work.

| Proposal | Adds | Depends on | Cost |
|---|---|---|---|
| [dsp-offload.md](dsp-offload.md) | `SubmitDspJob` / `AllocateDspBuffer` — DSP geometry + blend for apps | none | RPC + scheduler |
| [frame-injection.md](frame-injection.md) | `PushFrame` — app frames into the encoded stream | dsp-offload (P1) | media-graph wiring |
| [web-stream-url.md](web-stream-url.md) | `GetWebStreamUrl` — console-origin video URLs for app pages | none | RPC + nginx |
| [ai-overlay-extended.md](ai-overlay-extended.md) | `AiOverlayConfig` v2 — polygons, tracks, per-app sources | dsp-offload (P1, optional) | renderer extension |
| [hardware-first-roadmap.md](hardware-first-roadmap.md) | post-P0 work breakdown: HAL + platform + SDK items | dsp-offload experiments | cross-team checklist |

Suggested reading order for reviewers: `dsp-offload` (foundation) →
`hardware-first-roadmap` (what the experiments commit each layer to) →
`ai-overlay-extended` (cheapest win) → `frame-injection` →
`web-stream-url`.

Evidence conventions: `file:line` references point into the platform
repo (`ne503-aipc`); device probes are dated and were read-only.
