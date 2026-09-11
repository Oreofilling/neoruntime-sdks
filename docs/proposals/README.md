# Daemon Contract Proposals

Design proposals for daemon-side capabilities the SDK cannot add on its
own. Implementation status varies; consult each proposal's status and
dated verification records. They capture app-developer needs, contracts,
hardware/code evidence, and phased rollouts so the daemon team can size
the remaining work.

| Proposal | Adds | Depends on | Cost |
|---|---|---|---|
| [dsp-offload.md](dsp-offload.md) | `SubmitDspJob` / `AllocateDspBuffer` — DSP geometry + blend for apps | none | RPC + scheduler |
| [frame-injection.md](frame-injection.md) | `PushFrame` — app frames into the encoded stream | dsp-offload (P1) | media-graph wiring |
| [web-stream-url.md](web-stream-url.md) | `GetWebStreamUrl` — console-origin video URLs for app pages | none | RPC + nginx |
| [ai-overlay-extended.md](ai-overlay-extended.md) | `AiOverlayConfig` v2 — polygons, tracks, per-app sources | dsp-offload (P1, optional) | renderer extension |
| [hardware-first-roadmap.md](hardware-first-roadmap.md) | post-P0 work breakdown: HAL + platform + SDK items | dsp-offload experiments | cross-team checklist |
| [sdk-hardware-routing.md](sdk-hardware-routing.md) | SDK capability router + `convert_hw` (both landed) + asks: NMS params, JPEG encode | dsp-offload (P2 items only) | SDK + ai-runtime |
| [composable-pipeline-contracts.md](composable-pipeline-contracts.md) | umbrella: capture→infer→draw→output contract status, gaps, and the P0–P2 priority order | all of the above | cross-team contract doc |
| [video-fd-output-isolation.md](video-fd-output-isolation.md) | source/output buffer isolation, explicit overlay submission, frame synchronization, and DSP requirements (中文) | DSP buffers + encoder-feed changes | camera-daemon + HAL + SDK |

Suggested reading order for reviewers: `composable-pipeline-contracts`
(umbrella — what every step must obey, and the order to build it in) →
`dsp-offload` (foundation) →
`hardware-first-roadmap` (what the experiments commit each layer to) →
`ai-overlay-extended` (drawing vocabulary) →
`video-fd-output-isolation` (source isolation and frame synchronization) → `frame-injection` →
`web-stream-url`.

Evidence conventions: `file:line` references point into the platform
repo (`ne503-aipc`); device probes are dated and were read-only.
