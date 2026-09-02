# Proposal: Extended AI Overlay (`AiOverlayConfig` v2)

Status: draft — daemon-side contract proposal, no code yet
Target service: camera-daemon (`aipc.camera.CameraControl`)
SDK layer affected: Python + C++ (`camera.py set_ai_overlay` callers)

## Motivation (the app developer's problem)

The overlay path is the cheapest way for an app to get detections onto
the encoded stream — no video decoding, no re-encoding, one small RPC.
Today it draws rectangles, and that is where it stops. Real detection
apps want:

- **Polygons**: zone definitions (tripwire, region-entry), segmentation
  masks, parking-spot outlines — drawn as filled or outlined polygons,
  not axis-aligned boxes.
- **Tracks and IDs**: object trajectories (a fading trail of
  centroids), stable label badges with IDs that persist across frames.
- **Per-class styling**: `person` red, `vehicle` blue, `face` blurred
  marker — one call, not one call per class.
- **Multiple app sources**: two apps (e.g. people-counter + LPR)
  overlaying the same stream without clobbering each other's settings.
  Today a second `set_ai_overlay` overwrites the first.

None of this needs frame access. It needs the overlay renderer to
accept richer geometry and per-source scoping — a renderer extension,
not a pipeline change.

## Existing path (evidence)

- `AiOverlayConfig` is defined in `camera.proto` and handled in
  `camera_control_service.cpp:259`; the daemon-side subscriber
  machinery is `ai_overlay_subscriber.cpp` (subscription entry at
  `ai_overlay_subscriber.cpp:29`), wired into the daemon main loop at
  `camera_daemon.cpp:3734`. The render site is the same DSP/encoder
  pre-scale stage discussed in `dsp-offload.md`.
- Apps already publish detections through the ai-runtime event path
  (`inference.py` subscribe results); the overlay renderer consumes
  box+label today, so the plumbing from app → daemon exists — only the
  vocabulary (boxes) is limiting.
- The daemon already arbitrates multiple concurrent overlay subscribers
  (`ai_overlay_subscriber` tracks per-client sessions), which is the
  natural place to hang per-app source ids.

## Proposed proto

Extend rather than replace — existing box-only callers must keep
working untouched.

```protobuf
// additions to camera.proto (aipc.camera)

message OverlayPoint {
  float x = 1;                    // normalized 0..1 within stream frame
  float y = 2;
}

message OverlayShape {
  string kind = 1;                // "box" | "polygon" | "polyline"
                                   //          | "circle" | "badge"
  repeated OverlayPoint points = 2;   // box: 2 corners; polygon: N
  string label = 3;               // text; for "badge": the ID text
  float score = 4;                // optional confidence display
  string class_name = 5;          // styling key into color map
}

message OverlayTrack {            // trajectory trail
  repeated OverlayPoint history = 1;  // oldest..newest, capped by daemon
  string track_id = 2;
  string class_name = 3;
}

message AiOverlaySource {         // one app's overlay vocabulary
  string source_id = 1;           // app-scoped, e.g. "people-counter"
  bool enabled = 2;
  repeated OverlayShape shapes = 3;
  repeated OverlayTrack tracks = 4;
}

// extended AiOverlayConfig — additive fields only
message AiOverlayConfig {
  // ... existing fields (enabled, box color/thickness, label style) ...

  // v2 additions:
  repeated AiOverlaySource sources = 10;   // replaces single-box vocab
  map<string, uint32> class_colors = 11;   // class_name -> 0xRRGGBB
  bool draw_scores = 12;
  uint32 trail_length = 13;                // max points kept per track
  float trail_fade = 14;                   // 0=none, 1=full fade-out
}

service addition:
  // per-source CRUD so apps don't rewrite the whole config
  rpc UpdateOverlaySource(AiOverlaySource) returns (Status);
  rpc RemoveOverlaySource(SourceIdRequest) returns (Status);
```

Semantics:

- **Additive**: a config with `sources` unset behaves exactly like
  today's box overlay; set `sources` and the richer renderer engages.
- **Per-source isolation**: `UpdateOverlaySource` merges by
  `source_id`; an app can only mutate its own source (enforced by the
  daemon from the caller's app identity, same as overlay subscription
  auth today).
- **Normalized coordinates** everywhere (`0..1`), matching the existing
  overlay convention and making shapes resolution-independent across
  main/sub streams.

## Rendering constraints and risks

1. **Raster cost**: polygons and trails are more pixels than boxes, but
   the render happens on the pre-encode stage that already touches
   every frame; measured headroom from DPM work suggests shape counts
   in the tens are fine, hundreds are not. Daemon should cap
   `shapes`/`tracks` per source (e.g. 64/16) and reject the rest —
   loud, not silent.
2. **Text rendering**: badges/labels need a font rasterizer. The
   platform OSD already renders text overlays on-stream
   (`OsdTextOverlayConfig` in the same daemon), so the capability
   exists — reuse the OSD font path rather than adding a dependency.
3. **Trail state**: tracks imply daemon-side per-track history buffers.
   Bound total memory (trail_length × tracks) and reset on source
   disable to avoid leaks from long-lived apps.
4. **No DSP dependency for P0**: boxes/polygons/badges are drawable in
   the existing overlay renderer. Only *filled semi-transparent*
   regions and compositing anything beyond simple alpha benefit from
   the DSP `BLEND` op (`dsp-offload.md` P1) — hence P0 ships without
   it.

## Phased rollout

1. **P0**: `sources` with `box`/`polygon`/`polyline`/`badge` shapes,
   `class_colors`, per-source CRUD. Reuse OSD font rasterizer. SDK:
   `set_ai_overlay` gains `shapes=[...]`/`colors={...}` kwargs,
   plus `update_overlay_source()`.
2. **P1**: `OverlayTrack` trails with fade; filled polygons via DSP
   blend; `draw_scores`.
3. **P2**: overlay content driven directly by ai-runtime detection
   events (app opts in by stream+class filter instead of pushing
   shapes per frame).

## Relationship to other proposals

- `dsp-offload.md` P1 (`BLEND`) unlocks filled/alpha compositing here.
- `frame-injection.md` remains the escape hatch for heat maps and
  anything pixel-generated; this proposal covers the 90% vector case at
  a fraction of the cost.
