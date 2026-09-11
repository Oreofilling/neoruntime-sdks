# Proposal: Web Stream URL Negotiation (`GetWebStreamUrl`)

Status: P0 SHIPPED 2026-09-11 as an SDK-side helper — no RPC built (see
"P0 outcome" below); P1/P2 remain future daemon-side work
Target service: app-manager (`aipc.app.AppManager`) — or camera-daemon;
see open question below (only relevant to P1/P2 now)
SDK layer affected: Python shipped (`web.py` `platform_stream_url`);
C++ twin not yet

## Motivation (the app developer's problem)

An app that wants to show its camera view in a browser today has two
options, both wrong:

1. **Serve video itself** — open a port in its container, run an
   MJPEG/HLS server (SDK 0.6.0 `web.py` does exactly this), and teach
   the user to reach that port. It works on the bench and breaks in the
   field: NAT, HTTPS, and the camera's own reverse proxy sit between
   the browser and the app container.
2. **Hard-code the platform console's stream URL** — the web console
   already proxies the camera's HLS to browsers. But the URL layout is
   a platform implementation detail (it has changed across releases),
   so apps that bake it in break silently on upgrade.

Meanwhile the platform already solved browser→camera video: the web
console's HLS path. What is missing is a *contract* letting an app ask
"give me the URL a browser should use to see stream X, through the
platform's own proxy/HTTPS/domain", plus a way to register an app-served
page under the same origin so app UI and platform video compose into one
page without mixed-origin pain.

## Existing platform path (evidence)

- Apps already register web URLs with the platform:
  `AppClient.register_web_url` (SDK `app.py`) → app-manager RPC
  `RegisterWebUrlRequest`; the platform's nginx reverse-proxies the app
  container's HTTP surface under the camera's origin. The proxying
  machinery this proposal needs therefore exists — it is just not
  queryable for video.
- The web console serves the camera's live HLS to browsers today
  (platform `docs/services/media-streaming.md`), including transcoding
  negotiation and auth; an app-reimplemented server cannot inherit
  HTTPS, auth, or NAT traversal, but the console path gets all three
  for free.
- SDK 0.6.0 ships `MjpegServer`/`MjpegStream` for the self-serve path —
  this proposal does not replace it; it removes the need to expose it
  to WANs.

## Proposed proto

```protobuf
// in app.proto (aipc.app) — AppManager already owns app<->web concerns

message WebStreamQuery {
  string stream_id = 1;           // "main" | "sub" | app-defined
  string app_id = 2;              // requesting app, for auth scoping

  // Preferred delivery, in order; platform answers with what it serves
  repeated string accepted = 3;   // ["hls", "mjpeg", "webrtc"]
}

message WebStreamEndpoint {
  bool success = 1;
  string message = 2;

  string url = 3;                 // browser-reachable, same-origin as
                                  // the web console (HTTPS, authed)
  string kind = 4;                // "hls" | "mjpeg" | "webrtc"
  uint32 ttl_seconds = 5;         // signed-URL expiry; re-query after
  map<string, string> extra = 6;  // e.g. {"m3u8": "...", "ts": "..."}
}

service addition:
  rpc GetWebStreamUrl(WebStreamQuery) returns (WebStreamEndpoint);
```

And the sibling that composes app pages with platform video:

```protobuf
message RegisterWebUrlRequest {   // already exists in app.proto
  // ... current fields ...
  // addition: mount under console nav
  bool expose_in_console = 4;     // show link in web console UI
}
```

Semantics:

- **Same-origin guarantee**: returned `url` is relative to the web
  console origin the user already browses (`/api/.../streams/...`),
  so browser auth cookies apply and no CORS/mixed-content handling is
  left to the app.
- **ttl + re-query**: URLs may be signed and short-lived; the SDK
  helper caches and refreshes at `ttl_seconds * 0.8`.
- **Accept negotiation**: platform replies with the best `kind` it
  supports from `accepted`; SDK falls back to `web.py` MJPEG only when
  the RPC answers `success=false`.

## SDK surface once the daemon ships it

```python
url = app_client.get_web_stream_url("main")          # -> str, cached
page_src = app_client.register_web_url(path="/", expose_in_console=True)
```

One call replaces container port exposure for the common
"app page with live video" case.

P0 shipped as a different, RPC-free surface (see "P0 outcome"):

```python
from neoruntime_ipc_sdk import platform_stream_url

url = platform_stream_url("sub", host="192.168.1.10", token=jwt)
# wss://192.168.1.10/api/v1/h264/sub?token=...
```

## Risks and open questions

1. **Which service owns it?** Video originates from camera-daemon, but
   the URL namespace and auth belong to app-manager/nginx. Draft places
   it in app-manager (it can consult camera-daemon internally); if the
   team prefers locality, camera-daemon works and app-manager proxies
   the answer unchanged. Open question for review.
2. **Auth model for signed URLs**: if console sessions expire, embedded
   players must re-fetch. The `ttl` field forces this to be designed up
   front rather than discovered by broken iframes later.
3. **No new hard blocks**: everything here is RPC + nginx config; the
   riskiest item is URL-layout stability, which is precisely what the
   proposal turns into a contract.

## Phased rollout

1. **P0** — SHIPPED, delivered differently than drafted (see "P0
   outcome"): a pure SDK composer instead of `GetWebStreamUrl`. Apps
   stop hard-coding.
2. **P1**: signed URLs with ttl; `accepted` negotiation incl. MJPEG
   fallback served by the platform; `expose_in_console` on
   `RegisterWebUrl`.
3. **P2**: app-defined streams (composed via `frame-injection.md`)
   become addressable the same way.

## P0 outcome (2026-09-11): SDK composer, no RPC

Investigating the shipped platform before building the RPC collapsed
P0 to a client-side helper, `web.py: platform_stream_url()`:

- **The endpoint already exists.** The gateway (nginx, TLS on 443)
  reverse-proxies encoded camera streams at
  `/api/v1/h264/{stream_id}` (WebSocket) to platform-api; the web
  console's player consumes exactly this URL. With `FramePublisher`
  (frame-injection) landed, app content flows through platform
  streams — an app publishing REPLACE/OVERLAY frames needs **no port
  of its own** to be browser-visible.
- **An RPC adds no information.** The server cannot know its own
  external host (multi-interface devices), so host must come from
  caller context regardless; the path is deterministic and now
  test-covered as the contract (SDK 0.7.4, 20 offline tests). Rig
  probes confirmed the auth face: 401 without token, 404 unknown
  path, 301 plain-80 — matching the console's
  `?token=` + wss behavior the helper mirrors.
- **Helper semantics**: `platform_stream_url(stream_id, host=None,
  scheme=None, token=None)`; host defaults to `$AIPC_WEB_HOST` env
  then `localhost` (nothing injects the env yet — documented
  convention); full-origin hosts are accepted with scheme mapped
  http→ws / https,wss→wss; tokens are %-encoded with `%20` for
  spaces (mirrors the console's `encodeURIComponent`).

The RPC shape above stays as the draft for P1 (signed ttl URLs,
accept negotiation, `expose_in_console`) and P2 (app-defined stream
addressing), where server-side work is genuinely required.

## Relationship to other proposals

- SDK `web.py` (0.6.0) is the interim self-serve path; this proposal
  supersedes it for WAN-facing pages, not for LAN tools.
- `frame-injection.md` P2 app streams get their URL surface here.
