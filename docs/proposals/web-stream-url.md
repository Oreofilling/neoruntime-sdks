# Proposal: Web Stream URL Negotiation (`GetWebStreamUrl`)

Status: draft — daemon-side contract proposal, no code yet
Target service: app-manager (`aipc.app.AppManager`) — or camera-daemon;
see open question below
SDK layer affected: Python + C++ (`web.py` MJPEG helpers' platform twin)

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

1. **P0**: `GetWebStreamUrl` returning the console's existing HLS URL
   for `main`/`sub`, unsigned, no ttl. Apps stop hard-coding.
2. **P1**: signed URLs with ttl; `accepted` negotiation incl. MJPEG
   fallback served by the platform; `expose_in_console` on
   `RegisterWebUrl`.
3. **P2**: app-defined streams (composed via `frame-injection.md`)
   become addressable the same way.

## Relationship to other proposals

- SDK `web.py` (0.6.0) is the interim self-serve path; this proposal
  supersedes it for WAN-facing pages, not for LAN tools.
- `frame-injection.md` P2 app streams get their URL surface here.
