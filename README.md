# pi-gemini-router

Adds two router models to pi's `google` provider that walk a model ladder —
and rotate API keys — until something answers, and patches the Google
provider so it works in interactive mode at all.

```
google/gemini-flash        gemini-flash-latest → 3.8 → 3.7 → 3.6 → 3.5 → 3
google/gemini-flash-lite   gemini-flash-lite-latest → 3.5-flash-lite → 3.1-flash-lite
```

## Behavior

- Each ladder entry is tried on the current multi-account key
  (`google-key1..N`, registered by `pi-multi-account`).
- The router holds its `(model, key)` pair until that pair **errors**; success,
  user messages and continuations do not move it.
- On failure it tries the **same model on the next key**; only after every key
  has failed for that model does it move down the model ladder (restarting on
  key1).
- Missing `(model, key)` pairs are skipped (a key whose project lacks a model).
- `-latest` aliases head the ladder and share the concrete model's quota bucket,
  which is why they are tried first rather than counted as extra capacity.
- A walk covers every `(model, key)` pair exactly once. When it is complete the
  error surfaces; the next prompt starts a fresh walk.

## Quota-exhaustion rotation

Pi's own retry classifier treats quota exhaustion (429 `RESOURCE_EXHAUSTED`)
as a billing condition and never retries it, and `pi-multi-account` used to
replace the router's virtual model with a physical `google-keyN` on failure —
so an exhausted key ended the run instead of rotating to the next one.

The router therefore takes over at `agent_before_settle`: on a rotatable
failure (quota, rate limit, overload, 5xx, or an invalid key) it advances the
router state to the next `(model, key)` pair, drops the failed attempt from the
context, and asks for one more request. The walk stops once every pair has
been tried.

`pi-multi-account` no longer hijacks a selected router model either: a
virtual (`pi-virtual`) selection keeps its router, because the router picks
the account per request.

## The abort-signal and URL fixes

Interactive (TTY) turns die with `Error: callerSignal.addEventListener is not
a function` — and once that is fixed, every request still 404s. Three stacked
problems, all patched here by re-registering `google` and every `google-keyN`
provider with a guarded `streamSimple`:

1. **Duck-typed turn signals.** TTY turn options carry a truthy abort signal
   with `.aborted` but no `addEventListener`; the genai client calls
   `callerSignal.addEventListener(...)` and dies. Dropped (kept as a real
   `AbortSignal.abort()` when already aborted) in `signal.ts`.
2. **Payload clones strip the signal.** `before_provider_request` handlers
   rebuild the request — pi-output-cap, for one, runs it through
   `structuredClone`, and Node's structuredClone degrades an AbortSignal to a
   plain `{}` (verified on node 26). The adapter's `onPayload` hook is wrapped
   so the original signal is restored after any transform.
3. **Unversioned request URLs.** pi-ai's `createClient` sets `apiVersion: ""`
   whenever `model.baseUrl` is set ("baseUrl already includes version path"),
   but the catalog's Google baseUrl is the bare host — the SDK then builds
   `…googleapis.com/models/…` without `/v1beta` and Google answers 404 for
   every request. The adapter now receives a model whose baseUrl carries the
   version.

The wrapper reaches the adapter through `@earendil-works/pi-ai/compat`
(`googleGenerativeAIApi()`); the eager
`@earendil-works/pi-ai/api/google-generative-ai` specifier is not in pi's
extension module table (VIRTUAL_MODULES / aliases), so importing it fails the
whole extension at load time in compiled builds.

Rationale: AI Studio free-tier quotas are per model *and* per project — Flash
models get 20 RPD each, Flash Lite 500 RPD each — so 7 projects × 6 flash
models is ~840 flash requests/day before key rotation.

## Commands

```
/gemini-router          # show ladders and discovered google-keyN providers
/gemini-router reset    # restart at gemini-flash-latest on key1 (post-quota-reset)
```

Router state lives on the session branch, so it survives compaction and resumes;
`reset` bumps an epoch that forces the next request to the ladder head.

## Install

Symlinked into `~/.pi/agent/extensions/pi-gemini-router` like the other pi
extensions. Requires the `google` accounts in
`~/.pi/agent/multi-account.json` (the `pi-multi-account` extension).

## Typecheck

```bash
cd ~/projects/pi-gemini-router && ./node_modules/.bin/tsc --noEmit
cd ~/projects/pi-gemini-router && bun signal.test.ts
```
