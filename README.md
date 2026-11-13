# pi-gemini-router

Adds two router models to pi's `google` provider that walk a model ladder —
and rotate API keys — until something answers, and patches the Google
provider so it works in interactive mode at all.

```
google/gemini-flash        gemini-4-hydrogen → flash-latest → 3.8 → 3.7 → 3.6 → 3.5 → 3
google/gemini-flash-lite   gemini-flash-lite-latest → 3.5-flash-lite → 3.1-flash-lite
```

## Configuration

Optional `~/.pi/agent/gemini-router.json` overrides the ladders and the
advance order (picked up on session start):

```json
{
  "ladders": {
    "gemini-flash": ["gemini-4-hydrogen", "gemini-flash-latest", "gemini-3.8-flash"],
    "gemini-flash-lite": ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
  },
  "order": "key-first"
}
```

`order` is `key-first` (default: exhaust all keys on one model before the
next model) or `model-first` (walk the ladder per key before the next key).
`quotaAware: false` turns quota marking off.

## Quota awareness

Free-tier quotas are per model *and* per project (one project per
multi-account key), and the daily bucket resets at midnight Pacific. A 429
therefore means "this pair is done for hours" — so the router records hits
per `(model, key)` pair in `~/.pi/agent/gemini-router-quota.json` and skips
dead pairs instead of re-burning a request on each of them every prompt:

- **RPD** (`PerDay` / "per day" in the error body) → pair dead until the next
  midnight PT; the mark persists across restarts and expires by itself.
- **Per-minute** (RPM/TPM — Google's body usually says `Please retry in Xs`)
  → pair cooling for the hint duration (capped at 120s).
- **Ambiguous quota 429** → *possibly RPD*: skipped with a 30-minute probe
  TTL, then retried.
- A **successful response** on a pair clears its marks (self-healing).
- If every pair ends up marked, the walk falls back to the held pair and lets
  the network decide — stale marks never brick the router.

`/gemini-router` shows the current marks (RPD-dead / possibly RPD / cooling);
`/gemini-router quota-reset` clears them.

## Behavior

- Each ladder entry is tried on the current multi-account key
  (`google-key1..N`, registered by `pi-multi-account`).
- The router holds its `(model, key)` pair until that pair **errors**; success,
  user messages and continuations do not move it.
- On failure it tries the **same model on the next key**; only after every key
  has failed for that model does it move down the model ladder (restarting on
  key1). `order: "model-first"` flips this (ladder first, then keys).
- Missing `(model, key)` pairs are skipped (a key whose project lacks a model).
- `-latest` aliases sit near the ladder head and share the concrete model's
  quota bucket, which is why they are tried early rather than counted as
  extra capacity.
- A walk covers every `(model, key)` pair exactly once. When it is complete the
  error surfaces; the next prompt starts a fresh walk.

## Quota-exhaustion rotation

Pi's own retry classifier treats quota exhaustion (429 `RESOURCE_EXHAUSTED`)
as a billing condition and never retries it, and `pi-multi-account` used to
replace the router's virtual model with a physical `google-keyN` on failure —
so an exhausted key ended the run instead of rotating to the next one.

The router therefore takes over at `agent_before_settle`: on a rotatable
failure (quota, rate limit, overload, 5xx, an invalid key, or a pair-specific
400/404 — model not available for that project, strict signature rejection,
retired model) it advances the router state to the next `(model, key)` pair,
drops the failed attempt from the context, and asks for one more request. The
walk stops once every pair has been tried. Context-overflow 400s do **not**
rotate: no pair can serve them, so they are left to pi's compaction.

## The thought-signature fix

pi-ai only replays `thoughtSignature`s when the history comes from the exact
same provider+model (`resolveThoughtSignature`), so every model switch — the
router's core job — leaves all prior `functionCall` parts unsigned, and strict
Gemini 3/4 models answer 400 `Function call is missing a thought_signature`.
Google special-cases the sentinel value `skip_thought_signature_validator`
(verified live: any other non-base64 value fails the TYPE_BYTES base64 check;
the sentinel passes and is accepted by 3.5-flash and 4-hydrogen). The wrapped
`onPayload` injects it into every unsigned function call of Gemini 3+ models
before the request is sent.

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
/gemini-router             # show ladders, advance order, discovered keys, quota marks
/gemini-router reset       # restart at the ladder head on key1 (post-quota-reset)
/gemini-router quota-reset # clear all quota marks (pairs considered live again)
```

Router state lives on the session branch, so it survives compaction and resumes;
`reset` bumps an epoch that forces the next request to the ladder head.

## Install

Symlinked into `~/.pi/agent/extensions/pi-gemini-router` like the other pi
extensions. Requires the `google` accounts in
`~/.pi/agent/multi-account.json` (the `pi-multi-account` extension).

## Typecheck & smoke

```bash
cd ~/Projects/pi-gemini-router && ./node_modules/.bin/tsc --noEmit
scripts/smoke.sh                # routing ladder, rotation, rotatable errors
scripts/smoke.sh model-first    # config override case (order + custom ladder)
scripts/smoke.sh quota          # quota awareness (marks, skipping, self-heal, fallback)
cd ~/Projects/pi-gemini-router && bun signal.test.ts
```
