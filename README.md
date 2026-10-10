# pi-gemini-router

Adds two router models to pi's `google` provider that walk a model ladder —
and rotate API keys — until something answers, and patches the Google
provider so it works in interactive mode at all.

```
google/gemini-flash        gemini-flash-latest → 3.8 → 3.7 → 3.6 → 3.5 → 3 → 2.5
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

## The interactive-mode signal fix

Interactive mode passes a duck-typed abort signal into turn options: truthy,
with `.aborted`, but no `addEventListener`. pi-ai's Google adapter forwards
`options.signal` into genai's `config.abortSignal`, where the SDK's
`createAttemptSignal(timeout, callerSignal)` calls
`callerSignal.addEventListener(...)`. Every Google model then fails with

```
Error: callerSignal.addEventListener is not a function
```

Headless runs pass real AbortSignals, so only the TUI hits this. The extension
re-registers `google` and every `google-keyN` provider with a `streamSimple`
wrapper that drops duck-typed signals (keeping already-aborted ones as a real
`AbortSignal.abort()`); real signals pass through untouched.

The wrapper reaches the Google adapter through `@earendil-works/pi-ai/compat`
(`googleGenerativeAIApi().streamSimple`). Extension imports resolve against a
fixed module table in compiled pi builds, and the eager
`@earendil-works/pi-ai/api/google-generative-ai` specifier is not in it —
importing it there fails the whole extension at load time, which is why an
earlier revision of this fix never took effect.

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
