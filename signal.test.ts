// bun signal.test.ts — pure asserts, no network, no credentials.
// Covers the sanitize contract plus the module-resolution regression that made
// the whole extension fail to load in compiled pi builds.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sanitizeSignal } from "./signal.ts";

type Opts = { apiKey?: string; signal?: unknown };

// 1. Duck-typed, not aborted: signal dropped, other options survive.
const out1 = sanitizeSignal({ apiKey: "k", signal: { aborted: false } } as Opts);
assert.equal(out1?.signal, undefined, "duck signal must be dropped");
assert.equal(out1?.apiKey, "k", "other options survive");

// 2. Duck-typed, already aborted: replaced by a real aborted signal.
const out2 = sanitizeSignal({ signal: { aborted: true } } as Opts);
const s2 = out2?.signal as AbortSignal | undefined;
assert.ok(s2 instanceof AbortSignal, "aborted duck becomes a real AbortSignal");
assert.equal(s2?.aborted, true);

// 3. Real signal: options returned untouched, same references.
const real = new AbortController().signal;
const realOpts: Opts = { apiKey: "k", signal: real };
const out3 = sanitizeSignal(realOpts);
assert.equal(out3, realOpts, "real signal: options object returned as-is");
assert.equal(out3?.signal, real);

// 4. No signal / no options: untouched.
const noSignal: Opts = { apiKey: "k" };
assert.equal(sanitizeSignal(noSignal), noSignal);
assert.equal(sanitizeSignal(undefined), undefined);

// 5. Resolution regression: the eager pi-ai api subpath is not in pi's
// extension module table (verified against VIRTUAL_MODULES/aliases), so the
// adapter must be reached through the compat entrypoint instead.
const src = readFileSync(new URL("./index.ts", import.meta.url), "utf-8");
assert.ok(
  !/from "@earendil-works\/pi-ai\/api\//.test(src),
  "eager @earendil-works/pi-ai/api/* import must not come back — it fails to resolve in compiled pi builds",
);
assert.ok(
  src.includes(`from "@earendil-works/pi-ai/compat"`),
  "the google adapter must be reached through the compat entrypoint",
);
assert.ok(/sanitizeSignal\(/.test(src), "streamSimple wrapper must sanitize signals");

console.log("OK — sanitize contract holds, adapter reached via compat entrypoint");
