// Smoke test for pi-gemini-router: stubs the registry and exercises the
// ladder/rotation logic without touching the network.
//
// Run with node's type stripping and pi's node_modules on the resolution path
// (see scripts/smoke.sh). GEMINI_ROUTER_SMOKE selects the case:
//   model-first — config override case (seeded config with order "model-first")
//   quota       — quota awareness: marking, skipping, self-healing, fallback
// HOME is redirected by smoke.sh for every case, so config/quota files never
// touch the real ~/.pi/agent.
import activate, { isRotatableError, restoreXhighWire, xhighModel } from "./index.ts";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const virtualModels = new Map<string, { provider: string; id: string; route: (r: unknown, c: unknown) => unknown }>();
const commands: string[] = [];
const handlers = new Map<string, (event: never, ctx: never) => unknown>();

const pi = {
  registerProvider: () => {},
  registerVirtualModel: (def: { provider: string; id: string }) => {
    virtualModels.set(def.provider + "/" + def.id, def as never);
  },
  registerCommand: (name: string) => commands.push(name),
  on: (name: string, fn: (event: never, ctx: never) => unknown) => {
    handlers.set(name, fn);
  },
  appendEntry: () => {},
} as never;

activate(pi);

if (virtualModels.size !== 2) throw new Error("expected 2 virtual models, got " + virtualModels.size);
if (commands.join(",") !== "gemini-router") throw new Error("command not registered");

// Stub registry: 7 google-keyN providers, each with the full flash ladders.
const FLASH = [
  "gemini-4-hydrogen",
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3-flash-preview",
];
const FLASH_LITE = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"];
const ALL = new Map<string, { provider: string; id: string }>();
for (let k = 1; k <= 7; k++) {
  for (const id of [...FLASH, ...FLASH_LITE]) {
    ALL.set(`google-key${k}/${id}`, { provider: `google-key${k}`, id });
  }
}
const ctx = {
  modelRegistry: {
    find: (p: string, id: string) => ALL.get(`${p}/${id}`),
    getAll: () => Array.from(ALL.values()),
  },
};

const flash = virtualModels.get("google/gemini-flash")!;
const lite = virtualModels.get("google/gemini-flash-lite")!;

function route(model: typeof flash, request: Record<string, unknown>) {
  return model.route(request, ctx) as { model: { provider: string; id: string }; state: { m: number; k: number } };
}

if (process.env.GEMINI_ROUTER_SMOKE === "xhigh") {
	// xhigh support: the virtual model offers xhigh, the clone points at
	// v1alpha with a resolver-legal map, and the wire rewrite restores XHIGH.
	const levels = (flash as unknown as { thinkingLevels: string[] }).thinkingLevels;
	assert(levels.includes("xhigh"), "virtual flash model offers xhigh");
	const liteLevels = (lite as unknown as { thinkingLevels: string[] }).thinkingLevels;
	assert(liteLevels.includes("xhigh"), "virtual flash-lite model offers xhigh");

	const hydrogen = { provider: "google-key1", id: "gemini-4-hydrogen", reasoning: true, thinkingLevelMap: { off: null, low: "low", medium: "medium", high: "high", xhigh: "xhigh" } };
	const clone = xhighModel(hydrogen as never) as Record<string, unknown>;
	assertEq(clone.baseUrl, "https://generativelanguage.googleapis.com/v1alpha", "clone routes to v1alpha");
	assertEq((clone.thinkingLevelMap as Record<string, unknown>).xhigh, "high", "clone resolves xhigh legally");
	assertEq(clone.id, "gemini-4-hydrogen", "clone keeps the model id");
	assertEq((clone.thinkingLevelMap as Record<string, unknown>).high, "high", "clone keeps other mappings");
	assertEq((hydrogen as Record<string, unknown>).baseUrl, undefined, "original model untouched");

	const payload = { config: { thinkingConfig: { thinkingLevel: "HIGH" } } };
	restoreXhighWire(payload);
	assertEq(payload.config.thinkingConfig.thinkingLevel, "XHIGH", "wire level restored to XHIGH");
	restoreXhighWire(payload);
	assertEq(payload.config.thinkingConfig.thinkingLevel, "XHIGH", "rewrite is idempotent");
	const budgetPayload = { config: { thinkingConfig: { thinkingBudget: 1234 } } };
	restoreXhighWire(budgetPayload);
	assertEq((budgetPayload.config.thinkingConfig as Record<string, unknown>).thinkingLevel, undefined, "budget configs untouched");
	assertEq(budgetPayload.config.thinkingConfig.thinkingBudget, 1234, "budget value untouched");
	console.log("OK — xhigh virtual levels, v1alpha clone, and wire rewrite all pass");
} else if (process.env.GEMINI_ROUTER_SMOKE === "model-first") {
  // Config case: order model-first, ladder [3.5-flash, 3.8-flash] (from the
  // seeded config). The ladder advances before the key.
  let r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key1/gemini-3.5-flash", "config first pick");
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: r.state, failed: {} });
  assertEq(r.model.provider + "/" + r.model.id, "google-key1/gemini-3.8-flash", "model-first advances the model first");
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: r.state, failed: {} });
  assertEq(r.model.provider + "/" + r.model.id, "google-key2/gemini-3.5-flash", "model-first wraps to the next key");
  console.log("OK — model-first config order and custom ladder honored");
} else if (process.env.GEMINI_ROUTER_SMOKE === "quota") {
  // Quota awareness: marks are recorded from failed requests, dead pairs are
  // skipped without burning requests, success clears marks, and a fully
  // marked router falls back to the held pair instead of throwing.
  const quotaFile = join(homedir(), ".pi", "agent", "gemini-router-quota.json");
  const readMarks = (): Record<string, { kind: string; until: number }> => {
    try {
      return JSON.parse(readFileSync(quotaFile, "utf-8"));
    } catch {
      return {};
    }
  };

  const perDayErr = '429 RESOURCE_EXHAUSTED quotaId "GenerateRequestsPerDayPerProjectPerModel" requests per day limit exceeded';
  const minuteErr = '429 RESOURCE_EXHAUSTED Generate content input tokens per minute limit exceeded. Please retry in 33.81s.';
  const genericErr = '429 RESOURCE_EXHAUSTED You exceeded your current quota, please check your plan and billing details.';

  // 1. RPD hit via route()'s failed path → the pair is skipped on the next route.
  let r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key1/gemini-4-hydrogen", "quota case first pick");
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: r.state, failed: { provider: "google-key1", model: "gemini-4-hydrogen", errorMessage: perDayErr } });
  assertEq(r.model.provider + "/" + r.model.id, "google-key2/gemini-4-hydrogen", "failed retry advances to next key");
  r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key2/gemini-4-hydrogen", "RPD-dead key1 skipped, still on hydrogen");
  const marks1 = readMarks();
  assertEq(marks1["gemini-4-hydrogen|google-key1"]?.kind, "rpd", "RPD mark persisted");

  // 2. Per-minute hit (with Google's retry hint) → pair cooling, skipped.
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: r.state, failed: { provider: "google-key2", model: "gemini-4-hydrogen", errorMessage: minuteErr } });
  assertEq(r.model.provider + "/" + r.model.id, "google-key3/gemini-4-hydrogen", "cooling advance");
  r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key3/gemini-4-hydrogen", "cooling key2 skipped");
  const marks2 = readMarks();
  assertEq(marks2["gemini-4-hydrogen|google-key2"]?.kind, "minute", "minute mark persisted");
  const cooldown = marks2["gemini-4-hydrogen|google-key2"].until - Date.now();
  assert(cooldown > 0 && cooldown <= 120000, "minute cooldown honors retry hint cap");

  // 3. Ambiguous quota 429 → possibly RPD, probe TTL.
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: r.state, failed: { provider: "google-key3", model: "gemini-4-hydrogen", errorMessage: genericErr } });
  r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key4/gemini-4-hydrogen", "suspected pair skipped");
  const marks3 = readMarks();
  assertEq(marks3["gemini-4-hydrogen|google-key3"]?.kind, "suspected", "suspected mark persisted");
  assert(marks3["gemini-4-hydrogen|google-key3"].until - Date.now() <= 31 * 60000, "suspected TTL bounded");

  // 4. Success clears marks: drive the captured turn_end handler with a
  // completed event for the held pair (hydrogen|key1 would be m0,k0 — mark key4's
  // pair via its state instead: the current route is key4/hydrogen).
  const stateEntry = { type: "custom", customType: "pi.virtual-model-state", data: { provider: "google", modelId: "gemini-flash", state: { m: 0, k: 3, epoch: 0 } } };
  const handlerCtx = {
    model: { provider: "google", id: "gemini-flash" },
    modelRegistry: { find: (p: string, id: string) => ALL.get(`${p}/${id}`), getAll: () => Array.from(ALL.values()) },
    sessionManager: { getEntries: () => [stateEntry] },
    ui: { notify: () => {} },
  };
  const turnEnd = handlers.get("turn_end")!;
  await turnEnd({ outcome: "completed" } as never, handlerCtx as never);
  assert(!readMarks()["gemini-4-hydrogen|google-key4"], "success clears the pair's mark");

  // 5. Settle-path marking: turn_end error → agent_before_settle records the
  // failed pair (stub state m0,k3 = hydrogen|key4) and continues the walk.
  await turnEnd({ outcome: "error", message: { role: "assistant", errorMessage: perDayErr }, messageEntryId: "e1" } as never, handlerCtx as never);
  const settle = handlers.get("agent_before_settle")!;
  const result = (await settle({ outcome: "error", context: { contextEntries: [], canContinue: true } } as never, handlerCtx as never)) as { continue?: boolean } | undefined;
  assertEq(result?.continue, true, "settle continues on quota error");
  const marks4 = readMarks();
  assertEq(marks4["gemini-4-hydrogen|google-key4"]?.kind, "rpd", "settle re-marks the failed pair");

  // 6. Fully marked router falls back instead of throwing: mark every pair of
  // the flash ladder by walking retries that fail the CURRENT pair each time,
  // then route must still return a pair (the held one) rather than throw.
  let cur = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  for (let i = 0; i < 7 * 7 + 2; i++) {
    cur = route(flash, {
      reason: "retry",
      thinkingLevel: "off",
      state: cur.state,
      failed: { provider: cur.model.provider, model: cur.model.id, errorMessage: perDayErr },
    });
  }
  const afterFlood = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assert(!!afterFlood.model, "fully-marked router still routes");
  console.log("OK — quota marking, skipping, self-healing, settle marking, and fallback all pass");
} else {
  // 1. First request: hydrogen (ladder head) on key1.
  let r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(r.model.provider + "/" + r.model.id, "google-key1/gemini-4-hydrogen", "first pick");
  const s0 = r.state;

  // 2. Retry after failure: same model, next key.
  r = route(flash, { reason: "retry", thinkingLevel: "off", state: s0, failed: { model: r.model } });
  assertEq(r.model.provider, "google-key2", "key advances on retry, model holds");
  assertEq(r.model.id, "gemini-4-hydrogen", "model holds across keys");

  // 3. User message with same state: no movement.
  r = route(flash, { reason: "user", thinkingLevel: "off", state: r.state });
  assertEq(r.model.provider, "google-key2", "no advance on success/user");

  // 4. Continuation: no movement.
  r = route(flash, { reason: "continuation", thinkingLevel: "off", state: r.state });
  assertEq(r.model.provider, "google-key2", "no advance on continuation");

  // 5. Exhaust all 7 keys on gemini-4-hydrogen; the next failure moves the model.
  let state = r.state;
  for (let i = 0; i < 4; i++) {
    state = route(flash, { reason: "retry", thinkingLevel: "off", state, failed: {} }).state;
  }
  const lastKeyOnHead = route(flash, { reason: "retry", thinkingLevel: "off", state, failed: {} });
  assertEq(lastKeyOnHead.model.provider, "google-key7", "last key on the model");
  const nextModel = route(flash, { reason: "retry", thinkingLevel: "off", state: lastKeyOnHead.state, failed: {} });
  assertEq(nextModel.model.provider, "google-key1", "model change restarts on key1");
  assertEq(nextModel.model.id, "gemini-flash-latest", "model advances only after all keys failed");

  // 6. Flash-lite router: same model across all keys, then next lite model.
  let lr = route(lite, { reason: "user", thinkingLevel: "off", state: undefined });
  assertEq(lr.model.id, "gemini-flash-lite-latest", "lite first pick");
  let lstate = lr.state;
  for (let i = 0; i < 6; i++) {
    lstate = route(lite, { reason: "retry", thinkingLevel: "off", state: lstate, failed: {} }).state;
  }
  assertEq(lstate.k, 6, "all 7 keys tried on flash-lite-latest");
  assertEq(lstate.m, 0, "lite model holds across keys");
  lr = route(lite, { reason: "retry", thinkingLevel: "off", state: lstate, failed: {} });
  assertEq(lr.model.provider, "google-key1", "lite model change restarts on key1");
  assertEq(lr.model.id, "gemini-3.5-flash-lite", "lite advances to next model after all keys");

  // 7. Missing models are skipped: key3 lacks flash-lite entries entirely.
  ALL.delete("google-key3/gemini-flash-lite-latest");
  ALL.delete("google-key3/gemini-3.5-flash-lite");
  ALL.delete("google-key3/gemini-3.1-flash-lite");
  const withHole = route(lite, { reason: "retry", thinkingLevel: "off", state: { m: 0, k: 2, epoch: 0 }, failed: {} });
  assertEq(withHole.model.provider + "/" + withHole.model.id, "google-key4/gemini-flash-lite-latest", "skips key without the model");

  // 8. Rotatable errors: quota and 4xx pair rejections rotate; context-overflow does not.
  const overflow = "Request contains an input that exceeds the maximum context length";
  assert(isRotatableError("429 RESOURCE_EXHAUSTED quota"), "quota rotates");
  assert(isRotatableError("400 Function call is missing a thought_signature in functionCall parts"), "signature 400 rotates");
  assert(isRotatableError("models/gemini-4-hydrogen is not found for API version v1beta"), "404 rotates");
  assert(isRotatableError("API key not valid"), "bad key rotates");
  assert(!isRotatableError(overflow), "context overflow does not rotate");
  assert(!isRotatableError(""), "empty does not rotate");

  console.log("OK — hydrogen-first ladder, key rotation, hold-on-success, model-skip, and rotatable-error classes all pass");
}

function assert(cond: unknown, what: string): void {
  if (!cond) throw new Error(what + ": failed");
}

function assertEq(actual: unknown, expected: unknown, what: string) {
  if (actual !== expected) {
    throw new Error(`${what}: expected ${expected}, got ${actual}`);
  }
}
