// Smoke test for pi-gemini-router: stubs the registry and exercises the
// ladder/rotation logic without touching the network.
import activate from "./index.ts";

const virtualModels = new Map<string, { provider: string; id: string; route: (r: unknown, c: unknown) => unknown }>();
const commands: string[] = [];

const pi = {
  registerVirtualModel: (def: { provider: string; id: string }) => {
    virtualModels.set(def.provider + "/" + def.id, def as never);
  },
  registerCommand: (name: string) => commands.push(name),
} as never;

activate(pi);

if (virtualModels.size !== 2) throw new Error("expected 2 virtual models, got " + virtualModels.size);
if (commands.join(",") !== "gemini-router") throw new Error("command not registered");

// Stub registry: 7 google-keyN providers, each with the full flash ladders.
const FLASH = [
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3-flash-preview",
  "gemini-2.5-flash",
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

// 1. First request: latest flash on key1.
let r = route(flash, { reason: "user", thinkingLevel: "off", state: undefined });
assertEq(r.model.provider + "/" + r.model.id, "google-key1/gemini-flash-latest", "first pick");
const s0 = r.state;

// 2. Retry after failure: same model, next key.
r = route(flash, { reason: "retry", thinkingLevel: "off", state: s0, failed: { model: r.model } });
assertEq(r.model.provider, "google-key2", "key advances on retry, model holds");
assertEq(r.model.id, "gemini-flash-latest", "model holds across keys");

// 3. User message with same state: no movement.
r = route(flash, { reason: "user", thinkingLevel: "off", state: r.state });
assertEq(r.model.provider, "google-key2", "no advance on success/user");

// 4. Continuation: no movement.
r = route(flash, { reason: "continuation", thinkingLevel: "off", state: r.state });
assertEq(r.model.provider, "google-key2", "no advance on continuation");

// 5. Exhaust all 7 keys on gemini-flash-latest; the next failure moves the model.
let state = r.state;
for (let i = 0; i < 4; i++) {
  state = route(flash, { reason: "retry", thinkingLevel: "off", state, failed: {} }).state;
}
const lastKeyOnLatest = route(flash, { reason: "retry", thinkingLevel: "off", state, failed: {} });
assertEq(lastKeyOnLatest.model.provider, "google-key7", "last key on the model");
const nextModel = route(flash, { reason: "retry", thinkingLevel: "off", state: lastKeyOnLatest.state, failed: {} });
assertEq(nextModel.model.provider, "google-key1", "model change restarts on key1");
assertEq(nextModel.model.id, "gemini-3.8-flash", "model advances only after all keys failed");

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

console.log("OK — routing ladder, key rotation, hold-on-success, and model-skip all pass");

function assertEq(actual: unknown, expected: unknown, what: string) {
  if (actual !== expected) {
    throw new Error(`${what}: expected ${expected}, got ${actual}`);
  }
}
