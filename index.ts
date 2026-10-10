/**
 * pi-gemini-router — fallback router models for Google AI Studio.
 *
 * Registers two virtual models on the `google` provider:
 *
 *   google/gemini-flash       ladder: gemini-flash-latest -> 3.8 -> 3.7 -> 3.6
 *                                  -> 3.5 -> 3-flash-preview -> 2.5-flash
 *   google/gemini-flash-lite  ladder: gemini-flash-lite-latest -> 3.5-flash-lite
 *                                  -> 3.1-flash-lite
 *
 * Each ladder entry is tried on every multi-account key before the next
 * model (`google-key1..N` from pi-multi-account): the same model across all
 * keys multiplies that model's quota, and the model only changes once every
 * key has failed for it. A (model, key) pair that succeeds is held until it
 * errors. Nothing advances on success.
 *
 * Why: AI Studio free-tier quotas are per model AND per project. Gemini Flash
 * models get 20 RPD each and Flash Lite 500 RPD each, so a model ladder plus
 * key rotation multiplies daily capacity without touching per-request limits.
 * `-latest` aliases share the concrete model's quota, which is why they sit at
 * the head of the ladder (tried first) rather than counting as extra buckets.
 *
 * `/gemini-router` shows the ladders and discovered keys; `/gemini-router reset`
 * bumps an epoch so the next request starts at gemini-flash-latest on key1
 * (useful after the daily quota resets).
 */

import type { ExtensionAPI, ExtensionContext, ExtensionVirtualModel } from "@earendil-works/pi-coding-agent";

const PROVIDER = "google";
const KEY_PROVIDER_PATTERN = /^google-key(\d+)$/;

/** Ladder order: newest first. `-latest` aliases head the ladder. */
const FLASH_LADDER = [
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3-flash-preview",
  "gemini-2.5-flash",
];

const FLASH_LITE_LADDER = [
  "gemini-flash-lite-latest",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
];

const FLASH_CONTEXT_WINDOW = 1048576;
const FLASH_MAX_TOKENS = 65536;

interface RouterState {
  /** Index into the ladder. */
  m: number;
  /** Index into the discovered google-keyN providers. */
  k: number;
  /** Bumped by /gemini-router reset; a mismatch restarts the ladder. */
  epoch: number;
}

/** Module-level epoch; `/gemini-router reset` advances it. */
let epoch = 0;

type RouteFn = ExtensionVirtualModel<RouterState>["route"];

function keyProviders(ctx: ExtensionContext): string[] {
  const ids = new Set<string>();
  for (const m of ctx.modelRegistry.getAll()) {
    if (KEY_PROVIDER_PATTERN.test(m.provider)) ids.add(m.provider);
  }
  return Array.from(ids).sort((a, b) => {
    const na = Number(a.match(KEY_PROVIDER_PATTERN)![1]);
    const nb = Number(b.match(KEY_PROVIDER_PATTERN)![1]);
    return na - nb;
  });
}

function ladderLabel(ladder: string[]): string {
  return ladder.join(" → ");
}

/**
 * Advance order: key index fastest, model index slowest — exhaust all keys on
 * one model before moving down the ladder. Missing (model, key) pairs (a
 * provider without that catalog entry) are skipped.
 */
function makeRouter(ladder: string[]): RouteFn {
  return function route(request, ctx) {
    let state: RouterState =
      request.state && request.state.epoch === epoch
        ? request.state
        : { m: 0, k: 0, epoch };
    const keys = keyProviders(ctx);
    const keyCount = Math.max(1, keys.length);

    // Only a failed request moves the pair; success, user messages and
    // continuations keep the current (model, key).
    if (request.reason === "retry" && request.failed) {
      const total = Math.max(1, ladder.length * keyCount);
      const from = state.m * keyCount + state.k;
      const next = (from + 1) % total;
      state = { m: Math.floor(next / keyCount) % ladder.length, k: next % keyCount, epoch };
    }

    if (keys.length === 0) {
      throw new Error(
        "gemini-router: no google-keyN providers found — is pi-multi-account configured with google accounts?"
      );
    }
    const total = ladder.length * keys.length;
    const from = state.m * keys.length + state.k;
    for (let step = 0; step < total; step++) {
      const idx = (from + step) % total;
      const m = Math.floor(idx / keys.length) % ladder.length;
      const k = idx % keys.length;
      const model = ctx.modelRegistry.find(keys[k], ladder[m]);
      if (model) {
        return { model, thinkingLevel: request.thinkingLevel, state: { m, k, epoch } };
      }
    }
    throw new Error(
      "gemini-router: ladder exhausted — none of [" + ladderLabel(ladder) + "] found on " + keys.join(", ")
    );
  };
}

function virtualModel(
  id: string,
  name: string,
  ladder: string[]
): ExtensionVirtualModel<RouterState> {
  const def: ExtensionVirtualModel<RouterState> = {
    provider: PROVIDER,
    id,
    name,
    thinkingLevels: ["off", "low", "medium", "high"],
    contextWindow: FLASH_CONTEXT_WINDOW,
    maxTokens: FLASH_MAX_TOKENS,
    input: ["text", "image"],
    route: makeRouter(ladder),
  };
  return def;
}

export default function (pi: ExtensionAPI): void {
  pi.registerVirtualModel(virtualModel("gemini-flash", "Gemini Flash (router)", FLASH_LADDER));
  pi.registerVirtualModel(virtualModel("gemini-flash-lite", "Gemini Flash Lite (router)", FLASH_LITE_LADDER));

  pi.registerCommand("gemini-router", {
    description: "Show the Gemini router ladders and discovered keys, or reset the ladder position",
    handler: async (args: string, ctx: ExtensionContext) => {
      const cmd = (args || "").trim().toLowerCase();
      if (cmd === "reset") {
        epoch += 1;
        ctx.ui.notify(
          "gemini-router: reset — next request starts at gemini-flash-latest on key1",
          "info"
        );
        return;
      }
      const keys = keyProviders(ctx);
      const lines = [
        "gemini-router ladders:",
        "  gemini-flash       " + ladderLabel(FLASH_LADDER),
        "  gemini-flash-lite  " + ladderLabel(FLASH_LITE_LADDER),
        "keys discovered: " + (keys.length ? keys.join(", ") : "none — configure google accounts in multi-account.json"),
        "advance: key first (all keys on one model), then next model; holds (model, key) until it errors",
        "/gemini-router reset — restart at latest flash on key1",
      ];
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}
