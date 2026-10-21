/**
 * pi-gemini-router — fallback router models for Google AI Studio.
 *
 * Registers two virtual models on the `google` provider:
 *
 *   google/gemini-flash       ladder: gemini-flash-latest -> 3.8 -> 3.7 -> 3.6
 *                                  -> 3.5 -> 3-flash-preview
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
 *
 * The provider patch below also keeps the Google adapter alive across two
 * abort-signal hazards (see restoreAbortSignal and sanitizeSignal).
 */

import type { ExtensionAPI, ExtensionContext, ExtensionVirtualModel } from "@earendil-works/pi-coding-agent";
import type { Api, AssistantMessageEventStream, Model, SimpleStreamOptions, TranscriptContext } from "@earendil-works/pi-ai";
import { googleGenerativeAIApi, registerApiProvider } from "@earendil-works/pi-ai/compat";
import { sanitizeSignal } from "./signal.ts";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

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
  /**
   * How many (model, key) pairs have failed in the current walk.
   *
   * A walk covers every pair exactly once; when it is complete the error
   * surfaces instead of looping over the same exhausted keys forever.
   */
  tried?: number;
}

/** Custom session entry type holding the router state for a virtual model. */
const ROUTER_STATE_ENTRY = "pi.virtual-model-state";

/** The failure the `agent_before_settle` handler rotates away from. */
interface RouterFailure {
  /** Session entry id of the failed assistant message. */
  entryId: string;
  errorMessage: string;
  /** Router state in effect when the request failed. */
  state: RouterState;
  /** How many (model, key) pairs have already failed in this walk. */
  tried: number;
}

/** Latest failure, stashed at `turn_end` and consumed at `agent_before_settle`. */
let lastFailure: RouterFailure | undefined;

/** Module-level epoch; `/gemini-router reset` advances it. */
let epoch = 0;

const GOOGLE_API = "google-generative-ai" as const;

/** The Google adapter, reached through the compat entrypoint.
 *
 * Extension imports resolve against a fixed module table (VIRTUAL_MODULES in
 * compiled pi builds, aliases otherwise) that maps only the pi-ai root and a
 * few subpaths — the eager `@earendil-works/pi-ai/api/google-generative-ai`
 * specifier is not among them, so importing it kills the whole extension at
 * load time. The compat entrypoint is mapped everywhere and lazily loads the
 * same adapter module on first stream call.
 */
const googleStreams = googleGenerativeAIApi();

/** True only for real AbortSignals; false for `{}` clones and duck types. */
function isRealAbortSignal(value: unknown): value is AbortSignal {
  return !!value && typeof (value as AbortSignal).addEventListener === "function";
}

/**
 * Give the SDK a versioned baseUrl.
 *
 * pi-ai's createClient sets `apiVersion: ""` whenever `model.baseUrl` is set
 * ("baseUrl already includes version path, don't append"), but the Google
 * catalog baseUrl is the bare host — so the SDK builds
 * `https://generativelanguage.googleapis.com/models/...` without the
 * `/v1beta` segment and Google answers 404 for every request. Append the
 * version to the baseUrl instead.
 */
function versionedModel(model: Model<Api>): Model<Api> {
  const base = typeof model.baseUrl === "string" ? model.baseUrl : "";
  if (!base.startsWith("https://generativelanguage.googleapis.com") || /\/v\d+[a-z]*$/.test(base)) return model;
  return { ...model, baseUrl: base.replace(/\/+$/, "") + "/v1beta" };
}

/**
 * Repair the abort signal after `before_provider_request` transforms.
 *
 * The adapter receives `options.onPayload` and replaces its params with the
 * returned payload. Payload handlers clone the params — pi-output-cap, for
 * one, runs them through `structuredClone` — and Node's structuredClone
 * degrades an AbortSignal to a plain `{}` (verified on node 26). The genai
 * client then dies with
 * `Error: callerSignal.addEventListener is not a function`
 * inside `createAttemptSignal`, before any bytes reach the wire.
 *
 * Put the original signal back when the transformed payload lost it; drop the
 * field when there was none. Non-mutating: the clone belongs to the handler
 * chain.
 */
function restoreAbortSignal<T>(payload: T, signal: unknown): T {
  const config = (payload as { config?: Record<string, unknown> } | undefined)?.config;
  if (!config || !("abortSignal" in config) || isRealAbortSignal(config.abortSignal)) return payload;
  const fixedConfig = { ...config, abortSignal: isRealAbortSignal(signal) ? signal : undefined };
  return { ...(payload as object), config: fixedConfig } as T;
}

type AdapterMethod = "stream" | "streamSimple";

/**
 * Call the Google adapter with a survivable options object:
 *
 * 1. `sanitizeSignal` drops duck-typed turn signals (truthy, `.aborted`, no
 *    `addEventListener`) before they reach the genai client.
 * 2. `onPayload` is wrapped so a payload rebuilt by `before_provider_request`
 *    handlers cannot ship a degraded `{}` abort signal to the SDK.
 */
function callGoogleAdapter(
  method: AdapterMethod,
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  const clean = sanitizeSignal(options);
  const signal = clean?.signal;
  const origOnPayload = clean?.onPayload;
  const callOptions: SimpleStreamOptions | undefined = origOnPayload
    ? ({
        ...clean,
        onPayload: async (payload: unknown, m) =>
          restoreAbortSignal((await origOnPayload(payload, m as never)) ?? payload, signal),
      } as SimpleStreamOptions)
    : clean;
  const versioned = versionedModel(model);
  return method === "stream"
    ? googleStreams.stream(versioned as Model<typeof GOOGLE_API>, context, callOptions as never)
    : googleStreams.streamSimple(versioned as Model<typeof GOOGLE_API>, context, callOptions);
}

function googleStreamSimpleFixed(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  return callGoogleAdapter("streamSimple", model, context, options);
}

/** Providers whose physical models stream through the Google adapter. */
function googleProviderIds(): string[] {
  const ids = ["google"];
  try {
    const path = join(homedir(), ".pi", "agent", "multi-account.json");
    if (existsSync(path)) {
      const cfg = JSON.parse(readFileSync(path, "utf-8")) as {
        providers?: Record<string, { api?: string; accounts?: unknown[] }>;
      };
      for (const [name, pc] of Object.entries(cfg.providers ?? {})) {
        if (pc.api === GOOGLE_API && Array.isArray(pc.accounts)) {
          for (let i = 0; i < pc.accounts.length; i++) ids.push(name + "-key" + (i + 1));
        }
      }
    }
  } catch {
    // Config unreadable: the plain google provider is still patched.
  }
  return ids;
}

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

/** Errors that a different key can still answer.
 *
 * Pi's own retry classifier treats quota exhaustion (429 RESOURCE_EXHAUSTED)
 * as a billing condition and never retries it, so a router that only moves on
 * `reason === "retry"` would sit on an exhausted key forever. AI Studio
 * quotas are per project, so the next key usually has quota left.
 */
const ROTATABLE_ERROR_PATTERN =
  /429|RESOURCE_EXHAUSTED|quota|rate.?limit|too many requests|high demand|UNAVAILABLE|503|500|502|504|overloaded/i;

/** Whether the failure is worth trying another key (and then model) for. */
function isRotatableError(errorMessage: string): boolean {
  if (!errorMessage) return false;
  // A bad key is a permanent per-key problem: rotating the key is exactly the
  // right response, so it is rotatable too.
  if (/API[_ ]KEY[_ ]INVALID|API key not valid|PERMISSION_DENIED|401|403/i.test(errorMessage)) {
    return true;
  }
  return ROTATABLE_ERROR_PATTERN.test(errorMessage);
}

/** Advance the (model, key) pair by one step: key first, then model. */
function advanceState(state: RouterState, ladderLength: number, keyCount: number): RouterState {
  const total = Math.max(1, ladderLength * keyCount);
  const from = state.m * keyCount + state.k;
  const next = (from + 1) % total;
  return {
    m: Math.floor(next / keyCount) % ladderLength,
    k: next % keyCount,
    epoch,
    tried: (state.tried ?? 0) + 1,
  };
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
      state = advanceState(state, ladder.length, keyCount);
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

/** Router state of the virtual model that is currently selected. */
interface SelectedRouter {
  modelId: string;
  ladder: string[];
}

/** The ladder of the selected virtual model, if it is one of ours. */
function selectedRouter(ctx: ExtensionContext): SelectedRouter | undefined {
  const model = ctx.model;
  if (!model || model.provider !== PROVIDER) return undefined;
  if (model.id === "gemini-flash") return { modelId: model.id, ladder: FLASH_LADDER };
  if (model.id === "gemini-flash-lite") return { modelId: model.id, ladder: FLASH_LITE_LADDER };
  return undefined;
}

/** The router state stored on the session branch for a virtual model. */
function routerStateOf(ctx: ExtensionContext, modelId: string): RouterState | undefined {
  const entries = ctx.sessionManager.getEntries() as Array<{
    type: string;
    customType?: string;
    data?: { provider?: string; modelId?: string; state?: RouterState };
  }>;
  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i];
    if (entry.type !== "custom" || entry.customType !== ROUTER_STATE_ENTRY) continue;
    const data = entry.data;
    if (!data || data.provider !== PROVIDER || data.modelId !== modelId) continue;
    return data.state;
  }
  return undefined;
}

/** Notify the user, tolerating contexts without UI. */
function notify(ctx: ExtensionContext, message: string): void {
  try {
    ctx.ui.notify(message, "info");
  } catch {
    // No UI (background/direct request): the rotation still happens.
  }
}

export default function (pi: ExtensionAPI): void {
  // Defense in depth: also cover calls that reach the adapter through the
  // api-provider registry (compat dispatch) instead of a composed provider.
  // registerBuiltInApiProviders() does not clobber existing entries.
  registerApiProvider(
    {
      api: GOOGLE_API,
      stream: (model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) =>
        callGoogleAdapter("stream", model, context, options),
      streamSimple: (model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) =>
        callGoogleAdapter("streamSimple", model, context, options),
    } as never,
    "pi-gemini-router"
  );

  // Patch every Google provider with the signal fix. No models are listed, so
  // existing model catalogs (built-in google, pi-multi-account's google-keyN)
  // stay; the extension layer only overrides streaming for the Google api.
  for (const providerId of googleProviderIds()) {
    try {
      pi.registerProvider(providerId, {
        api: GOOGLE_API,
        streamSimple: googleStreamSimpleFixed,
      });
    } catch (e) {
      console.error("[gemini-router] failed to patch provider " + providerId, e);
    }
  }

  pi.registerVirtualModel(virtualModel("gemini-flash", "Gemini Flash (router)", FLASH_LADDER));
  pi.registerVirtualModel(virtualModel("gemini-flash-lite", "Gemini Flash Lite (router)", FLASH_LITE_LADDER));

  // Stash the failure (and the router state in effect) so the settle handler
  // can advance to the next (key, model) pair.
  pi.on("turn_end", async (event, ctx) => {
    if (event.outcome === "error" && event.message.role === "assistant") {
      const router = selectedRouter(ctx);
      if (router) {
        const state = routerStateOf(ctx, router.modelId) ?? { m: 0, k: 0, epoch };
        lastFailure = {
          entryId: event.messageEntryId,
          errorMessage: (event.message as { errorMessage?: string }).errorMessage ?? "",
          state,
          tried: state.tried ?? 0,
        };
        return undefined;
      }
    }
    if (event.outcome === "completed") {
      lastFailure = undefined;
      // A good response ends the walk: the next prompt starts a fresh one.
      const router = selectedRouter(ctx);
      if (router) {
        const state = routerStateOf(ctx, router.modelId);
        if (state && state.tried) {
          pi.appendEntry(ROUTER_STATE_ENTRY, {
            provider: PROVIDER,
            modelId: router.modelId,
            state: { ...state, tried: 0 },
          });
        }
      }
    }
    return undefined;
  });

  // A new user prompt starts a fresh walk, so the tried counter is cleared.
  pi.on("before_agent_start", async (_event, ctx) => {
    lastFailure = undefined;
    const router = selectedRouter(ctx);
    if (!router) return undefined;
    const state = routerStateOf(ctx, router.modelId);
    if (state && state.tried) {
      pi.appendEntry(ROUTER_STATE_ENTRY, {
        provider: PROVIDER,
        modelId: router.modelId,
        state: { ...state, tried: 0 },
      });
    }
    return undefined;
  });

  // Pi's retry classifier treats quota exhaustion (429 RESOURCE_EXHAUSTED) as
  // a billing condition and never retries it, so the router would sit on the
  // exhausted key forever. Take over at `agent_before_settle`: advance the
  // router state to the next (key, model) pair and ask for one more request.
  pi.on("agent_before_settle", async (event, ctx) => {
    if (event.outcome !== "error") return undefined;
    const router = selectedRouter(ctx);
    if (!router) return undefined;
    const failure = lastFailure;
    lastFailure = undefined;
    if (!failure || !isRotatableError(failure.errorMessage)) return undefined;

    const keys = keyProviders(ctx);
    if (keys.length === 0) return undefined;
    const keyCount = Math.max(1, keys.length);

    // The walk covers every (model, key) pair exactly once. When it is
    // complete the error surfaces instead of looping over exhausted keys.
    const totalPairs = router.ladder.length * keyCount;
    if (failure.tried >= totalPairs) return undefined;

    const current = failure.state ?? { m: 0, k: 0, epoch };
    const next = advanceState(current, router.ladder.length, keyCount);

    // The failed attempt must leave the context, or the retry repeats it.
    const targetPresent = event.context.contextEntries.some(
      (entry) => entry.sourceEntry.id === failure.entryId
    );
    if (!targetPresent && !event.context.canContinue) return undefined;
    const entries = targetPresent
      ? [{ type: "context_edit" as const, targetId: failure.entryId, replacement: null }]
      : [];

    // Store the advanced state so the next routing picks the new pair.
    pi.appendEntry(ROUTER_STATE_ENTRY, {
      provider: PROVIDER,
      modelId: router.modelId,
      state: next,
    });

    notify(
      ctx,
      "gemini-router: " +
        (failure.errorMessage.slice(0, 60).replace(/\s+/g, " ")) +
        " — trying " +
        (keys[next.k] ?? "?") +
        " / " +
        (router.ladder[next.m] ?? "?")
    );
    return { entries, continue: true };
  });

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
