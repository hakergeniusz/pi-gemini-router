/**
 * pi-gemini-router — fallback router models for Google AI Studio.
 *
 * Registers two virtual models on the `google` provider:
 *
 *   google/gemini-flash       ladder: gemini-4-hydrogen -> gemini-flash-latest
 *                                  -> 3.8 -> 3.7 -> 3.6 -> 3.5 -> 3-flash-preview
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
 * `-latest` aliases share the concrete model's quota, which is why they sit
 * near the head of the ladder (tried early) rather than counting as extra
 * buckets.
 *
 * `/gemini-router` shows the ladders and discovered keys; `/gemini-router reset`
 * bumps an epoch so the next request starts at the ladder head on key1
 * (useful after the daily quota resets).
 *
 * Both ladders and the advance order (key-first vs model-first) are
 * configurable in `~/.pi/agent/gemini-router.json`.
 *
 * The provider patch below also keeps the Google adapter alive across two
 * abort-signal hazards (see restoreAbortSignal and sanitizeSignal) and
 * neutralizes a third: pi-ai only replays `thoughtSignature`s when the
 * history comes from the exact same provider+model, so every model switch
 * leaves `functionCall` parts unsigned and strict Gemini 3/4 models answer
 * 400 "Function call is missing a thought_signature". Google special-cases
 * the sentinel value `skip_thought_signature_validator` for exactly this
 * (verified: it bypasses the base64 TYPE_BYTES check that rejects any other
 * non-base64 value, and is accepted by 3.5-flash/4-hydrogen), so unsigned
 * function calls get it injected before the request is sent
 * (see injectThoughtSignatures).
 *
 * Startup cost is kept near zero: only the built-in `google` provider is
 * patched eagerly; `google-keyN` providers are patched once they appear in
 * the registry (session_start, then model_select for keys registered later
 * by pi-multi-account). Re-registration merges configs, so a patch survives
 * pi-multi-account re-registering the same provider. The pi-ai compat module
 * (and with it the Google adapter) is imported lazily — long before the first
 * stream, with a synchronous require as a fallback.
 */

import type { ExtensionAPI, ExtensionContext, ExtensionVirtualModel } from "@earendil-works/pi-coding-agent";
import type { Api, AssistantMessageEventStream, Model, SimpleStreamOptions, TranscriptContext } from "@earendil-works/pi-ai";
import { sanitizeSignal } from "./signal.ts";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROVIDER = "google";
const KEY_PROVIDER_PATTERN = /^google-key(\d+)$/;

/** Default ladder order: newest first. `-latest` aliases sit right behind the newest concrete model. */
const DEFAULT_FLASH_LADDER = [
  "gemini-4-hydrogen",
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3-flash-preview",
];

const DEFAULT_FLASH_LITE_LADDER = [
  "gemini-flash-lite-latest",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
];

const FLASH_CONTEXT_WINDOW = 1048576;
const FLASH_MAX_TOKENS = 65536;

/** User overrides for the ladders and the advance order. */
interface RouterConfig {
  ladders?: Partial<Record<"gemini-flash" | "gemini-flash-lite", string[]>>;
  /** `key-first` (default): exhaust all keys on one model, then the next model. `model-first`: walk the ladder per key. */
  order?: "key-first" | "model-first";
}

const CONFIG_PATH = join(homedir(), ".pi", "agent", "gemini-router.json");

let routerConfig: RouterConfig = {};

function loadConfig(): void {
  try {
    routerConfig = JSON.parse(readFileSync(CONFIG_PATH, "utf-8")) as RouterConfig;
  } catch {
    routerConfig = {};
  }
}

function flashLadder(): string[] {
  const fromConfig = routerConfig.ladders?.["gemini-flash"];
  return fromConfig && fromConfig.length > 0 ? fromConfig : DEFAULT_FLASH_LADDER;
}

function flashLiteLadder(): string[] {
  const fromConfig = routerConfig.ladders?.["gemini-flash-lite"];
  return fromConfig && fromConfig.length > 0 ? fromConfig : DEFAULT_FLASH_LITE_LADDER;
}

function advanceOrder(): "key-first" | "model-first" {
  return routerConfig.order === "model-first" ? "model-first" : "key-first";
}

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
 *
 * The compat module itself (~100ms of module graph) is loaded lazily: an
 * async import kicks off at startup and lands long before the first stream;
 * a synchronous require covers a stream that arrives before it resolves.
 */
type CompatModule = typeof import("@earendil-works/pi-ai/compat");
type GoogleStreams = ReturnType<CompatModule["googleGenerativeAIApi"]>;

let compatModule: CompatModule | undefined;
let compatPreloadStarted = false;

const nodeRequire = createRequire(import.meta.url);

function ensureCompat(): CompatModule {
	if (!compatModule) {
		compatModule = nodeRequire("@earendil-works/pi-ai/compat") as CompatModule;
	}
	return compatModule;
}

function getStreams(): GoogleStreams {
	return ensureCompat().googleGenerativeAIApi();
}

/** Kick off the async compat import; registers the compat api entry on arrival. */
function preloadCompat(): void {
	if (compatModule || compatPreloadStarted) return;
	compatPreloadStarted = true;
	void import("@earendil-works/pi-ai/compat")
		.then((compat) => {
			compatModule = compat;
			registerCompatApiEntry();
			if (process.env.PI_GR_DEBUG === "1") console.error("[gemini-router] compat module loaded lazily");
		})
		.catch(() => {
			// Async import unavailable in this runtime — ensureCompat() covers the
			// stream path with a synchronous require.
		});
}

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
 * Google's escape hatch for replayed tool calls: a `thoughtSignature` with
 * this exact value skips signature validation. Needed because pi-ai drops
 * signatures on provider/model mismatch (resolveThoughtSignature), so a
 * router model switch leaves every prior `functionCall` part unsigned and
 * strict Gemini 3/4 models reject the request with
 * `Function call is missing a thought_signature` (400). Verified live: any
 * other non-base64 value dies in the TYPE_BYTES base64 check before this
 * special case, and the sentinel is accepted by 3.5-flash and 4-hydrogen.
 */
const THOUGHT_SIGNATURE_SENTINEL = "skip_thought_signature_validator";

/** Gemini major version, or undefined for non-gemini ids (gemma, claude-behind-google, …). */
function geminiMajorVersion(modelId: string): number | undefined {
  const match = modelId.toLowerCase().match(/^gemini(?:-live)?-(\d+)/);
  return match ? Number.parseInt(match[1], 10) : undefined;
}

/**
 * Fill every unsigned `functionCall` part in the outgoing payload with the
 * sentinel. Only strict Gemini 3+ models need it; other models (gemma, …)
 * keep their payload untouched. Mutates in place and is idempotent.
 */
function injectThoughtSignatures(payload: unknown, model: unknown): void {
  const id = (model as { id?: unknown } | undefined)?.id;
  if (typeof id !== "string") return;
  const major = geminiMajorVersion(id);
  if (major === undefined || major < 3) return;
  const contents = (payload as { contents?: unknown } | undefined)?.contents;
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    const parts = (content as { parts?: unknown } | undefined)?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (
        part &&
        typeof part === "object" &&
        "functionCall" in part &&
        !(part as { thoughtSignature?: unknown }).thoughtSignature
      ) {
        (part as { thoughtSignature?: string }).thoughtSignature = THOUGHT_SIGNATURE_SENTINEL;
      }
    }
  }
}

/**
 * Call the Google adapter with a survivable options object:
 *
 * 1. `sanitizeSignal` drops duck-typed turn signals (truthy, `.aborted`, no
 *    `addEventListener`) before they reach the genai client.
 * 2. `onPayload` is always installed: it injects sentinel thought signatures
 *    (injectThoughtSignatures), then chains the incoming handler so a payload
 *    rebuilt by `before_provider_request` handlers cannot ship a degraded
 *    `{}` abort signal (restoreAbortSignal) or lose the signatures.
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
  const callOptions: SimpleStreamOptions = {
    ...clean,
    onPayload: async (payload: unknown, m) => {
      injectThoughtSignatures(payload, m);
      const replaced = origOnPayload ? await origOnPayload(payload, m as never) : undefined;
      const final = restoreAbortSignal((replaced ?? payload) as never, signal);
      injectThoughtSignatures(final, m);
      return final;
    },
  } as SimpleStreamOptions;
  const versioned = versionedModel(model);
  const streams = getStreams();
  return method === "stream"
    ? streams.stream(versioned as Model<typeof GOOGLE_API>, context, callOptions as never)
    : streams.streamSimple(versioned as Model<typeof GOOGLE_API>, context, callOptions);
}

function googleStreamSimpleFixed(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  return callGoogleAdapter("streamSimple", model, context, options);
}

/**
 * Register the patched adapter in pi-ai's api-provider registry, so calls that
 * reach Google through compat dispatch get the same fixes as composed-provider
 * calls. Runs once the lazy compat import resolves (registerApiProvider comes
 * from that module).
 */
function registerCompatApiEntry(): void {
  try {
    ensureCompat().registerApiProvider(
      {
        api: GOOGLE_API,
        stream: (model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) =>
          callGoogleAdapter("stream", model, context, options),
        streamSimple: (model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) =>
          callGoogleAdapter("streamSimple", model, context, options),
      } as never,
      "pi-gemini-router"
    );
  } catch (e) {
    console.error("[gemini-router] failed to register compat api entry", e);
  }
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

/**
 * Pair-specific rejections: a 400/404 usually means the (model, key) pair
 * cannot serve this conversation — model unavailable for the project, strict
 * Gemini 3/4 signature enforcement (`Function call is missing a
 * thought_signature`), or a retired model id. Without rotating, the walk
 * would stick on that pair forever (pi never retries a 400). Context-overflow
 * 400s are the exception: no pair can serve them (they are the request's
 * size, not the pair's), so they are left to pi's compaction instead of
 * burning the whole ladder.
 */
const PAIR_SPECIFIC_ERROR_PATTERN =
  /\b400\b|INVALID_ARGUMENT|NOT_FOUND|FAILED_PRECONDITION|thought_signature|not found|not supported/i;
const NON_ROTATABLE_ERROR_PATTERN =
  /exceeds the maximum|token limit|input token|context length|request too large|payload too large|too many tokens/i;

/** Whether the failure is worth trying another key (and then model) for. */
export function isRotatableError(errorMessage: string): boolean {
  if (!errorMessage) return false;
  // A bad key is a permanent per-key problem: rotating the key is exactly the
  // right response, so it is rotatable too.
  if (/API[_ ]KEY[_ ]INVALID|API key not valid|PERMISSION_DENIED|401|403/i.test(errorMessage)) {
    return true;
  }
  if (NON_ROTATABLE_ERROR_PATTERN.test(errorMessage)) {
    return false;
  }
  return ROTATABLE_ERROR_PATTERN.test(errorMessage) || PAIR_SPECIFIC_ERROR_PATTERN.test(errorMessage);
}

/** Advance the (model, key) pair by one step.
 *
 * `key-first`: key index fastest, model index slowest — exhaust all keys on
 * one model before moving down the ladder. `model-first`: walk the ladder on
 * one key, then move to the next key.
 */
function advanceState(
  state: RouterState,
  ladderLength: number,
  keyCount: number,
  order: "key-first" | "model-first"
): RouterState {
  const total = Math.max(1, ladderLength * keyCount);
  const from =
    order === "model-first"
      ? state.k * ladderLength + state.m
      : state.m * keyCount + state.k;
  const next = (from + 1) % total;
  return {
    m: order === "model-first" ? next % ladderLength : Math.floor(next / keyCount) % ladderLength,
    k: order === "model-first" ? Math.floor(next / ladderLength) % keyCount : next % keyCount,
    epoch,
    tried: (state.tried ?? 0) + 1,
  };
}

/** Missing (model, key) pairs (a provider without that catalog entry) are skipped. */
function makeRouter(ladderOf: () => string[]): RouteFn {
  return function route(request, ctx) {
    let state: RouterState =
      request.state && request.state.epoch === epoch
        ? request.state
        : { m: 0, k: 0, epoch };
    const keys = keyProviders(ctx);
    if (keys.length === 0) {
      throw new Error(
        "gemini-router: no google-keyN providers found — is pi-multi-account configured with google accounts?"
      );
    }
    const keyCount = Math.max(1, keys.length);
    const ladder = ladderOf();
    const order = advanceOrder();

    // Only a failed request moves the pair; success, user messages and
    // continuations keep the current (model, key).
    if (request.reason === "retry" && request.failed) {
      state = advanceState(state, ladder.length, keyCount, order);
    }

    const total = ladder.length * keys.length;
    const from =
      order === "model-first"
        ? state.k * ladder.length + state.m
        : state.m * keys.length + state.k;
    for (let step = 0; step < total; step++) {
      const idx = (from + step) % total;
      const m = order === "model-first" ? idx % ladder.length : Math.floor(idx / keys.length) % ladder.length;
      const k = order === "model-first" ? Math.floor(idx / ladder.length) % keys.length : idx % keys.length;
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
  ladderOf: () => string[]
): ExtensionVirtualModel<RouterState> {
  const def: ExtensionVirtualModel<RouterState> = {
    provider: PROVIDER,
    id,
    name,
    thinkingLevels: ["off", "low", "medium", "high"],
    contextWindow: FLASH_CONTEXT_WINDOW,
    maxTokens: FLASH_MAX_TOKENS,
    input: ["text", "image"],
    route: makeRouter(ladderOf),
  };
  return def;
}

/** Router state of the virtual model that is currently selected. */
interface SelectedRouter {
  modelId: string;
  ladder: string[];
}

/** The live ladder of the selected virtual model, if it is one of ours. */
function selectedRouter(ctx: ExtensionContext): SelectedRouter | undefined {
  const model = ctx.model;
  if (!model || model.provider !== PROVIDER) return undefined;
  if (model.id === "gemini-flash") return { modelId: model.id, ladder: flashLadder() };
  if (model.id === "gemini-flash-lite") return { modelId: model.id, ladder: flashLiteLadder() };
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

/** Google-ish providers already patched with the signal-safe streamSimple. */
const patchedProviders = new Set<string>();

/**
 * Patch one Google provider with the signal-safe streaming. No models are
 * listed, so existing model catalogs stay; the extension layer only overrides
 * streaming for the Google api. Re-registration merges configs, so this
 * survives pi-multi-account re-registering the same provider.
 */
function patchGoogleProvider(pi: ExtensionAPI, providerId: string): void {
  if (patchedProviders.has(providerId)) return;
  patchedProviders.add(providerId);
  try {
    pi.registerProvider(providerId, {
      api: GOOGLE_API,
      streamSimple: googleStreamSimpleFixed,
    });
    if (process.env.PI_GR_DEBUG === "1") console.error("[gemini-router] patched " + providerId);
  } catch (e) {
    patchedProviders.delete(providerId);
    console.error("[gemini-router] failed to patch provider " + providerId, e);
  }
}

export default function (pi: ExtensionAPI): void {
  loadConfig();

  // The compat module (and its ~100ms module graph) loads in the background;
  // it also registers the patched adapter in pi-ai's api-provider registry so
  // compat-dispatched calls get the same fixes. registerBuiltInApiProviders()
  // does not clobber existing entries.
  preloadCompat();

  // Only patch at startup: the built-in google provider. Each
  // registerProvider triggers pi's model snapshot rebuild, so google-keyN
  // providers are patched when they actually appear in the registry instead
  // of being created eagerly as empty providers.
  patchGoogleProvider(pi, "google");

  pi.registerVirtualModel(virtualModel("gemini-flash", "Gemini Flash (router)", flashLadder));
  pi.registerVirtualModel(virtualModel("gemini-flash-lite", "Gemini Flash Lite (router)", flashLiteLadder));

  // By now pi-multi-account has registered google-key1; patch every
  // Google-ish provider that has models in the registry, and pick up config
  // edits without a restart.
  pi.on("session_start", ((_event: unknown, ctx: ExtensionContext) => {
    loadConfig();
    const providers = new Set<string>([PROVIDER]);
    for (const m of ctx.modelRegistry.getAll()) {
      if (KEY_PROVIDER_PATTERN.test(m.provider) || m.provider === PROVIDER) providers.add(m.provider);
    }
    for (const providerId of providers) patchGoogleProvider(pi, providerId);
  }) as never);

  // Keys registered later (pi-multi-account account switches) get patched the
  // moment they are selected, before anything streams through them.
  pi.on("model_select", ((event: { model?: { provider?: string } }) => {
    const provider = event.model?.provider;
    if (!provider) return;
    if (provider === PROVIDER || KEY_PROVIDER_PATTERN.test(provider)) {
      patchGoogleProvider(pi, provider);
    }
  }) as never);

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
    const next = advanceState(current, router.ladder.length, keyCount, advanceOrder());

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
          "gemini-router: reset — next request starts at the ladder head on key1",
          "info"
        );
        return;
      }
      const keys = keyProviders(ctx);
      const flashFromConfig = routerConfig.ladders?.["gemini-flash"];
      const liteFromConfig = routerConfig.ladders?.["gemini-flash-lite"];
      const lines = [
        "gemini-router ladders" + (flashFromConfig || liteFromConfig ? " (config: " + CONFIG_PATH + ")" : " (defaults; override in " + CONFIG_PATH + "):"),
        "  gemini-flash       " + ladderLabel(flashLadder()) + (flashFromConfig ? "  [config]" : ""),
        "  gemini-flash-lite  " + ladderLabel(flashLiteLadder()) + (liteFromConfig ? "  [config]" : ""),
        "keys discovered: " + (keys.length ? keys.join(", ") : "none — configure google accounts in multi-account.json"),
        "advance: " + advanceOrder() + " (" + (advanceOrder() === "key-first" ? "all keys on one model, then next model" : "whole ladder per key, then next key") + "); holds (model, key) until it errors",
        "/gemini-router reset — restart at the ladder head on key1",
      ];
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}
