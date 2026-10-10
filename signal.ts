/**
 * Signal sanitization for the Google Generative AI adapter.
 *
 * pi-ai's Google adapter forwards `options.signal` into genai's
 * `config.abortSignal`, where `createAttemptSignal(timeout, callerSignal)`
 * calls `callerSignal.addEventListener("abort", ...)`. Interactive-mode turn
 * options carry a duck-typed signal — truthy, with `.aborted`, but no
 * `addEventListener` — so every Google model dies with
 * `callerSignal.addEventListener is not a function`. Headless runs pass a real
 * AbortSignal, which is why only the TUI hits this.
 *
 * Drop such signals (keeping already-aborted ones as a real aborted signal).
 * Real AbortSignals pass through untouched.
 */
export function sanitizeSignal<T extends { signal?: unknown }>(options: T | undefined): T | undefined {
  const signal = options?.signal;
  if (!signal) return options;
  if (typeof (signal as AbortSignal).addEventListener === "function") return options;
  return {
    ...(options as T),
    signal: (signal as { aborted?: boolean }).aborted ? AbortSignal.abort() : undefined,
  };
}
