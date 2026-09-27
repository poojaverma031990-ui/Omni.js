/**
 * mini.js — Text generation engine: greedy / temperature / top-k / top-p sampling,
 * repetition penalty, EOS stopping, stop-callbacks, KV-cache stepping.
 * `generateTokens` is fully synchronous (Node-friendly); `generateAsync`
 * yields to the event loop between tokens (keeps browser UIs responsive).
 */
import { arenaRun } from './tensor.js';

function softmaxTopNothing(logits) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); sum += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

function applyRepetitionPenalty(logits, generatedIds, penalty) {
  const seen = new Set(generatedIds);
  for (const id of seen) {
    const v = logits[id];
    logits[id] = v > 0 ? v / penalty : v * penalty;
  }
}

function sampleTopKTopP(probs, { topK = 0, topP = 0 } = {}) {
  const V = probs.length;
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  let cand = idx;
  if (topK && topK > 0 && topK < V) cand = idx.slice(0, topK);
  if (topP && topP < 1) {
    let acc = 0, cut = cand.length;
    for (let i = 0; i < cand.length; i++) {
      acc += probs[cand[i]];
      if (acc >= topP) { cut = i + 1; break; }
    }
    cand = cand.slice(0, Math.max(1, cut));
  }
  let total = 0;
  for (const i of cand) total += probs[i];
  let r = Math.random() * total;
  for (const i of cand) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return cand[cand.length - 1];
}

function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

function normalizeOptions(options) {
  return {
    maxNewTokens: options.maxNewTokens ?? options.max_new_tokens ?? 50,
    doSample: options.doSample ?? options.do_sample ?? true,
    temperature: options.temperature ?? 1.0,
    topK: options.topK ?? options.top_k ?? 50,
    topP: options.topP ?? options.top_p ?? 0.95,
    repetitionPenalty: options.repetitionPenalty ?? options.repetition_penalty ?? 1.0,
    eosTokenId: options.eosTokenId ?? options.eos_token_id ?? null,
    onToken: options.onToken || null,
  };
}

/** Pick the next token from last-token logits. Returns [tokenId, keep]. */
function pickToken(lastLogits, generated, inputIds, opts) {
  const work = Float32Array.from(lastLogits);
  if (opts.repetitionPenalty && opts.repetitionPenalty !== 1.0 && generated.length) {
    applyRepetitionPenalty(work, inputIds.concat(generated), opts.repetitionPenalty);
  }
  if (!opts.doSample || opts.temperature <= 0.0001) return [argmax(work), true];
  if (opts.temperature !== 1.0) {
    for (let i = 0; i < work.length; i++) work[i] /= opts.temperature;
  }
  return [sampleTopKTopP(softmaxTopNothing(work), { topK: opts.topK, topP: opts.topP }), true];
}

function lastTokenLogits(model, inputIds, cache) {
  const logits = arenaRun(() => model.forward(inputIds, { cache, lastOnly: true }));
  let last = logits.read();
  if (logits.rows !== 1) last = last.subarray((logits.rows - 1) * logits.cols);
  else if (last.length !== logits.size) last = last.subarray(0, logits.cols);
  logits.dispose();
  return last;
}

/**
 * Synchronous generation. options: maxNewTokens, doSample, temperature, topK,
 * topP, repetitionPenalty, eosTokenId, onToken(id, n) — returning `false` from
 * onToken stops generation. Returns array of generated token ids.
 */
export function generateTokens(model, inputIds, options = {}) {
  const opts = normalizeOptions(options);
  const cache = model.makeCache();
  try {
    const generated = [];
    let lastLogits = lastTokenLogits(model, inputIds, cache);
    for (let step = 0; step < opts.maxNewTokens; step++) {
      const [nextId] = pickToken(lastLogits, generated, inputIds, opts);
      generated.push(nextId);
      if (opts.onToken && opts.onToken(nextId, step + 1) === false) break;
      if (opts.eosTokenId !== null && nextId === opts.eosTokenId) break;
      lastLogits = lastTokenLogits(model, [nextId], cache);
    }
    return generated;
  } finally {
    model.disposeCache(cache);
  }
}

/**
 * Async generation — identical API, but yields to the event loop between
 * tokens (via `tick`, default setTimeout 0) so browser UIs stay responsive
 * and onToken can update the DOM mid-generation.
 */
export async function generateAsync(model, inputIds, options = {}) {
  const opts = normalizeOptions(options);
  const tick = options.tick || (typeof window !== 'undefined'
    ? () => new Promise(r => setTimeout(r, 0))
    : null);
  const cache = model.makeCache();
  try {
    const generated = [];
    let lastLogits = lastTokenLogits(model, inputIds, cache);
    for (let step = 0; step < opts.maxNewTokens; step++) {
      const [nextId] = pickToken(lastLogits, generated, inputIds, opts);
      generated.push(nextId);
      if (opts.onToken && opts.onToken(nextId, step + 1) === false) break;
      if (opts.eosTokenId !== null && nextId === opts.eosTokenId) break;
      if (tick) await tick();
      lastLogits = lastTokenLogits(model, [nextId], cache);
    }
    return generated;
  } finally {
    model.disposeCache(cache);
  }
}
