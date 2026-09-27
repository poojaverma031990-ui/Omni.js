/**
 * mini.js — Text generation engine: greedy / temperature / top-k / top-p sampling,
 * repetition penalty, EOS stopping, KV-cache stepping. From scratch.
 */
import { Tensor } from './tensor.js';

function softmaxTopNothing(logits) { // full softmax over Float32Array (small: vocab)
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
  let cand = idx;
  if (topK && topK > 0 && topK < V) {
    idx.sort((a, b) => probs[b] - probs[a]);
    cand = idx.slice(0, topK);
  } else {
    idx.sort((a, b) => probs[b] - probs[a]);
    cand = idx;
  }
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

/**
 * Generate tokens with a GPT2-style model.
 * options: maxNewTokens, doSample, temperature, topK, topP, repetitionPenalty,
 *          eosTokenId, onToken(id, n)
 * Returns array of generated token ids (excluding prompt).
 */
export function generateTokens(model, inputIds, options = {}, tokenizer = null) {
  const {
    maxNewTokens = 50,
    doSample = true,
    temperature = 1.0,
    topK = 50,
    topP = 0.95,
    repetitionPenalty = 1.0,
    eosTokenId = null,
    onToken = null,
  } = options;

  const cache = model.makeCache();
  try {
    const generated = [];
    let lastLogits;
    // prefill: single pass over the whole prompt
    {
      const logits = arenaRun(() => model.forward(inputIds, { cache, lastOnly: true }));
      lastLogits = logits.read();
      if (logits.rows !== 1) lastLogits = lastLogits.subarray((logits.rows - 1) * logits.cols);
      logits.dispose();
    }
    // avoid immediately emitting the BOS of some tokenizers
    for (let step = 0; step < maxNewTokens; step++) {
      const work = Float32Array.from(lastLogits);
      if (repetitionPenalty && repetitionPenalty !== 1.0 && generated.length) {
        applyRepetitionPenalty(work, inputIds.concat(generated), repetitionPenalty);
      }
      let nextId;
      if (!doSample || (temperature !== undefined && temperature <= 0.0001)) {
        nextId = argmax(work);
      } else {
        if (temperature !== 1.0) for (let i = 0; i < work.length; i++) work[i] /= temperature;
        const probs = softmaxTopNothing(work);
        nextId = sampleTopKTopP(probs, { topK, topP });
      }
      generated.push(nextId);
      if (onToken) onToken(nextId, step + 1);
      if (eosTokenId !== null && nextId === eosTokenId) break;
      // decode step: single token, uses cache
      const l2 = model.forward([nextId], { cache, lastOnly: true });
      lastLogits = l2.read();
      if (l2.disposed) break;
    }
    return generated;
  } finally {
    model.disposeCache(cache);
  }
}
