/**
 * mini.js — Model Hub: streams files from the Hugging Face Hub with progress,
 * caches them in the browser Cache API (repeat runs are instant), no deps.
 * Supports sharded safetensors models (model.safetensors.index.json).
 */
import { concatBytes, ProgressEmitter } from './utils.js';
import { parseSafetensors } from './safetensors.js';

const HF_BASE = 'https://huggingface.co';

export const env = {
  hfBase: HF_BASE,
  useBrowserCache: true,   // cache downloaded files (Cache API)
  cacheName: 'mini-js-models-v1',
};

function fileUrl(repo, filename, revision = 'main') {
  return `${env.hfBase}/${repo}/resolve/${revision}/${filename}`;
}

async function openCache() {
  if (!env.useBrowserCache) return null;
  try {
    if (typeof caches !== 'undefined' && caches.open) return await caches.open(env.cacheName);
  } catch (_) { /* private mode etc. */ }
  return null;
}

/**
 * Download (or fetch from cache) one file as Uint8Array.
 * progressCallback receives {status, file, loaded, total, progress}.
 */
export async function downloadFile(repo, filename, { progressCallback = null, revision = 'main' } = {}) {
  const emitter = new ProgressEmitter(progressCallback);
  const url = fileUrl(repo, filename, revision);
  emitter.status('initiate', filename, { url });

  const cache = await openCache();
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        emitter.status('cached', filename, { loaded: buf.length, total: buf.length });
        return buf;
      }
    } catch (_) { /* ignore cache errors */ }
  }

  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) {
    throw new Error(`[mini.js] failed to download ${repo}/${filename} — HTTP ${res.status}` +
      (res.status === 401 ? ' (repository is gated; skipping)' : ''));
  }

  let bytes;
  if (res.body && res.body.getReader) {
    const total = Number(res.headers.get('content-length') || 0);
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    let lastEmit = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      const now = Date.now();
      if (now - lastEmit > 100 || loaded === total) { // throttle to ~10 events/sec
        emitter.progress(filename, loaded, total || 0);
        lastEmit = now;
      }
    }
    emitter.progress(filename, loaded, loaded);
    bytes = concatBytes(chunks, loaded);
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    emitter.progress(filename, bytes.length, bytes.length);
  }

  if (cache) {
    try {
      await cache.put(url, new Response(bytes.slice(), {
        headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.length) },
      }));
    } catch (_) { /* quota exceeded — ignore */ }
  }
  emitter.status('done', filename, { loaded: bytes.length, total: bytes.length });
  return bytes;
}

async function fetchJSON(repo, filename) {
  const bytes = await downloadFile(repo, filename);
  return JSON.parse(new TextDecoder('utf-8').decode(bytes));
}

/**
 * Download everything a repo needs: config, tokenizer files, weights.
 * Supports single `model.safetensors` AND sharded models via
 * `model.safetensors.index.json` (weight_map → multiple shard files).
 * Returns {config, tokenizerConfig, tokenizerJson, tensors: Map, weightFiles}
 */
export async function downloadModel(repo, { progressCallback = null, weightFile = null } = {}) {
  const files = {};
  const grabJSON = async (name) => {
    try { files[name] = await fetchJSON(repo, name); } catch (_) { files[name] = null; }
  };
  await Promise.all([
    grabJSON('config.json'),
    grabJSON('tokenizer_config.json'),
    grabJSON('special_tokens_map.json'),
  ]);

  const dlOpts = { progressCallback };

  // ── sharded safetensors? ──
  let index = null;
  try { index = await fetchJSON(repo, 'model.safetensors.index.json'); } catch (_) { /* single-file */ }

  let tensors = null;
  const weightFiles = [];
  if (index && index.weight_map) {
    const shardNames = [...new Set(Object.values(index.weight_map))];
    for (const shard of shardNames) {
      const bytes = await downloadFile(repo, shard, dlOpts);
      const parsed = parseSafetensors(bytes);
      if (!tensors) tensors = parsed.tensors;
      else for (const [k, v] of parsed.tensors) tensors.set(k, v);
      weightFiles.push(shard);
    }
    // sanity: every mapped tensor must exist
    for (const name of Object.keys(index.weight_map)) {
      if (!tensors.has(name)) throw new Error(`[mini.js] shard index references missing tensor "${name}"`);
    }
  } else {
    const candidates = weightFile ? [weightFile] : ['model.safetensors'];
    let lastErr = null;
    for (const f of candidates) {
      try {
        const bytes = await downloadFile(repo, f, dlOpts);
        tensors = parseSafetensors(bytes).tensors;
        weightFiles.push(f);
        break;
      } catch (e) { lastErr = e; }
    }
    if (!tensors) {
      throw lastErr || new Error('[mini.js] no safetensors weights found in ' + repo);
    }
  }

  // tokenizer.json (may be missing on very old repos)
  try { files['tokenizer.json'] = await fetchJSON(repo, 'tokenizer.json'); }
  catch (_) { files['tokenizer.json'] = null; }

  return {
    config: files['config.json'],
    tokenizerConfig: files['tokenizer_config.json'] || files['special_tokens_map.json'] || {},
    tokenizerJson: files['tokenizer.json'],
    tensors,
    weightFiles,
  };
}

export async function clearCache() {
  const cache = await openCache();
  if (cache) await cache.keys().then(keys => Promise.all(keys.map(k => cache.delete(k))));
  return true;
}
