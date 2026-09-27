/**
 * mini.js — High-level pipelines, Transformers.js-style but 100% from scratch:
 *   const gen = await mini.pipeline('text-generation', 'openai-community/gpt2');
 *   const out = await gen('Once upon a time');
 */
import { Tensor, arenaRun, currentBackend, detectBestBackend, setBackend } from './tensor.js';
import { parseSafetensors } from './safetensors.js';
import { createTokenizer, Tokenizer } from './tokenizer.js';
import { downloadModel, env } from './hub.js';
import { GPT2Model, BertModel, DistilBertModel, classificationHead, mlmHead } from './models.js';
import { generateTokens } from './generation.js';
import { GLBackend } from './webgl-backend.js';

/** Upload a parsed safetensors map to Tensors (GPU-aware, transposes huge rows). */
function loadWeights(parsed, backend) {
  const map = new Map();
  let gl = null;
  if (backend === 'webgl') gl = getGLSafe();
  for (const [name, t] of parsed.tensors) {
    if (backend === 'webgl') {
      const rows = t.shape.length >= 2 ? t.shape[t.shape.length - 2] : 1;
      const needTranspose = t.shape.length >= 2 && rows > gl.maxTex;
      if (needTranspose) {
        const R = t.shape[t.shape.length - 2], C = t.shape[t.shape.length - 1];
        const transposed = GLBackend.transpose(t.data, R, C);
        map.set(name, new Tensor([R, C], transposed, { transposedStorage: true }));
      } else {
        map.set(name, new Tensor(t.shape, t.data));
      }
    } else {
      map.set(name, new Tensor(t.shape, t.data, { backend: 'cpu' }));
    }
    t.data = null; // free staging copy (tensor owns the Float32Array now / uploaded)
  }
  return map;
}

let _glSafe = null;
function getGLSafe() {
  if (!_glSafe) _glSafe = (typeof document !== 'undefined') ? new GLBackend() : null;
  return _glSafe;
}

class PipelineBase {
  constructor(task, repo, model, tokenizer, config) {
    this.task = task;
    this.repo = repo;
    this.model = model;
    this.tokenizer = tokenizer;
    this.config = config;
  }
  dispose() { /* models hold weight tensors; drop references for GC */ }
}

// ---------------------------------------------------------------------------
export class TextGenerationPipeline extends PipelineBase {
  /**
   * await gen('prompt', { maxNewTokens, doSample, temperature, topK, topP,
   *                       repetitionPenalty, onToken }) ->
   *   [{ generated_text: "prompt + completion" }]
   */
  async _call(text, options = {}) {
    const ids = this.tokenizer.encode(text);
    if (ids.length === 0) ids.push(this.tokenizer.eosTokenId ?? 0);
    const maxPos = this.model.maxPos;
    if (ids.length > maxPos - 1) ids.splice(0, ids.length - (maxPos - 1));
    const gen = generateTokens(this.model, ids, {
      maxNewTokens: options.max_new_tokens ?? options.maxNewTokens ?? 50,
      doSample: options.do_sample ?? options.doSample ?? true,
      temperature: options.temperature ?? 1.0,
      topK: options.top_k ?? options.topK ?? 50,
      topP: options.top_p ?? options.topP ?? 0.95,
      repetitionPenalty: options.repetition_penalty ?? options.repetitionPenalty ?? 1.0,
      eosTokenId: options.eos_token_id ?? this.model.eosTokenId ?? this.tokenizer.eosTokenId ?? null,
      onToken: options.onToken || options.on_token || null,
    });
    const full = ids.concat(gen);
    const text2 = this.tokenizer.decode(full, { skipSpecialTokens: true });
    return [{ generated_text: text2 }];
  }
}

// ---------------------------------------------------------------------------
export class FeatureExtractionPipeline extends PipelineBase {
  /**
   * await fx('hello world', { pooling: 'mean'|'cls', normalize: true }) ->
   *   { data: Float32Array[d] }
   */
  async _call(text, options = {}) {
    const pooling = options.pooling || 'mean';
    const normalize = options.normalize ?? false;
    const ids = this.tokenizer.encode(text, { addSpecialTokens: true });
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    const [hidden] = arenaRun(() => this.model.forward(ids));
    const h = hidden.read(); // [s, d] contiguous
    const s = hidden.rows, d = hidden.cols;
    hidden.dispose();
    const out = new Float32Array(d);
    if (pooling === 'cls') {
      out.set(h.subarray(0, d));
    } else {
      for (let r = 0; r < s; r++) {
        const off = r * d;
        for (let c = 0; c < d; c++) out[c] += h[off + c];
      }
      for (let c = 0; c < d; c++) out[c] /= s;
    }
    if (normalize) {
      let norm = 0;
      for (let c = 0; c < d; c++) norm += out[c] * out[c];
      norm = Math.sqrt(norm) || 1;
      for (let c = 0; c < d; c++) out[c] /= norm;
    }
    return { data: out, dims: [d] };
  }
}

// ---------------------------------------------------------------------------
export class TextClassificationPipeline extends PipelineBase {
  /** await cls('text') -> [{ label, score }] (sorted, best first) */
  async _call(text, options = {}) {
    const ids = this.tokenizer.encode(text, { addSpecialTokens: true });
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    let logits;
    if (this.model.kind === 'distilbert') {
      logits = arenaRun(() => {
        const [hidden] = this.model.forward(ids);
        const first = hidden.firstRow();
        const out = classificationHead(this.model, this._weights, this.config, first);
        hidden.dispose();
        return out;
      });
    } else {
      logits = arenaRun(() => {
        const usePooler = !!this.model.hasPooler;
        const [hidden, pooled] = this.model.forward(ids, { pooled: usePooler });
        const src = usePooler ? pooled : hidden.firstRow();
        const out = classificationHead(this.model, this._weights, this.config, src);
        hidden.dispose();
        return out;
      });
    }
    const arr = logits.read();
    logits.dispose();
    const pairs = [];
    const id2label = (this.config && this.config.id2label) || {};
    for (let i = 0; i < arr.length; i++) {
      pairs.push({ label: String(id2label[i] ?? i), score: arr[i] });
    }
    // softmax over labels if logits look unnormalized
    let mx = -Infinity;
    for (const p of pairs) mx = Math.max(mx, p.score);
    let sum = 0;
    for (const p of pairs) { p.score = Math.exp(p.score - mx); sum += p.score; }
    for (const p of pairs) p.score /= sum;
    pairs.sort((a, b) => b.score - a.score);
    return pairs;
  }
}

// ---------------------------------------------------------------------------
export class FillMaskPipeline extends PipelineBase {
  /** await fm('The capital of France is [MASK].') -> top token predictions */
  async _call(text, options = {}) {
    const tok = this.tokenizer;
    const maskId = tok.maskTokenId;
    if (maskId === undefined) throw new Error('[mini.js] this tokenizer has no [MASK] token');
    const ids = tok.encode(text, { addSpecialTokens: true });
    const positions = [];
    for (let i = 0; i < ids.length; i++) if (ids[i] === maskId) positions.push(i);
    if (positions.length === 0) throw new Error('[mini.js] input must contain [MASK]');
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    const topK = options.top_k ?? options.topK ?? 5;

    const logits = arenaRun(() => {
      const [hidden] = this.model.forward(ids);
      const rows = [];
      for (const p of positions) rows.push(hidden.rowSlice(p));
      const stacked = rows.length === 1 ? rows[0] : cat1x(rows);
      const out = mlmHead(this.model, this._weights, this.config, stacked);
      hidden.dispose();
      return out;
    });
    const arr = logits.read();
    logits.dispose();

    const results = [];
    const perPos = arr.length / positions.length;
    for (let pi = 0; pi < positions.length; pi++) {
      const row = arr.subarray(pi * perPos, (pi + 1) * perPos);
      const idx = Array.from(row.keys());
      idx.sort((a, b) => row[b] - row[a]);
      const chosen = idx.slice(0, topK);
      // softmax over full vocab for calibrated scores
      let mx = -Infinity;
      for (let i = 0; i < row.length; i++) mx = Math.max(mx, row[i]);
      let sum = 0;
      const ex = new Float32Array(chosen.length);
      // compute denominators lazily on top-1000 for speed
      const idxTop = idx.slice(0, 1000);
      for (const i of idxTop) sum += Math.exp(row[i] - mx);
      for (let ci = 0; ci < chosen.length; ci++) {
        const id = chosen[ci];
        ex[ci] = Math.exp(row[id] - mx) / (sum || 1);
        const idsCopy = ids.slice();
        idsCopy[positions[pi]] = id;
        results.push({
          score: ex[ci],
          token: tok.idToToken.get(id),
          token_id: id,
          sequence: tok.decode(idsCopy, { skipSpecialTokens: true }),
        });
      }
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}

/** Stack [1,d] tensors -> [n, d] (backend-agnostic). */
function cat1x(rows) {
  const d = rows[0].cols;
  const out = Tensor.zeros([rows.length, d]);
  for (let i = 0; i < rows.length; i++) out.pasteRows(rows[i], i);
  return out;
}

// ---------------------------------------------------------------------------
// Registry / factory
// ---------------------------------------------------------------------------
const MODEL_CLASSES = {
  GPT2LMHeadModel: ['gpt2', GPT2Model],
  GPT2Model: ['gpt2', GPT2Model],
  BertModel: ['bert', BertModel],
  BertForMaskedLM: ['bert', BertModel],
  BertForSequenceClassification: ['bert', BertModel],
  BertForTokenClassification: ['bert', BertModel],
  DistilBertModel: ['distilbert', DistilBertModel],
  DistilBertForMaskedLM: ['distilbert', DistilBertModel],
  DistilBertForSequenceClassification: ['distilbert', DistilBertModel],
};

const TASKS = {
  'text-generation': TextGenerationPipeline,
  'feature-extraction': FeatureExtractionPipeline,
  'text-classification': TextClassificationPipeline,
  'fill-mask': FillMaskPipeline,
};

const pipelineCache = new Map();

/**
 * main entry: mini.pipeline(task, repo, options)
 */
export async function pipeline(task, repo, options = {}) {
  if (!TASKS[task]) {
    throw new Error(`[mini.js] unknown task "${task}". Available: ${Object.keys(TASKS).join(', ')}`);
  }
  const cacheKey = `${task}::${repo}`;
  if (!options.force && pipelineCache.has(cacheKey)) return pipelineCache.get(cacheKey);

  const progressCallback = options.progress_callback || options.progressCallback || null;
  const dl = await downloadModel(repo, { progressCallback, weightFile: options.weight_file || null });
  if (!dl.config) throw new Error('[mini.js] config.json missing in ' + repo);

  const parsed = parseSafetensors(dl.weightsBytes);
  dl.weightsBytes = null; // free raw bytes

  const backend = currentBackend();
  let weights;
  try {
    weights = loadWeights(parsed, backend);
  } catch (e) {
    // e.g. huge embedding matrices exceed this device's MAX_TEXTURE_SIZE
    if (backend === 'webgl') {
      console.warn('[mini.js] weights exceed GPU limits on this device — switching to CPU backend (' + e.message + ')');
      setBackend('cpu');
      weights = loadWeights(parsed, 'cpu');
    } else {
      throw e;
    }
  }

  const config = dl.config;
  const arch = (config.architectures && config.architectures[0]) || '';
  let entry = MODEL_CLASSES[arch];
  if (!entry) {
    // infer from model_type
    const mt = (config.model_type || '').toLowerCase();
    if (mt === 'gpt2') entry = MODEL_CLASSES.GPT2LMHeadModel;
    else if (mt === 'bert') entry = MODEL_CLASSES.BertModel;
    else if (mt === 'distilbert') entry = MODEL_CLASSES.DistilBertModel;
  }
  if (!entry) throw new Error(`[mini.js] unsupported architecture "${arch}" in ${repo} (supported: GPT2*, Bert*, DistilBert*)`);

  const [, ModelClass] = entry;
  const model = new ModelClass(weights, config);

  if (!dl.tokenizerJson) {
    throw new Error(`[mini.js] ${repo} has no tokenizer.json (legacy vocab/merges repos not supported yet)`);
  }
  const tokenizer = createTokenizer(dl.tokenizerJson, dl.tokenizerConfig, config.model_type);

  const pipe = new TASKS[task](task, repo, model, tokenizer, config);
  pipe._weights = weights;

  // Expose the pipeline as a real callable: `await pipe(text, options)`
  const fn = (input, options) => pipe._call(input, options || {});
  Object.assign(fn, {
    task,
    repo,
    model,
    tokenizer,
    config,
    _call: pipe._call.bind(pipe),
    dispose: () => { pipe.dispose(); pipelineCache.delete(cacheKey); },
  });
  pipelineCache.set(cacheKey, fn);
  progressCallback && progressCallback({ status: 'ready', task, model: repo });
  return fn;
}

/** Auto backend selection at import time (browser: WebGL2 if possible). */
export function ready() {
  detectBestBackend();
  return currentBackend();
}

export function disposePipeline(task, repo) {
  const key = `${task}::${repo}`;
  const p = pipelineCache.get(key);
  if (p) { p.dispose(); pipelineCache.delete(key); }
}
