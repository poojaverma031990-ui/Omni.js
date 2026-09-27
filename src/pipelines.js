/**
 * mini.js — High-level pipelines, Transformers.js-style but 100% from scratch:
 *   const gen = await mini.pipeline('text-generation', 'HuggingFaceTB/SmolLM2-135M-Instruct');
 *   const chat = await gen.chat([{ role: 'user', content: 'Hello!' }]);
 *
 * Tasks: text-generation (+chat), feature-extraction, text-classification,
 * fill-mask, question-answering, token-classification.
 */
import { Tensor, arenaRun, currentBackend, detectBestBackend, setBackend } from './tensor.js';
import { createTokenizer } from './tokenizer.js';
import { downloadModel } from './hub.js';
import {
  GPT2Model, BertModel, DistilBertModel, LlamaModel, RobertaModel,
  classificationHead, tokenClassificationHead, qaHead, mlmHead,
} from './models.js';
import { generateAsync } from './generation.js';
import { GLBackend, getGL } from './webgl-backend.js';

/** Upload a parsed safetensors map to Tensors (GPU-aware, transposes huge rows). */
function loadWeights(tensors, backend) {
  const map = new Map();
  let gl = null;
  if (backend === 'webgl') gl = getGL();
  for (const [name, t] of tensors) {
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
    t.data = null; // free staging copy
  }
  return map;
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
  /** Encode with context-window truncation (keep the END of the prompt). */
  _encodeBounded(text) {
    const ids = this.tokenizer.encode(text);
    const maxPos = this.model.maxPos;
    if (ids.length > maxPos - 1) ids.splice(0, ids.length - (maxPos - 1));
    return ids;
  }

  /** Plain completion. `onToken(id, n)` — return false to stop. */
  async _call(text, options = {}) {
    const ids = this._encodeBounded(text);
    if (ids.length === 0) ids.push(this.tokenizer.eosTokenId ?? 0);
    const gen = await generateAsync(this.model, ids, {
      maxNewTokens: options.max_new_tokens ?? options.maxNewTokens ?? 50,
      doSample: options.do_sample ?? options.doSample ?? true,
      temperature: options.temperature ?? 1.0,
      topK: options.top_k ?? options.topK ?? 50,
      topP: options.top_p ?? options.topP ?? 0.95,
      repetitionPenalty: options.repetition_penalty ?? options.repetitionPenalty ?? 1.0,
      eosTokenId: options.eos_token_id ?? this.model.eosTokenId ?? this.tokenizer.eosTokenId ?? null,
      onToken: options.onToken || options.on_token || null,
      tick: options.tick,
    });
    const full = ids.concat(gen);
    return [{ generated_text: this.tokenizer.decode(full, { skipSpecialTokens: true }) }];
  }

  /**
   * Chat! Applies the model's real chat template (or a builtin family one).
   *   await pipe.chat([{role:'user', content:'Hi'}], {max_new_tokens: 200})
   * Returns [{ generated_text, assistant_message }]
   * options.onText(delta, fullText) streams the assistant reply as text.
   */
  async chat(messages, options = {}) {
    if (typeof messages === 'string') messages = [{ role: 'user', content: messages }];
    const prompt = this.tokenizer.applyChatTemplate(messages, {
      addGenerationPrompt: options.add_generation_prompt ?? true,
      modelType: this.config.model_type,
    });
    const promptIds = this.tokenizer.encode(prompt);
    const maxPos = this.model.maxPos;
    if (promptIds.length > maxPos - 1) promptIds.splice(0, promptIds.length - (maxPos - 1));

    let text = '';
    const eos = options.eos_token_id ?? this.tokenizer.eosTokenId ?? this.model.eosTokenId ?? null;
    const onToken = options.onToken || null;
    const onText = options.onText || null;

    const gen = await generateAsync(this.model, promptIds, {
      maxNewTokens: options.max_new_tokens ?? options.maxNewTokens ?? 256,
      doSample: options.do_sample ?? options.doSample ?? true,
      temperature: options.temperature ?? 0.7,
      topK: options.top_k ?? options.topK ?? 50,
      topP: options.top_p ?? options.topP ?? 0.95,
      repetitionPenalty: options.repetition_penalty ?? options.repetitionPenalty ?? 1.0,
      eosTokenId: eos,
      tick: options.tick,
      onToken: (id, n) => {
        if (onToken && onToken(id, n) === false) return false;
        if (onText && id !== eos) {
          const full = this.tokenizer.decode(promptIds.concat(gen), { skipSpecialTokens: true });
          // token-level decode delta (fast path): decode just the new id
          const piece = this.tokenizer.decode([id], { skipSpecialTokens: true });
          onText(piece, text + piece);
          text += piece;
        }
        return true;
      },
    });

    const fullText = this.tokenizer.decode(promptIds.concat(gen), { skipSpecialTokens: true });
    const assistant = this.tokenizer.decode(gen, { skipSpecialTokens: true });
    return [{ generated_text: fullText, assistant_message: assistant }];
  }
}

// ---------------------------------------------------------------------------
export class FeatureExtractionPipeline extends PipelineBase {
  async _call(text, options = {}) {
    const pooling = options.pooling || 'mean';
    const normalize = options.normalize ?? false;
    const ids = this.tokenizer.encode(text, { addSpecialTokens: true });
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    const [hidden] = arenaRun(() => this.model.forward(ids));
    const h = hidden.read();
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
  async _call(text, options = {}) {
    const ids = this.tokenizer.encode(text, { addSpecialTokens: true });
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    const logits = arenaRun(() => {
      let src;
      if (this.model.kind === 'distilbert') {
        const [hidden] = this.model.forward(ids);
        src = hidden.firstRow();
        const out = classificationHead(this.model, this._weights, this.config, src);
        hidden.dispose();
        return out;
      }
      if (this.model.kind === 'roberta') {
        const [hidden] = this.model.forward(ids); // no pooler: first token directly
        src = hidden.firstRow();
        const out = classificationHead(this.model, this._weights, this.config, src);
        hidden.dispose();
        return out;
      }
      const usePooler = !!this.model.hasPooler;
      const [hidden, pooled] = this.model.forward(ids, { pooled: usePooler });
      src = usePooler ? pooled : hidden.firstRow();
      const out = classificationHead(this.model, this._weights, this.config, src);
      hidden.dispose();
      return out;
    });
    const arr = logits.read();
    logits.dispose();
    const pairs = [];
    const id2label = (this.config && this.config.id2label) || {};
    for (let i = 0; i < arr.length; i++) {
      pairs.push({ label: String(id2label[i] ?? i), score: arr[i] });
    }
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
export class TokenClassificationPipeline extends PipelineBase {
  /** NER-style: returns per-token predictions [{token, label, score}] */
  async _call(text, options = {}) {
    const ids = this.tokenizer.encode(text, { addSpecialTokens: true });
    if (ids.length > this.model.maxPos) ids.length = this.model.maxPos;
    const logits = arenaRun(() => {
      const [hidden] = this.model.forward(ids);
      const out = tokenClassificationHead(this.model, this._weights, hidden);
      hidden.dispose();
      return out;
    });
    const arr = logits.read(); // [s, L]
    logits.dispose();
    const L = arr.length / ids.length;
    const id2label = (this.config && this.config.id2label) || {};
    const results = [];
    for (let r = 0; r < ids.length; r++) {
      let best = 0;
      for (let l = 1; l < L; l++) if (arr[r * L + l] > arr[r * L + best]) best = l;
      // softmax over labels for the score
      let mx = -Infinity, sum = 0;
      for (let l = 0; l < L; l++) mx = Math.max(mx, arr[r * L + l]);
      for (let l = 0; l < L; l++) sum += Math.exp(arr[r * L + l] - mx);
      results.push({
        token: this.tokenizer.idToToken.get(ids[r]),
        label: String(id2label[best] ?? best),
        score: Math.exp(arr[r * L + best] - mx) / sum,
      });
    }
    return results;
  }
}

// ---------------------------------------------------------------------------
export class FillMaskPipeline extends PipelineBase {
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
      let mx = -Infinity;
      for (let i = 0; i < row.length; i++) mx = Math.max(mx, row[i]);
      let sum = 0;
      const idxTop = idx.slice(0, 1000);
      for (const i of idxTop) sum += Math.exp(row[i] - mx);
      for (let ci = 0; ci < chosen.length; ci++) {
        const id = chosen[ci];
        const idsCopy = ids.slice();
        idsCopy[positions[pi]] = id;
        results.push({
          score: Math.exp(row[id] - mx) / (sum || 1),
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

// ---------------------------------------------------------------------------
export class QuestionAnsweringPipeline extends PipelineBase {
  /** await qa({question, context}) -> [{answer, score}] */
  async _call(input, options = {}) {
    let question, context;
    if (typeof input === 'string') { question = input; context = options.context || ''; }
    else { question = input.question; context = input.context; }
    if (!question || !context) throw new Error('[mini.js] question-answering needs {question, context}');

    const tok = this.tokenizer;
    const cls = tok.clsTokenId ?? 101, sep = tok.sepTokenId ?? 102;
    const qIds = tok.encode(question);
    let cIds = tok.encode(context);
    const maxPos = this.model.maxPos;
    const budget = maxPos - qIds.length - 3;
    if (budget <= 0) throw new Error('[mini.js] question too long for this model');
    if (cIds.length > budget) cIds = cIds.slice(0, budget);
    const ids = [cls, ...qIds, sep, ...cIds, sep];
    const ctxStart = qIds.length + 2;            // first context token index
    const ctxEnd = ids.length - 2;               // last context token index (inclusive)

    const logits = arenaRun(() => {
      const [hidden] = this.model.forward(ids);
      const out = qaHead(this.model, this._weights, hidden);
      hidden.dispose();
      return out;
    });
    const arr = logits.read(); // [s, 2]
    logits.dispose();

    // best span inside the context
    let bestS = ctxStart, bestE = ctxStart, bestScore = -Infinity;
    for (let a = ctxStart; a <= ctxEnd; a++) {
      const sa = arr[a * 2];
      // limit span length to 30 tokens for sane answers + speed
      const eMax = Math.min(ctxEnd, a + 30);
      for (let b = a; b <= eMax; b++) {
        const sc = sa + arr[b * 2 + 1];
        if (sc > bestScore) { bestScore = sc; bestS = a; bestE = b; }
      }
    }
    const spanIds = ids.slice(bestS, bestE + 1);
    const answer = tok.decode(spanIds, { skipSpecialTokens: true });
    // calibrated score over valid positions (softmax of chosen start+end)
    let sMx = -Infinity, eMx = -Infinity;
    for (let a = ctxStart; a <= ctxEnd; a++) sMx = Math.max(sMx, arr[a * 2]);
    for (let b = ctxStart; b <= ctxEnd; b++) eMx = Math.max(eMx, arr[b * 2 + 1]);
    let sSum = 0, eSum = 0;
    for (let a = ctxStart; a <= ctxEnd; a++) sSum += Math.exp(arr[a * 2] - sMx);
    for (let b = ctxStart; b <= ctxEnd; b++) eSum += Math.exp(arr[b * 2 + 1] - eMx);
    const score = (Math.exp(arr[bestS * 2] - sMx) / sSum) * (Math.exp(arr[bestE * 2 + 1] - eMx) / eSum);
    return [{ answer, score, start: bestS, end: bestE }];
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
const ARCH_REGISTRY = {
  // causal LMs
  GPT2LMHeadModel: ['gpt2', GPT2Model],
  GPT2Model: ['gpt2', GPT2Model],
  LlamaForCausalLM: ['llama', LlamaModel],
  MistralForCausalLM: ['llama', LlamaModel],
  Qwen2ForCausalLM: ['llama', LlamaModel],
  Qwen3ForCausalLM: ['llama', LlamaModel],
  GemmaForCausalLM: ['llama', LlamaModel],
  Gemma2ForCausalLM: ['llama', LlamaModel],
  Gemma3ForCausalLM: ['llama', LlamaModel],
  SmolLM2ForCausalLM: ['llama', LlamaModel],
  // encoders
  BertModel: ['bert', BertModel],
  BertForMaskedLM: ['bert', BertModel],
  BertForSequenceClassification: ['bert', BertModel],
  BertForTokenClassification: ['bert', BertModel],
  BertForQuestionAnswering: ['bert', BertModel],
  DistilBertModel: ['distilbert', DistilBertModel],
  DistilBertForMaskedLM: ['distilbert', DistilBertModel],
  DistilBertForSequenceClassification: ['distilbert', DistilBertModel],
  DistilBertForQuestionAnswering: ['distilbert', DistilBertModel],
  RobertaModel: ['roberta', RobertaModel],
  RobertaForMaskedLM: ['roberta', RobertaModel],
  RobertaForSequenceClassification: ['roberta', RobertaModel],
  RobertaForTokenClassification: ['roberta', RobertaModel],
};

const TYPE_FALLBACK = {
  gpt2: 'gpt2', llama: 'llama', mistral: 'llama', qwen2: 'llama', qwen3: 'llama',
  gemma: 'llama', gemma2: 'llama', gemma3: 'llama', smollm2: 'llama',
  bert: 'bert', distilbert: 'distilbert', roberta: 'roberta',
};

const TASKS = {
  'text-generation': TextGenerationPipeline,
  'feature-extraction': FeatureExtractionPipeline,
  'text-classification': TextClassificationPipeline,
  'token-classification': TokenClassificationPipeline,
  'fill-mask': FillMaskPipeline,
  'question-answering': QuestionAnsweringPipeline,
};

const pipelineCache = new Map();

/** main entry: mini.pipeline(task, repo, options) */
export async function pipeline(task, repo, options = {}) {
  if (!TASKS[task]) {
    throw new Error(`[mini.js] unknown task "${task}". Available: ${Object.keys(TASKS).join(', ')}`);
  }
  const cacheKey = `${task}::${repo}`;
  if (!options.force && pipelineCache.has(cacheKey)) return pipelineCache.get(cacheKey);

  const progressCallback = options.progress_callback || options.progressCallback || null;
  const dl = await downloadModel(repo, { progressCallback, weightFile: options.weight_file || null });
  if (!dl.config) throw new Error('[mini.js] config.json missing in ' + repo);

  const backend = currentBackend();
  let weights;
  try {
    weights = loadWeights(dl.tensors, backend);
  } catch (e) {
    // e.g. huge embedding matrices exceed this device's MAX_TEXTURE_SIZE
    if (backend === 'webgl') {
      console.warn('[mini.js] weights exceed GPU limits on this device — switching to CPU backend (' + e.message + ')');
      setBackend('cpu');
      weights = loadWeights(dl.tensors, 'cpu');
    } else {
      throw e;
    }
  }

  const config = dl.config;
  const arch = (config.architectures && config.architectures[0]) || '';
  let entry = ARCH_REGISTRY[arch];
  if (!entry) {
    const mt = (config.model_type || '').toLowerCase();
    const kind = TYPE_FALLBACK[mt];
    if (kind === 'gpt2') entry = ARCH_REGISTRY.GPT2LMHeadModel;
    else if (kind === 'llama') entry = ARCH_REGISTRY.LlamaForCausalLM;
    else if (kind === 'bert') entry = ARCH_REGISTRY.BertModel;
    else if (kind === 'distilbert') entry = ARCH_REGISTRY.DistilBertModel;
    else if (kind === 'roberta') entry = ARCH_REGISTRY.RobertaModel;
  }
  if (!entry) {
    throw new Error(`[mini.js] unsupported architecture "${arch}" in ${repo}\n` +
      `Supported: GPT-2, Llama, Mistral, Qwen2/3, Gemma 1/2/3, BERT, DistilBERT, RoBERTa`);
  }

  const [, ModelClass] = entry;
  const model = new ModelClass(weights, config);

  if (!dl.tokenizerJson) {
    throw new Error(`[mini.js] ${repo} has no tokenizer.json (legacy vocab/merges repos not supported yet)`);
  }
  const tokenizer = createTokenizer(dl.tokenizerJson, dl.tokenizerConfig, config.model_type);

  const pipe = new TASKS[task](task, repo, model, tokenizer, config);
  pipe._weights = weights;

  // Expose the pipeline as a real callable: `await pipe(input, options)`
  const fn = (input, options2) => pipe._call(input, options2 || {});
  Object.assign(fn, {
    task,
    repo,
    model,
    tokenizer,
    config,
    chat: pipe.chat ? pipe.chat.bind(pipe) : undefined,
    _call: pipe._call.bind(pipe),
    dispose: () => { pipe.dispose(); pipelineCache.delete(cacheKey); },
  });
  pipelineCache.set(cacheKey, fn);
  progressCallback && progressCallback({ status: 'ready', task, model: repo });
  return fn;
}

/** Auto backend selection (browser: WebGL2 if possible, after kernel self-test). */
export function ready() {
  detectBestBackend();
  return currentBackend();
}

export function disposePipeline(task, repo) {
  const key = `${task}::${repo}`;
  const p = pipelineCache.get(key);
  if (p) { p.dispose(); pipelineCache.delete(key); }
}
