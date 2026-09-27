/**
 * mini.js — a Transformers.js alternative, written 100% from scratch in pure JS.
 *
 * One line of code, any browser in the world:
 *
 *   <script src="mini.js"><\/script>
 *   <script>
 *     const gen = await mini.pipeline('text-generation', 'openai-community/gpt2');
 *     const [out] = await gen('Once upon a time');
 *   <\/script>
 *
 * Backends: WebGL2 (hand-written GLSL kernels) with automatic CPU fallback.
 * No ONNX Runtime. No TF.js. No external code. HTML+CSS+JS only.
 */
export { MINI_VERSION as version, formatBytes } from './utils.js';
export { Tensor, setBackend, detectBestBackend, currentBackend, arenaRun } from './tensor.js';
export { GLBackend } from './webgl-backend.js';
export { parseSafetensors } from './safetensors.js';
export { Tokenizer, createTokenizer } from './tokenizer.js';
export { env, downloadFile, clearCache } from './hub.js';
export { GPT2Model, BertModel, DistilBertModel, LlamaModel, RobertaModel } from './models.js';
export { generateTokens, generateAsync } from './generation.js';
export { pipeline, ready, disposePipeline, TextGenerationPipeline, FeatureExtractionPipeline, TextClassificationPipeline, TokenClassificationPipeline, FillMaskPipeline, QuestionAnsweringPipeline } from './pipelines.js';
import { buildChatPrompt, renderChatTemplate } from './chat-template.js';
export { buildChatPrompt, renderChatTemplate };
export * from './pipelines.js';
import { selfTest, initBackend } from './selftest.js';
export { selfTest, initBackend };

// One-line convenience APIs (download + cache the default model on first use)
const DEFAULT_MODELS = {
  'text-generation': 'openai-community/gpt2',
  'feature-extraction': 'sentence-transformers/all-MiniLM-L6-v2',
  'text-classification': 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
  'fill-mask': 'distilbert/distilbert-base-uncased',
};

const _defaultPipelines = {};
async function defaultPipeline(task) {
  if (!_defaultPipelines[task]) {
    _defaultPipelines[task] = await pipeline(task, DEFAULT_MODELS[task]);
  }
  return _defaultPipelines[task];
}

/** One line text generation: await mini.generate('Once upon a time') */
async function generate(text, options = {}) {
  const model = options.model || options.repo || DEFAULT_MODELS['text-generation'];
  const pipe = model === DEFAULT_MODELS['text-generation']
    ? await defaultPipeline('text-generation')
    : await pipeline('text-generation', model);
  const [out] = await pipe(text, options);
  return out.generated_text;
}

/** One line embeddings: const v = await mini.embed('hello') */
async function embed(text, options = {}) {
  const model = options.model || DEFAULT_MODELS['feature-extraction'];
  const pipe = model === DEFAULT_MODELS['feature-extraction']
    ? await defaultPipeline('feature-extraction')
    : await pipeline('feature-extraction', model);
  const out = await pipe(text, { pooling: 'mean', normalize: true, ...options });
  return out.data;
}

/** One line sentiment/etc: const r = await mini.classify('I love this!') */
async function classify(text, options = {}) {
  const model = options.model || DEFAULT_MODELS['text-classification'];
  const pipe = model === DEFAULT_MODELS['text-classification']
    ? await defaultPipeline('text-classification')
    : await pipeline('text-classification', model);
  const out = await pipe(text, options);
  return out;
}

/** One line masked-lm: const r = await mini.fillMask('Paris is the [MASK] of France') */
async function fillMask(text, options = {}) {
  const model = options.model || DEFAULT_MODELS['fill-mask'];
  const pipe = model === DEFAULT_MODELS['fill-mask']
    ? await defaultPipeline('fill-mask')
    : await pipeline('fill-mask', model);
  return pipe(text, options);
}

const mini = {
  version: MINI_VERSION,
  env,
  Tensor,
  setBackend,
  detectBestBackend,
  currentBackend,
  arenaRun,
  parseSafetensors,
  Tokenizer,
  downloadFile,
  clearCache,
  pipeline,
  ready,
  disposePipeline,
  selfTest,
  initBackend,
  generate,
  generateTokens,
  generateAsync,
  buildChatPrompt,
  renderChatTemplate,
  embed,
  classify,
  fillMask,
  TASK_INFO: {
    tasks: Object.keys(DEFAULT_MODELS),
    defaultModels: { ...DEFAULT_MODELS },
  },
};

// Browser global + Node/worker friendly export
if (typeof globalThis !== 'undefined') globalThis.mini = mini;
if (typeof self !== 'undefined') self.mini = mini;
if (typeof window !== 'undefined') window.mini = mini;
if (typeof module !== 'undefined' && module.exports) module.exports = mini;

export default mini;
