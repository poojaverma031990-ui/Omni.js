/**
 * Builds tiny synthetic models in EXACT Hugging Face format (config.json,
 * tokenizer.json, model.safetensors) so the full mini.js stack can be tested
 * offline through a local HTTP server.
 */
import { writeSafetensors } from '../src/safetensors.js';
import { createServer } from 'node:http';

/** deterministic PRNG */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function randn(rng) {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// GPT-2 byte <-> unicode char for a byte value
const BYTE_TO_CHAR = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  const map = new Map();
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCharCode(cs[i]));
  return map;
})();
export const charOfByte = (b) => BYTE_TO_CHAR.get(b);

/**
 * Tiny GPT-2. vocab: "<|endoftext|>" + all 256 byte-chars + merge tokens.
 */
export function makeGPT2Files({
  seed = 42, nLayer = 2, nHead = 2, nEmb = 16, maxPos = 32,
} = {}) {
  const rng = mulberry32(seed);
  const vocab = new Map();
  const merges = [];
  vocab.set('<|endoftext|>', 0);
  let nextId = 1;
  for (let b = 0; b < 256; b++) {
    const ch = charOfByte(b);
    if (!vocab.has(ch)) vocab.set(ch, nextId++);
  }
  // merge chain h+e -> he, he+l -> hel, hel+l -> hell, hell+o -> hello, Ġ+t -> Ġt
  for (const [a, b] of [['h', 'e'], ['he', 'l'], ['hel', 'l'], ['hell', 'o'], [charOfByte(32), 't']]) {
    merges.push(`${a} ${b}`);
    vocab.set(a + b, nextId++);
  }
  const V = vocab.size;

  const entries = [];
  const w = (name, shape, scale = 0.25) => {
    const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    for (let i = 0; i < data.length; i++) data[i] = randn(rng) * scale;
    entries.push({ name, shape, data });
  };
  const ones = (name, shape) => {
    const data = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1);
    entries.push({ name, shape, data });
  };
  const zeros = (name, shape) => {
    entries.push({ name, shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)) });
  };

  w('transformer.wte.weight', [V, nEmb]);
  w('transformer.wpe.weight', [maxPos, nEmb]);
  for (let i = 0; i < nLayer; i++) {
    const p = `transformer.h.${i}`;
    ones(`${p}.ln_1.weight`, [nEmb]); zeros(`${p}.ln_1.bias`, [nEmb]);
    w(`${p}.attn.c_attn.weight`, [nEmb, 3 * nEmb]);
    zeros(`${p}.attn.c_attn.bias`, [3 * nEmb]);
    w(`${p}.attn.c_proj.weight`, [nEmb, nEmb]);
    zeros(`${p}.attn.c_proj.bias`, [nEmb]);
    ones(`${p}.ln_2.weight`, [nEmb]); zeros(`${p}.ln_2.bias`, [nEmb]);
    w(`${p}.mlp.c_fc.weight`, [nEmb, 4 * nEmb]);
    zeros(`${p}.mlp.c_fc.bias`, [4 * nEmb]);
    w(`${p}.mlp.c_proj.weight`, [4 * nEmb, nEmb]);
    zeros(`${p}.mlp.c_proj.bias`, [nEmb]);
  }
  ones('transformer.ln_f.weight', [nEmb]); zeros('transformer.ln_f.bias', [nEmb]);

  const config = {
    architectures: ['GPT2LMHeadModel'],
    model_type: 'gpt2',
    n_layer: nLayer, n_head: nHead, n_embd: nEmb,
    n_positions: maxPos, n_ctx: maxPos,
    vocab_size: V,
    bos_token_id: 0, eos_token_id: 0,
    layer_norm_epsilon: 1e-5,
  };
  const tokenizerJson = {
    version: '1.0',
    added_tokens: [{ id: 0, special: true, content: '<|endoftext|>', single_word: false, lstrip: false, rstrip: false, normalized: false }],
    normalizer: null,
    pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true },
    post_processor: null,
    decoder: { type: 'ByteLevel' },
    model: {
      type: 'BPE', dropout: null, unk_token: null,
      continuing_subword_prefix: '', end_of_word_suffix: '', fuse_unk: false,
      vocab: Object.fromEntries(vocab),
      merges,
    },
  };
  const tokenizerConfig = {
    unk_token: '<|endoftext|>', bos_token: '<|endoftext|>', eos_token: '<|endoftext|>',
    tokenizer_class: 'GPT2Tokenizer', model_max_length: maxPos,
  };

  const st = writeSafetensors(entries.map(e => [e.name, e]));
  return {
    files: {
      'config.json': JSON.stringify(config),
      'tokenizer.json': JSON.stringify(tokenizerJson),
      'tokenizer_config.json': JSON.stringify(tokenizerConfig),
      'model.safetensors': st,
    },
    vocab, merges, config,
  };
}

/**
 * Tiny BERT with WordPiece tokenizer + MLM head + 2-label classification head.
 */
export function makeBertFiles({
  seed = 7, nLayer = 2, nHead = 2, dim = 16, inter = 24, maxPos = 32, typeVocab = 2,
} = {}) {
  const rng = mulberry32(seed + 100);
  const vocab = new Map();
  const toks = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'the', 'cat', '##s', 'dog', 'big', 'small', 'a', 'runs', '.', 'hello', 'world'];
  toks.forEach((t, i) => vocab.set(t, i));
  const V = vocab.size;

  const entries = [];
  const w = (name, shape, scale = 0.25) => {
    const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    for (let i = 0; i < data.length; i++) data[i] = randn(rng) * scale;
    entries.push({ name, shape, data });
  };
  const ones = (name, shape) => entries.push({ name, shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1) });
  const zeros = (name, shape) => entries.push({ name, shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)) });

  w('bert.embeddings.word_embeddings.weight', [V, dim]);
  w('bert.embeddings.position_embeddings.weight', [maxPos, dim]);
  w('bert.embeddings.token_type_embeddings.weight', [typeVocab, dim]);
  ones('bert.embeddings.LayerNorm.weight', [dim]); zeros('bert.embeddings.LayerNorm.bias', [dim]);
  for (let i = 0; i < nLayer; i++) {
    const p = `bert.encoder.layer.${i}`;
    w(`${p}.attention.self.query.weight`, [dim, dim]); zeros(`${p}.attention.self.query.bias`, [dim]);
    w(`${p}.attention.self.key.weight`, [dim, dim]); zeros(`${p}.attention.self.key.bias`, [dim]);
    w(`${p}.attention.self.value.weight`, [dim, dim]); zeros(`${p}.attention.self.value.bias`, [dim]);
    w(`${p}.attention.output.dense.weight`, [dim, dim]); zeros(`${p}.attention.output.dense.bias`, [dim]);
    ones(`${p}.attention.output.LayerNorm.weight`, [dim]); zeros(`${p}.attention.output.LayerNorm.bias`, [dim]);
    w(`${p}.intermediate.dense.weight`, [inter, dim]); zeros(`${p}.intermediate.dense.bias`, [inter]);
    w(`${p}.output.dense.weight`, [dim, inter]); zeros(`${p}.output.dense.bias`, [dim]);
    ones(`${p}.output.LayerNorm.weight`, [dim]); zeros(`${p}.output.LayerNorm.bias`, [dim]);
  }
  w('bert.pooler.dense.weight', [dim, dim]); zeros('bert.pooler.dense.bias', [dim]);
  // MLM head
  ones('cls.predictions.transform.LayerNorm.weight', [dim]); zeros('cls.predictions.transform.LayerNorm.bias', [dim]);
  w('cls.predictions.decoder.weight', [V, dim]); zeros('cls.predictions.bias', [V]);
  // classification head (2 labels)
  w('classifier.weight', [2, dim]); zeros('classifier.bias', [2]);

  const config = {
    architectures: ['BertForMaskedLM'],
    model_type: 'bert',
    num_hidden_layers: nLayer, num_attention_heads: nHead, hidden_size: dim,
    intermediate_size: inter, vocab_size: V, max_position_embeddings: maxPos,
    type_vocab_size: typeVocab, layer_norm_eps: 1e-12,
    id2label: { 0: 'NEGATIVE', 1: 'POSITIVE' },
  };
  const tokenizerJson = {
    version: '1.0',
    added_tokens: [0, 1, 2, 3, 4].map(id => ({
      id, special: true, content: toks[id], single_word: false, lstrip: false, rstrip: false, normalized: false,
    })),
    normalizer: { type: 'BertNormalizer', lowercase: true, strip_accents: true },
    pre_tokenizer: { type: 'BertPreTokenizer' },
    model: {
      type: 'WordPiece',
      unk_token: '[UNK]',
      continuing_subword_prefix: '##',
      max_input_chars_per_word: 100,
      vocab: Object.fromEntries(vocab),
    },
  };
  const tokenizerConfig = {
    do_lower_case: true, unk_token: '[UNK]', pad_token: '[PAD]',
    cls_token: '[CLS]', sep_token: '[SEP]', mask_token: '[MASK]',
    tokenizer_class: 'BertTokenizer',
  };

  const st = writeSafetensors(entries.map(e => [e.name, e]));
  return {
    files: {
      'config.json': JSON.stringify(config),
      'tokenizer.json': JSON.stringify(tokenizerJson),
      'tokenizer_config.json': JSON.stringify(tokenizerConfig),
      'model.safetensors': st,
    },
    vocab, config,
  };
}

/** Simple static file server for tests. Returns {url, close}. */
export function serveFiles(files, port = 0) {
  
  const server = createServer((req, res) => {
    const path = decodeURIComponent(req.url.split('?')[0]);
    let name = path.replace(/^\/+/, '');
    // mimic HF: /{repo}/resolve/{revision}/{file} -> {repo}/{file}
    const m = name.match(/^(.+)\/resolve\/[^/]+\/(.+)$/);
    if (m) name = `${m[1]}/${m[2]}`;
    const body = files[name];
    if (body === undefined) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('not found: ' + name);
      return;
    }
    const buf = typeof body === 'string' ? Buffer.from(body) : Buffer.from(body);
    res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': buf.length });
    res.end(buf);
  });
  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => {
      resolve({ url: `http://127.0.0.1:${server.address().port}`, close: () => server.close() });
    });
  });
}
