/**
 * mini.js unit tests — CPU kernels, safetensors, tokenizers, KV-cache consistency.
 * Run: node test/unit.test.mjs
 */
import assert from 'node:assert/strict';
import { writeSafetensors, parseSafetensors } from '../src/safetensors.js';
import * as K from '../src/cpu-kernels.js';
import { Tokenizer } from '../src/tokenizer.js';
import { makeGPT2Files, makeBertFiles } from './synth.mjs';
import '../mini.js';
const mini = globalThis.mini;
mini.setBackend('cpu');

let passed = 0;
function ok(name, fn) {
  try { fn(); passed++; console.log('  ✓', name); }
  catch (e) { console.error('  ✗', name, '\n    ', e.message); process.exitCode = 1; }
}
async function okAsync(name, fn) {
  try { await fn(); passed++; console.log('  ✓', name); }
  catch (e) { console.error('  ✗', name, '\n    ', e.message); process.exitCode = 1; }
}

function randArr(n, seed = 1) {
  let a = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a = (a * 1103515245 + 12345) & 0x7fffffff;
    out[i] = ((a / 0x7fffffff) * 2 - 1);
  }
  return out;
}
function close(a, b, tol = 1e-5) {
  assert.equal(a.length, b.length, 'length mismatch');
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  assert.ok(maxDiff < tol, `max diff ${maxDiff} >= ${tol}`);
}

console.log('\n[1] CPU kernels');

ok('matmul NN matches naive reference', () => {
  const M = 5, Kd = 7, N = 3;
  const A = randArr(M * Kd, 1), B = randArr(Kd * N, 2);
  const out = new Float32Array(M * N);
  K.cpuMatmul(A, B, out, M, Kd, N, false);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let ref = 0;
    for (let k = 0; k < Kd; k++) ref += A[m * Kd + k] * B[k * N + n];
    assert.ok(Math.abs(out[m * N + n] - ref) < 1e-4);
  }
});

ok('matmul TN matches naive reference', () => {
  const M = 4, Kd = 6, N = 5;
  const A = randArr(M * Kd, 3), B = randArr(N * Kd, 4); // B [N,K]
  const out = new Float32Array(M * N);
  K.cpuMatmul(A, B, out, M, Kd, N, true);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let ref = 0;
    for (let k = 0; k < Kd; k++) ref += A[m * Kd + k] * B[n * Kd + k];
    assert.ok(Math.abs(out[m * N + n] - ref) < 1e-4);
  }
});

ok('softmaxRows is stable and normalized', () => {
  const x = new Float32Array([1000, 1001, 1002, -1000, 3, 1, 2, 0]);
  K.cpuSoftmaxRows(x, 2, 4);
  assert.ok(Math.abs(x[0] + x[1] + x[2] - 1) < 1e-6);
  assert.ok(Math.abs(x[3]) < 1e-30); // -3000 exp -> 0
  assert.ok(Math.abs(x[4] + x[5] + x[6] + x[7] - 1) < 1e-6);
});

ok('layerNorm matches reference', () => {
  const rows = 3, cols = 8;
  const x = randArr(rows * cols, 5);
  const g = randArr(cols, 6).map(v => 1 + v * 0.1);
  const b = randArr(cols, 7).map(v => v * 0.1);
  const out = K.cpuLayerNorm(x, rows, cols, g, b, 1e-5);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[r * cols + c];
    mean /= cols;
    let varr = 0;
    for (let c = 0; c < cols; c++) { const d = x[r * cols + c] - mean; varr += d * d; }
    varr /= cols;
    for (let c = 0; c < cols; c++) {
      const ref = (x[r * cols + c] - mean) / Math.sqrt(varr + 1e-5) * g[c] + b[c];
      assert.ok(Math.abs(out[r * cols + c] - ref) < 1e-5);
    }
  }
});

ok('gelu(0)=0, gelu is monotone-ish around large |x|', () => {
  const g = K.cpuGelu(new Float32Array([-100, 0, 100]));
  assert.ok(Math.abs(g[0] - 0) < 1e-6 && g[1] === 0 && Math.abs(g[2] - 100) < 1e-3);
});

ok('embedding gather and sliceCols', () => {
  const V = 4, C = 6;
  const w = randArr(V * C, 8);
  const out = K.cpuEmbedding(w, C, [2, 0, 3], 3);
  for (let c = 0; c < C; c++) assert.equal(out[c], w[2 * C + c]);
  const sliced = K.cpuSliceCols(out, 3, C, 2, 3);
  assert.equal(sliced[0], out[2]);
  assert.equal(sliced[2], out[4]);
});

console.log('\n[2] safetensors parser');

ok('round-trips F32', () => {
  const entries = [
    ['a.weight', { shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) }],
    ['b.bias', { shape: [3], data: new Float32Array([7, 8, 9]) }],
  ];
  const bytes = writeSafetensors(entries, { note: 'test' });
  const parsed = parseSafetensors(bytes);
  assert.equal(parsed.tensors.size, 2);
  const a = parsed.tensors.get('a.weight');
  assert.deepEqual(a.shape, [2, 3]);
  close(a.data, new Float32Array([1, 2, 3, 4, 5, 6]));
});

ok('parses F16 with exact bit patterns', () => {
  // manual F16 safetensors: values 1.0 (0x3C00), -2.0 (0xC000), 0.5 (0x3800), NaN skip
  const f16vals = [0x3c00, 0xc000, 0x3800, 0x0000]; // 1.0, -2.0, 0.5, 0.0
  const header = { x: { dtype: 'F16', shape: [4], data_offsets: [0, 8] } };
  const hb = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + hb.length + 8);
  new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
  out.set(hb, 8);
  const dv = new DataView(out.buffer);
  f16vals.forEach((v, i) => dv.setUint16(8 + hb.length + i * 2, v, true));
  const parsed = parseSafetensors(out);
  const x = parsed.tensors.get('x');
  assert.equal(x.data[0], 1.0);
  assert.equal(x.data[1], -2.0);
  assert.equal(x.data[2], 0.5);
  assert.equal(x.data[3], 0.0);
});

ok('parses BF16', () => {
  const bf = new Uint16Array([0x3f80, 0xbf80, 0x4000]); // 1.0, -1.0, 2.0 (as truncated f32)
  const header = { x: { dtype: 'BF16', shape: [3], data_offsets: [0, 6] } };
  const hb = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + hb.length + 6);
  new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
  out.set(hb, 8);
  bf.forEach((v, i) => new DataView(out.buffer).setUint16(8 + hb.length + i * 2, v, true));
  const x = parseSafetensors(out).tensors.get('x');
  assert.equal(x.data[0], 1.0);
  assert.equal(x.data[1], -1.0);
  assert.equal(x.data[2], 2.0);
});

console.log('\n[3] tokenizers (from scratch)');

const gpt2synth = makeGPT2Files();
{
  const tj = JSON.parse(gpt2synth.files['tokenizer.json']);
  const tok = new Tokenizer(tj, { unk_token: '<|endoftext|>' }, 'bpe');

  ok('BPE: known merge chain collapses to single token', () => {
    const ids = tok.encode('hello');
    assert.deepEqual(ids, [gpt2synth.vocab.get('hello')]);
  });

  ok('BPE: space encodes to Ġ byte token', () => {
    const ids = tok.encode('he t');
    // 'he' merged; 'Ġ t' merged into 'Ġt' by the last merge rule
    assert.deepEqual(ids, [gpt2synth.vocab.get('he'), gpt2synth.vocab.get('Ġt')]);
  });

  ok('BPE round trip: ASCII', () => {
    const s = 'hello world, mini.js is REAL 100%! (test) #42';
    assert.equal(tok.decode(tok.encode(s)), s);
  });

  ok('BPE round trip: unicode + emoji + CJK', () => {
    const s = 'héllo 🌍 wörld — 你好世界 \n\ttab';
    assert.equal(tok.decode(tok.encode(s)), s);
  });

  ok('BPE: special tokens are atomic', () => {
    const ids = tok.encode('hello<|endoftext|>world');
    const eot = gpt2synth.vocab.get('<|endoftext|>');
    assert.ok(ids.includes(eot));
    assert.equal(ids.indexOf(eot), 1); // hello, eot, w,o,r,l,d
    assert.equal(ids.filter(i => i === eot).length, 1);
  });
}

{
  const bj = makeBertFiles();
  const tj = JSON.parse(bj.files['tokenizer.json']);
  const tok = new Tokenizer(tj, JSON.parse(bj.files['tokenizer_config.json']), 'wordpiece');

  ok('WordPiece: lowercase + subwords', () => {
    const ids = tok.encode('The CATS');
    assert.deepEqual(ids, [
      bj.vocab.get('the'),
      bj.vocab.get('cat'), bj.vocab.get('##s'),
    ]);
  });

  ok('WordPiece: unknown word -> [UNK]', () => {
    const ids = tok.encode('xyzzyqqq');
    assert.deepEqual(ids, [bj.vocab.get('[UNK]')]);
  });

  ok('WordPiece: addSpecialTokens wraps with [CLS]/[SEP]', () => {
    const ids = tok.encode('the dog', { addSpecialTokens: true });
    assert.equal(ids[0], bj.vocab.get('[CLS]'));
    assert.equal(ids[ids.length - 1], bj.vocab.get('[SEP]'));
  });

  ok('WordPiece: decode strips ## markers', () => {
    const s = tok.decode(tok.encode('The CATS'));
    assert.equal(s, 'the cats'); // '##s' must join back
    const s2 = tok.decode([bj.vocab.get('cat'), bj.vocab.get('##s')]);
    assert.equal(s2, 'cats');
  });

  ok('WordPiece: punctuation splits', () => {
    const ids = tok.encode('hello, world.');
    assert.ok(ids.length >= 4); // hello , world . = 4 pieces
  });
}

console.log('\n[4] full transformer math (KV-cache consistency)');

await okAsync('GPT-2: cache decode == full recompute', async () => {
  const { config } = makeGPT2Files({ seed: 99, nLayer: 2, nHead: 2, nEmb: 16, maxPos: 32 });
  
  const { GPT2Model } = await import("../src/models.js");
  const files = makeGPT2Files({ seed: 99, nLayer: 2, nHead: 2, nEmb: 16, maxPos: 32 });
  const parsed = parseSafetensors(files.files['model.safetensors']);
  const weights = new Map();
  for (const [name, t] of parsed.tensors) weights.set(name, new mini.Tensor(t.shape, t.data, { backend: 'cpu' }));
  const model = new GPT2Model(weights, files.config);

  const ids = [5, 90, 33, 7, 200, 15, 3];
  const full = mini.arenaRun(() => model.forward(ids, { lastOnly: true })).read();

  const cache = model.makeCache();
  const logits = mini.arenaRun(() => model.forward(ids, { cache, lastOnly: true })).read();
  close(Array.from(full), Array.from(logits), 2e-4);

  // also check non-lastOnly full logits [s, V]
  const fullAll = mini.arenaRun(() => model.forward(ids, {})).read();
  assert.equal(fullAll.length, ids.length * model.vocab);
  close(Array.from(full.subarray(full.length - model.vocab)),
        Array.from(fullAll.slice((ids.length - 1) * model.vocab, ids.length * model.vocab)), 2e-4);
});

await okAsync('BERT: forward runs and is deterministic', async () => {
  const { BertModel } = await import("../src/models.js");
  const files = makeBertFiles();
  const parsed = parseSafetensors(files.files['model.safetensors']);
  const weights = new Map();
  for (const [name, t] of parsed.tensors) weights.set(name, new mini.Tensor(t.shape, t.data, { backend: 'cpu' }));
  const model = new BertModel(weights, files.config);
  const [h1] = mini.arenaRun(() => model.forward([2, 9, 12, 3]));
  const [h2] = mini.arenaRun(() => model.forward([2, 9, 12, 3]));
  close(Array.from(h1.read()), Array.from(h2.read()), 1e-9);
  assert.equal(h1.rows, 4);
});

await okAsync('generation: greedy is deterministic', async () => {
  const files = makeGPT2Files({ seed: 99, nLayer: 2, nHead: 2, nEmb: 16, maxPos: 32 });
  const parsed = parseSafetensors(files.files['model.safetensors']);
  const weights = new Map();
  for (const [name, t] of parsed.tensors) weights.set(name, new mini.Tensor(t.shape, t.data, { backend: 'cpu' }));
  const { GPT2Model } = await import("../src/models.js");
  const model = new GPT2Model(weights, files.config);
  const a = mini.generateTokens(model, [5, 90, 33], { doSample: false, maxNewTokens: 8 });
  const b = mini.generateTokens(model, [5, 90, 33], { doSample: false, maxNewTokens: 8 });
  assert.deepEqual(a, b);
  assert.equal(a.length, 8);
});


// ── regression: lowercase derived from normalizer JSON (not just config) ──
{
  const bj = makeBertFiles();
  const tj = JSON.parse(bj.files['tokenizer.json']);
  const tokNoConfig = new Tokenizer(tj, {}, 'wordpiece'); // NO do_lower_case config
  ok('WordPiece: lowercase derived from normalizer.lowercase', () => {
    const ids = tokNoConfig.encode('The CATS');
    assert.deepEqual(ids, [bj.vocab.get('the'), bj.vocab.get('cat'), bj.vocab.get('##s')]);
  });
}

console.log(`\n${passed} tests passed${process.exitCode ? ' (WITH FAILURES)' : ''}`);

