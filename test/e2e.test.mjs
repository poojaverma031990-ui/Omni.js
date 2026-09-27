/**
 * mini.js end-to-end test — loads models through the REAL pipeline stack
 * (downloader -> safetensors -> tokenizer -> model -> generate) against a
 * local HTTP server that mimics the Hugging Face Hub layout.
 * Run: node test/e2e.test.mjs
 */
import assert from 'node:assert/strict';
import { makeGPT2Files, makeBertFiles, serveFiles } from './synth.mjs';
import '../mini.js';
const mini = globalThis.mini;

let passed = 0;
async function ok(name, fn) {
  try { await fn(); passed++; console.log('  ✓', name); }
  catch (e) { console.error('  ✗', name, '\n    ', e.stack ? e.stack.split('\n').slice(0, 3).join('\n     ') : e); process.exitCode = 1; }
}

console.log('\n[e2e] full pipeline through HTTP (hub-like)');

const gpt2 = makeGPT2Files({ seed: 11, nLayer: 2, nHead: 2, nEmb: 16, maxPos: 32 });
const bert = makeBertFiles({ seed: 12 });
const server = await serveFiles({
  'test/tiny-gpt2/config.json': gpt2.files['config.json'],
  'test/tiny-gpt2/tokenizer.json': gpt2.files['tokenizer.json'],
  'test/tiny-gpt2/tokenizer_config.json': gpt2.files['tokenizer_config.json'],
  'test/tiny-gpt2/model.safetensors': gpt2.files['model.safetensors'],
  'test/tiny-bert/config.json': bert.files['config.json'],
  'test/tiny-bert/tokenizer.json': bert.files['tokenizer.json'],
  'test/tiny-bert/tokenizer_config.json': bert.files['tokenizer_config.json'],
  'test/tiny-bert/model.safetensors': bert.files['model.safetensors'],
});

mini.env.hfBase = server.url;
mini.env.useBrowserCache = false;

await ok('pipeline(text-generation) downloads + generates', async () => {
  const events = [];
  const gen = await mini.pipeline('text-generation', 'test/tiny-gpt2', {
    progress_callback: (e) => events.push(e.status),
  });
  const [out] = await gen('hello', { do_sample: false, max_new_tokens: 6 });
  assert.equal(typeof out.generated_text, 'string');
  assert.ok(out.generated_text.startsWith('hello'), 'starts with prompt: ' + out.generated_text);
  assert.ok(events.includes('initiate') && events.includes('progress') && events.includes('ready'),
    'progress events: ' + events.join(','));
  // deterministic
  const [out2] = await gen('hello', { do_sample: false, max_new_tokens: 6 });
  assert.equal(out.generated_text, out2.generated_text);
});

await ok('pipeline(feature-extraction) produces normalized embeddings', async () => {
  const fx = await mini.pipeline('feature-extraction', 'test/tiny-bert');
  const out = await fx('the cat runs', { pooling: 'mean', normalize: true });
  assert.equal(out.dims[0], 16);
  let norm = 0;
  for (const v of out.data) norm += v * v;
  assert.ok(Math.abs(norm - 1) < 1e-4, 'unit norm, got ' + norm);
  const clsOut = await fx('the dog', { pooling: 'cls', normalize: false });
  assert.equal(clsOut.data.length, 16);
});

await ok('pipeline(fill-mask) predicts [MASK] tokens', async () => {
  const fm = await mini.pipeline('fill-mask', 'test/tiny-bert');
  const results = await fm('the [MASK] runs.', { top_k: 3 });
  assert.equal(results.length, 3);
  for (const r of results) {
    assert.ok(r.score > 0 && r.score <= 1.0001);
    assert.ok(typeof r.token === 'string');
    assert.ok(!r.sequence.includes('[MASK]'), 'mask replaced: ' + r.sequence);
    assert.ok(r.sequence.startsWith('the') || r.sequence.startsWith('the '));
  }
  assert.ok(results[0].score >= results[1].score, 'sorted by score');
});

await ok('pipeline(text-classification) returns labels', async () => {
  const cls = await mini.pipeline('text-classification', 'test/tiny-bert');
  const out = await cls('the big dog');
  assert.equal(out.length, 2);
  assert.ok(out[0].label === 'NEGATIVE' || out[0].label === 'POSITIVE');
  assert.ok(out[0].score >= out[1].score);
  assert.ok(Math.abs(out[0].score + out[1].score - 1) < 1e-3);
});

await ok('one-line API: mini.generate / mini.embed / mini.classify / mini.fillMask accept custom models', async () => {
  const text = await mini.generate('hello', { model: 'test/tiny-gpt2', doSample: false, maxNewTokens: 4 });
  assert.ok(text.startsWith('hello'));
  const vec = await mini.embed('the cat', { model: 'test/tiny-bert' });
  assert.equal(vec.length, 16);
  const cls = await mini.classify('the dog', { model: 'test/tiny-bert' });
  assert.equal(cls.length, 2);
  const fm = await mini.fillMask('the [MASK] runs', { model: 'test/tiny-bert' });
  assert.ok(fm.length >= 1);
});

await ok('sampling params: temperature/top-k run without error', async () => {
  const gen = await mini.pipeline('text-generation', 'test/tiny-gpt2');
  const [out] = await gen('hello', { do_sample: true, temperature: 0.7, top_k: 5, top_p: 0.9, max_new_tokens: 5 });
  assert.ok(typeof out.generated_text === 'string' && out.generated_text.length >= 5);
});

await ok('unknown task / repo fail with clear errors', async () => {
  await assert.rejects(() => mini.pipeline('quantum-foo', 'test/tiny-gpt2'), /unknown task/);
  await assert.rejects(() => mini.pipeline('text-generation', 'test/does-not-exist'), /failed to download|no safetensors/);
});

await ok('onToken streaming callback fires per token', async () => {
  const gen = await mini.pipeline('text-generation', 'test/tiny-gpt2');
  const seen = [];
  await gen('hello', { do_sample: false, max_new_tokens: 4, onToken: (id, n) => seen.push([id, n]) });
  assert.equal(seen.length, 4);
  assert.deepEqual(seen.map(s => s[1]), [1, 2, 3, 4]);
});

server.close();
console.log(`\n${passed} e2e tests passed${process.exitCode ? ' (WITH FAILURES)' : ''}`);
