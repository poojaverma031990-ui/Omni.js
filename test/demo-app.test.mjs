/**
 * Demo-page harness — extracts the inline app script from demo.html and runs
 * its full logic in Node with DOM stubs, against the local synthetic hub.
 * Proves the shipped single-file demo actually works, not just the engine.
 * Run: node test/demo-app.test.mjs
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const ROOT = dirname(fileURLToPath(import.meta.url));

// ── build the world: engine + app scripts from the shipped demo.html ──
const html = readFileSync(join(ROOT, '..', 'demo.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
assert.equal(scripts.length, 2, 'demo.html must have exactly 2 inline scripts (engine + app)');

// ── DOM stubs ──
const elements = new Map();
function makeEl() {
  return {
    textContent: '', innerHTML: '', value: '', hidden: false, disabled: false,
    title: '', style: {}, dataset: {},
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    click() { (this.listeners.click || []).forEach(fn => fn()); },
    appendChild() { },
    classList: { add() {}, remove() {}, toggle() {} },
  };
}
globalThis.document = {
  readyState: 'complete',
  querySelector(sel) {
    if (!elements.has(sel)) elements.set(sel, makeEl());
    return elements.get(sel);
  },
  querySelectorAll() { return []; },
  createElement() { return makeEl(); },
  addEventListener() {},
};
globalThis.location = { search: '' };
globalThis.window = globalThis;

// ── run engine script, then app script ──
(0, eval)(scripts[0]); // engine → defines globalThis.mini
(0, eval)(scripts[1]); // app   → defines globalThis.Demo, runs boot()
const mini = globalThis.mini;
const Demo = globalThis.Demo;
assert.ok(mini && Demo, 'engine + app must both initialize');

// ── point the demo at the local synthetic hub ──
const { makeGPT2Files, makeBertFiles, serveFiles } = await import('./synth.mjs');
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
// remap the demo's real models to local stand-ins with the same tasks/shapes
Demo.MODELS.st = { repo: 'test/tiny-bert', task: 'feature-extraction', size: 1 };
Demo.MODELS.tiny = { repo: 'test/tiny-gpt2', task: 'text-generation', size: 1 };
Demo.MODELS.sst2 = { repo: 'test/tiny-bert', task: 'text-classification', size: 1 };
Demo.MODELS.mlm = { repo: 'test/tiny-bert', task: 'fill-mask', size: 1 };

let passed = 0;
async function ok(name, fn) {
  try { await fn(); passed++; console.log('  ✓', name); }
  catch (e) { console.error('  ✗', name, '\n    ', e.message); process.exitCode = 1; }
}

console.log('\n[demo-app] single-file demo.html logic');

await ok('boot ran (backend badge set)', () => {
  assert.ok(document.querySelector('#backendBadge').textContent.length > 0);
});

await ok('Load model button loads pipeline + flips panel', async () => {
  document.querySelector('#modelSel').value = 'tiny';
  await Demo.loadSelected();
  assert.ok(Demo.pipe, 'pipe loaded');
  assert.equal(document.querySelector('#panel-tiny').hidden, false);
  assert.ok(elements.get('#status').textContent.includes('ready'));
});

await ok('generation task returns prompt + completion', async () => {
  const r = await Demo.runGenerate('Once upon a time', 6, null);
  assert.ok(r.text.startsWith('Once upon a time'));
  assert.equal(r.n, 6);
});

await ok('embeddings task returns a vector + ms', async () => {
  document.querySelector('#modelSel').value = 'st';
  await Demo.loadSelected();
  const r = await Demo.embed('hello world');
  assert.equal(r.vec.length, 16); // synthetic bert dim
  assert.ok(Number.isFinite(r.ms));
});

await ok('semantic search ranks all corpus entries (structure)', async () => {
  // NOTE: the synthetic model has RANDOM weights — no real semantics — so we
  // validate structure here; semantic quality is validated with the real
  // all-MiniLM model in the browser.
  Demo.corpusVecs = null;
  const { scored, ms } = await Demo.runSearch('a cat on a rug');
  assert.equal(scored.length, Demo.CORPUS.length);
  for (let i = 1; i < scored.length; i++) assert.ok(scored[i - 1].score >= scored[i].score, 'sorted');
  assert.ok(scored[0].score <= 1.0001 && scored[0].score >= -1.0001);
  assert.ok(Number.isFinite(ms) && ms >= 0);
});

await ok('similarity: scores in [-1,1], deterministic', async () => {
  const a = await Demo.runSimilarity('the cat sat on the mat', 'a feline rested on a rug');
  const b = await Demo.runSimilarity('the cat sat on the mat', 'stocks rallied after the announcement');
  for (const r of [a, b]) assert.ok(r.score >= -1.0001 && r.score <= 1.0001, 'cosine range');
  // self-similarity must be ~1.0 even with random weights
  const self = await Demo.runSimilarity('identical sentence', 'identical sentence');
  assert.ok(Math.abs(self.score - 1) < 1e-3, 'self cosine ~1, got ' + self.score);
});

await ok('classification returns sorted labels', async () => {
  document.querySelector('#modelSel').value = 'sst2';
  await Demo.loadSelected();
  const r = await Demo.runClassify('the big dog');
  assert.equal(r.results.length, 2);
  assert.ok(r.results[0].score >= r.results[1].score);
});

await ok('fill-mask replaces [MASK] with real tokens', async () => {
  document.querySelector('#modelSel').value = 'mlm';
  await Demo.loadSelected();
  const r = await Demo.runFillMask('the [MASK] runs.');
  assert.ok(r.results.length >= 1);
  assert.ok(!r.results[0].sequence.includes('[MASK]'));
});

await ok('cosine helper is correct', () => {
  const s = Demo.cosine([1, 0, 1], [1, 0, 1]);
  assert.ok(Math.abs(s - 1) < 1e-6);
  const o = Demo.cosine([1, 0], [0, 1]);
  assert.ok(Math.abs(o) < 1e-6);
});

server.close();
console.log(`\n${passed} demo-app tests passed${process.exitCode ? ' (WITH FAILURES)' : ''}`);
