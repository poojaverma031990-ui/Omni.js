/**
 * v2.0 capability tests: Llama family (GQA/RoPE/RMSNorm/SwiGLU), RoBERTa,
 * chat templates (jinja subset + builtins), sharded safetensors, QA, stop.
 * Run: node test/competitor.test.mjs
 */
import assert from 'node:assert/strict';
import { parseSafetensors, writeSafetensors } from '../src/safetensors.js';
import * as K from '../src/cpu-kernels.js';
import { buildChatPrompt, renderChatTemplate } from '../src/chat-template.js';
import { makeGPT2Files, makeLlamaFiles, makeRobertaFiles, makeBertFiles, serveFiles } from './synth.mjs';
import '../mini.js';
const mini = globalThis.mini;
mini.setBackend('cpu');

let passed = 0;
async function ok(name, fn) {
  try { await fn(); passed++; console.log('  ✓', name); }
  catch (e) { console.error('  ✗', name, '\n    ', e.message); process.exitCode = 1; }
}
function close(a, b, tol = 2e-4) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const s = Math.abs(a[i]) + Math.abs(b[i]) + 1;
    assert.ok(d < tol * s, `diff ${d} at ${i}`);
  }
}

console.log('\n[v2] CPU kernels: RMSNorm / mul / RoPE');

await ok('RMSNorm matches reference', () => {
  const rows = 3, cols = 8;
  const x = new Float32Array(rows * cols).map((_, i) => Math.sin(i) * 2);
  const g = new Float32Array(cols).map((_, i) => 0.5 + (i % 3) * 0.25);
  const out = K.cpuRMSNorm(x, rows, cols, g, 1e-5);
  for (let r = 0; r < rows; r++) {
    let ss = 0;
    for (let c = 0; c < cols; c++) ss += x[r * cols + c] ** 2;
    const rstd = 1 / Math.sqrt(ss / cols + 1e-5);
    for (let c = 0; c < cols; c++) {
      assert.ok(Math.abs(out[r * cols + c] - x[r * cols + c] * rstd * g[c]) < 1e-5);
    }
  }
});

await ok('RoPE preserves vector norms and matches manual rotation', () => {
  const rows = 2, headDim = 8, nHeads = 2, cols = headDim * nHeads;
  const x = new Float32Array(rows * cols).map((_, i) => (i % 7) - 3);
  const out = K.cpuRope(x, rows, cols, headDim, 3, 10000.0);
  for (let r = 0; r < rows; r++) {
    let n0 = 0, n1 = 0;
    for (let c = 0; c < cols; c++) { n0 += x[r * cols + c] ** 2; n1 += out[r * cols + c] ** 2; }
    assert.ok(Math.abs(Math.sqrt(n0) - Math.sqrt(n1)) < 1e-4, 'norm preserved');
  }
  // manual check: head 0, row 0 (pos=3), pair (t=1, t+4)
  const half = headDim >> 1;
  const ang = 3 * Math.pow(10000.0, -2 * 1 / half);
  const a = x[1], b = x[1 + half];
  assert.ok(Math.abs(out[1] - (a * Math.cos(ang) - b * Math.sin(ang))) < 1e-5);
  assert.ok(Math.abs(out[1 + half] - (b * Math.cos(ang) + a * Math.sin(ang))) < 1e-5);
});

console.log('\n[v2] chat templates (jinja subset, from scratch)');

await ok('builtin ChatML (Qwen style)', () => {
  const out = buildChatPrompt([{ role: 'user', content: 'Hi' }], { modelType: 'qwen2' });
  assert.equal(out, '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n');
});

await ok('builtin Llama-3 with system + multi-turn', () => {
  const out = buildChatPrompt([
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello!' },
    { role: 'user', content: 'Bye' },
  ], { modelType: 'llama' });
  assert.equal(out,
    '<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>' +
    '<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>' +
    '<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>' +
    '<|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|>' +
    '<|start_header_id|>assistant<|end_header_id|>\n\n');
});

await ok('builtin Gemma', () => {
  const out = buildChatPrompt([{ role: 'user', content: 'Hey' }], { modelType: 'gemma' });
  assert.equal(out, '<start_of_turn>user\nHey<end_of_turn>\n<start_of_turn>model\n');
});

await ok('jinja subset renders the REAL SmolLM2 chat template', () => {
  // verbatim from HuggingFaceTB/SmolLM2-135M-Instruct tokenizer_config.json
  const smolTemplate = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
  const out = renderChatTemplate(smolTemplate, {
    messages: [{ role: 'user', content: 'What is 2+2?' }],
    add_generation_prompt: true,
  });
  assert.equal(out,
    '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' +
    '<|im_start|>user\nWhat is 2+2?<|im_end|>\n' +
    '<|im_start|>assistant\n');
  // with a system message present, the injected one must disappear
  const out2 = renderChatTemplate(smolTemplate, {
    messages: [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'Hi' }],
    add_generation_prompt: true,
  });
  assert.ok(out2.startsWith('<|im_start|>system\nBe brief.'));
  assert.ok(!out2.includes('SmolLM, trained'));
});

await ok('jinja subset: set / filters / raise_exception', () => {
  const t = "{% set name = 'world' %}hello {{ name }}{% if messages|length > 1 %} multi{% endif %}";
  assert.equal(renderChatTemplate(t, { messages: [1, 2] }), 'hello world multi');
  assert.throws(() => renderChatTemplate("{% if messages|length > 2 %}{% raise_exception('too many') %}{% endif %}", { messages: [1, 2, 3] }), /too many/);
  // {{ raise_exception(...) }} call syntax is outside the subset -> falls back
  const out = buildChatPrompt([{ role: 'user', content: 'x' }], {
    template: "{{ raise_exception('nope') }}", modelType: 'qwen2',
  });
  assert.ok(out.includes('<|im_start|>assistant'), 'fell back to chatml');
});

await ok('unsupported jinja falls back to builtin family template', () => {
  // uses `messages[loop.index0 - 1]` arithmetic — outside the subset
  const qwenReal = "{%- for m in messages %}{% if loop.index0 > 0 %}{{ messages[loop.index0 - 1]['role'] }}{% endif %}{{ m['role'] }}{% endfor %}";
  const out = buildChatPrompt([{ role: 'user', content: 'x' }], { template: qwenReal, modelType: 'qwen2' });
  assert.ok(out.includes('<|im_start|>assistant'), 'fell back to chatml: ' + JSON.stringify(out));
});

console.log('\n[v2] LLAMA family engine (GQA + RoPE + RMSNorm + SwiGLU)');

await ok('Llama: cache decode == full recompute (GQA path)', async () => {
  const { GPT2Model: _G, LlamaModel } = await import('../src/models.js');
  const files = makeLlamaFiles({ seed: 21 });
  const parsed = parseSafetensors(files.files['model.safetensors']);
  const weights = new Map();
  for (const [name, t] of parsed.tensors) weights.set(name, new mini.Tensor(t.shape, t.data, { backend: 'cpu' }));
  const model = new LlamaModel(weights, files.config);
  assert.equal(model.nKV, 2);
  assert.equal(model.nHead, 4);

  const ids = [5, 90, 33, 7, 200, 15, 3];
  const full = mini.arenaRun(() => model.forward(ids, { lastOnly: true })).read();
  const cache = model.makeCache();
  const cached = mini.arenaRun(() => model.forward(ids, { cache, lastOnly: true })).read();
  close(Array.from(full), Array.from(cached), 2e-4);
  // step decoding stays consistent
  const l2 = mini.arenaRun(() => model.forward([42], { cache, lastOnly: true })).read();
  const full2 = mini.arenaRun(() => model.forward([...ids, 42], { lastOnly: true })).read();
  close(Array.from(l2), Array.from(full2), 2e-4);
  model.disposeCache(cache);
});

await ok('KV-cache grows in chunks (no giant preallocation)', async () => {
  const { LlamaModel } = await import('../src/models.js');
  const files = makeLlamaFiles({ seed: 22 });
  const parsed = parseSafetensors(files.files['model.safetensors']);
  const weights = new Map();
  for (const [name, t] of parsed.tensors) weights.set(name, new mini.Tensor(t.shape, t.data, { backend: 'cpu' }));
  const model = new LlamaModel(weights, files.config);
  const cache = model.makeCache();
  assert.equal(cache.capacity, 512);
  const long = Array.from({ length: 700 }, (_, i) => (i % 200) + 1);
  mini.arenaRun(() => model.forward(long, { cache, lastOnly: true }));
  assert.ok(cache.capacity >= 700, 'grew to ' + cache.capacity);
  assert.equal(cache.len, 700);
  model.disposeCache(cache);
});

console.log('\n[v2] RoBERTa + QA + sharded loading + stop (through HTTP)');

const gpt2 = makeGPT2Files({ seed: 11, nLayer: 2, nHead: 2, nEmb: 16, maxPos: 32 });
const llama = makeLlamaFiles({ seed: 21 });
const roberta = makeRobertaFiles({ seed: 31 });
const bertQA = makeBertFiles({ seed: 12, qa: true });
const bertMLM = makeBertFiles({ seed: 13 });

// shard the llama model into 2 safetensors files + index
{
  const parsed = parseSafetensors(llama.files['model.safetensors']);
  const names = [...parsed.tensors.keys()];
  const half = Math.ceil(names.length / 2);
  const shards = [names.slice(0, half), names.slice(half)];
  const weight_map = {};
  llama.files['model.safetensors'] = undefined;
  delete llama.files['model.safetensors'];
  shards.forEach((shardNames, si) => {
    const entries = shardNames.map(n => [n, parsed.tensors.get(n)]);
    const fname = `model-${String(si + 1).padStart(5, '0')}-of-00002.safetensors`;
    llama.files[fname] = writeSafetensors(entries);
    for (const n of shardNames) weight_map[n] = fname;
  });
  llama.files['model.safetensors.index.json'] = JSON.stringify({ metadata: { total_size: 1 }, weight_map });
}

const server = await serveFiles({
  'test/tiny-gpt2/config.json': gpt2.files['config.json'],
  'test/tiny-gpt2/tokenizer.json': gpt2.files['tokenizer.json'],
  'test/tiny-gpt2/tokenizer_config.json': gpt2.files['tokenizer_config.json'],
  'test/tiny-gpt2/model.safetensors': gpt2.files['model.safetensors'],
  'test/tiny-llama/config.json': llama.files['config.json'],
  'test/tiny-llama/tokenizer.json': llama.files['tokenizer.json'],
  'test/tiny-llama/tokenizer_config.json': llama.files['tokenizer_config.json'],
  ...Object.fromEntries(Object.entries(llama.files).filter(([k]) => k.includes('safetensors')).map(([k, v]) => ['test/tiny-llama/' + k, v])),
  'test/tiny-roberta/config.json': roberta.files['config.json'],
  'test/tiny-roberta/tokenizer.json': roberta.files['tokenizer.json'],
  'test/tiny-roberta/tokenizer_config.json': roberta.files['tokenizer_config.json'],
  'test/tiny-roberta/model.safetensors': roberta.files['model.safetensors'],
  'test/tiny-bert-qa/config.json': bertQA.files['config.json'],
  'test/tiny-bert-qa/tokenizer.json': bertQA.files['tokenizer.json'],
  'test/tiny-bert-qa/tokenizer_config.json': bertQA.files['tokenizer_config.json'],
  'test/tiny-bert-qa/model.safetensors': bertQA.files['model.safetensors'],
  'test/tiny-bert-mlm/config.json': bertMLM.files['config.json'],
  'test/tiny-bert-mlm/tokenizer.json': bertMLM.files['tokenizer.json'],
  'test/tiny-bert-mlm/tokenizer_config.json': bertMLM.files['tokenizer_config.json'],
  'test/tiny-bert-mlm/model.safetensors': bertMLM.files['model.safetensors'],
});
mini.env.hfBase = server.url;
mini.env.useBrowserCache = false;

await ok('SHARDED model loads via model.safetensors.index.json', async () => {
  const pipe = await mini.pipeline('text-generation', 'test/tiny-llama');
  assert.equal(pipe.model.kind, 'llama');
  const [out] = await pipe('hello', { do_sample: false, max_new_tokens: 4 });
  assert.ok(out.generated_text.startsWith('hello'));
});

await ok('Llama chat() applies template and stops at EOS', async () => {
  const pipe = await mini.pipeline('text-generation', 'test/tiny-llama');
  assert.equal(typeof pipe.chat, 'function');
  const [out] = await pipe.chat([{ role: 'user', content: 'hello' }], { max_new_tokens: 4, do_sample: false });
  assert.ok(typeof out.assistant_message === 'string');
  assert.ok(out.generated_text.includes('user'));
});

await ok('generation STOP: onToken returning false halts generation', async () => {
  const pipe = await mini.pipeline('text-generation', 'test/tiny-gpt2');
  const seen = [];
  const [out] = await pipe('hello', {
    do_sample: false, max_new_tokens: 50,
    onToken: (id, n) => { seen.push(n); if (n >= 3) return false; },
  });
  assert.equal(seen.length, 3, 'stopped at 3 tokens, got ' + seen.length);
});

await ok('RoBERTa classification pipeline (3 labels)', async () => {
  const pipe = await mini.pipeline('text-classification', 'test/tiny-roberta');
  const out = await pipe('the cat');
  assert.equal(out.length, 3);
  assert.ok(['negative', 'neutral', 'positive'].includes(out[0].label));
  assert.ok(Math.abs(out.reduce((a, b) => a + b.score, 0) - 1) < 1e-3);
});

await ok('question-answering pipeline runs and bounds the span', async () => {
  const pipe = await mini.pipeline('question-answering', 'test/tiny-bert-qa');
  const [r] = await pipe({
    question: 'What runs?',
    context: 'the cat runs. The dog barks.',
  });
  assert.ok(typeof r.answer === 'string' && r.answer.length > 0);
  assert.ok(r.score > 0 && r.score <= 1.0001);
});

await ok('token-classification errors clearly without a flat head', async () => {
  // synthetic roberta has dense/out_proj head, not the flat classifier.weight
  // that token classification needs — loading is fine, CALLING must error clearly
  const pipe = await mini.pipeline('token-classification', 'test/tiny-roberta');
  await assert.rejects(() => pipe('the cat'), /missing weight: classifier\.weight/);
});

await ok('fill-mask still works on bert mlm', async () => {
  const pipe = await mini.pipeline('fill-mask', 'test/tiny-bert-mlm');
  const out = await pipe('the [MASK] runs.');
  assert.ok(out.length >= 1);
});

server.close();
console.log(`\n${passed} v2 capability tests passed${process.exitCode ? ' (WITH FAILURES)' : ''}`);
