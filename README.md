<div align="center">

# ⚡ mini.js

### Run real AI models in your browser. One line of code.

**A Transformers.js alternative written 100% from scratch in pure HTML + CSS + JavaScript.**
Zero dependencies · No ONNX Runtime · No TensorFlow.js · No WebAssembly blobs · No native code.

`const text = await mini.generate('Once upon a time');`

</div>

---

## Why

Transformers.js is great — but it wraps ONNX Runtime (a giant C++ WebAssembly blob) and hundreds of
packages. **mini.js proves you don't need any of that.** Every layer of the stack is hand-written in
plain JavaScript and runs anywhere a browser runs — one `<script>` tag, anywhere in the world:

```html
<script src="mini.js"></script>
<script>
  const gen = await mini.pipeline('text-generation', 'openai-community/gpt2');
  const [out] = await gen('Once upon a time', { max_new_tokens: 60 });
  console.log(out.generated_text);
</script>
```

## What's inside (all from scratch)

| Layer | What was built |
|---|---|
| **Tensor engine** | Eager tensor class with two full backends: blocked/unrolled `Float32Array` kernels (CPU) and WebGL2 kernels (GPU) |
| **GPU kernels** | Hand-written GLSL ES 3.00: matmul (NN / transposed-B / transposed-storage), softmax, layernorm, GELU, ReLU, SiLU, sigmoid, tanh, embedding gathers, slice/scatter, mask add |
| **Tokenizers** | Byte-level BPE (GPT-2 byte↔unicode map, merge ranks, pre-tokenizer regex) and WordPiece (greedy longest-match, `##` prefixes, CJK/punct splitting) — reading real `tokenizer.json` |
| **safetensors parser** | Reads the binary format directly: JSON header + raw buffers, F32 / F16 / BF16 decoding via bit manipulation |
| **Models** | GPT-2 decoder (KV-cache, causal attention, tied/untied LM head), BERT and DistilBERT encoders + classification / MLM heads |
| **Generation** | Greedy, temperature, top-k, top-p (nucleus), repetition penalty, EOS stopping, streaming `onToken` callback |
| **Hub client** | Streams files from huggingface.co with progress callbacks, caches in the browser (Cache API) → instant & offline after first run |
| **GPU self-test** | Every WebGL kernel is verified against the CPU reference at load; mini.js silently falls back to CPU if anything disagrees |

~110 KB of pure JavaScript. No build step needed to *use* it (`mini.js` is a ready-made single file).

## Quick start (3 ways)

**1. Script tag (any website, any browser)**

```html
<script src="mini.js"></script>
<script>
  const [top] = await mini.classify('I love this product!');
  // → { label: 'POSITIVE', score: 0.999… }
</script>
```

Or straight from a CDN once merged to `main` (no install at all):

```html
<script src="https://cdn.jsdelivr.net/gh/poojaverma031990-ui/Omni.js@main/mini.js"></script>
```

**2. One-line convenience APIs**

```js
const text  = await mini.generate('Once upon a time');        // GPT-2
const vec   = await mini.embed('Hello world');                // 384-dim vector
const sent  = await mini.classify('What a day!');             // sentiment
const hole  = await mini.fillMask('Paris is the [MASK] of France.');
```

**3. Full control (Transformers.js-style pipelines)**

```js
const gen = await mini.pipeline('text-generation', 'openai-community/gpt2', {
  progress_callback: (p) => console.log(p.status, p.file, Math.round(p.progress * 100) + '%'),
});

const [out] = await gen('The meaning of life is', {
  max_new_tokens: 100,
  do_sample: true,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
  onToken: (tokenId, n) => process.stdout.write('*'),
});
```

## API

### `mini.pipeline(task, repoId, options?) → Promise<pipeline>`

Tasks: `text-generation` · `text-classification` · `feature-extraction` · `fill-mask`

The returned pipeline is a **callable function** (plus `.model`, `.tokenizer`, `.config`, `.dispose()`):

```js
const pipe = await mini.pipeline('text-generation', 'openai-community/gpt2');
const [out] = await pipe('hello', { max_new_tokens: 20 });
// out.generated_text — full text (prompt + completion)
```

| Pipeline | Call signature | Returns |
|---|---|---|
| `text-generation` | `(text, {max_new_tokens, do_sample, temperature, top_k, top_p, repetition_penalty, onToken})` | `[{ generated_text }]` |
| `text-classification` | `(text)` | `[{label, score}, …]` sorted |
| `feature-extraction` | `(text, {pooling:'mean'\|'cls', normalize})` | `{ data: Float32Array, dims }` |
| `fill-mask` | `(text, {top_k})` — must contain `[MASK]` | `[{token, score, sequence}]` |

### Other top-level functions

```js
mini.initBackend()      // pick WebGL2 (after running the GPU self-test) or CPU
mini.setBackend('cpu')  // force a backend
mini.selfTest()         // { backend, ok, failed, checks } — verify kernel numerics
mini.clearCache()       // delete all cached model files
mini.env.hfBase         // override the hub base URL (self-hosting / proxies)
mini.Tensor             // the raw tensor engine
mini.Tokenizer          // BPE + WordPiece tokenizers
mini.parseSafetensors() // the binary parser
```

### Models that work today

Any repo with `model.safetensors` + `tokenizer.json` from these families:

| Model | Task | Download |
|---|---|---|
| `openai-community/gpt2` | text-generation | ~548 MB |
| `distilbert/distilgpt2` | text-generation | ~355 MB |
| `hf-internal-testing/tiny-random-gpt2` | text-generation (smoke test) | <1 MB |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | text-classification | ~268 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | feature-extraction | ~90 MB |
| `distilbert/distilbert-base-uncased` | fill-mask | ~268 MB |
| `google-bert/bert-base-uncased` | fill-mask / embeddings | ~440 MB |

> 💡 Models are downloaded **once**, stored in the browser Cache API, and then load instantly and work offline.

## One-file evaluation demo

**`demo.html`** is a single self-contained HTML file (the whole engine is inlined).
Open it in any browser — no server, no install, works from `file://` — pick a model
and press Load. It ships with:

- `sentence-transformers/all-MiniLM-L6-v2` (~91 MB) — embeddings, cosine similarity, semantic search
- `hf-internal-testing/tiny-random-gpt2` (<1 MB) — instant text-generation engine test
- `distilbert/…-sst-2-english` (~268 MB) — sentiment
- `distilbert/distilbert-base-uncased` (~268 MB) — fill-mask

To edit the demo, change `demo.template.html` and run `npm run build` (it re-inlines the engine).
Append `?hub=<url>` to point the page at a different hub (self-hosting / testing).

## Run the demo site locally

```bash
npm run serve        # → http://localhost:8000  (pure Node, no deps)
```

Then open the playground, pick a model, press **Run**.

## Development

```bash
npm run build        # bundles src/*.js into the single-file mini.js (custom bundler, no deps)
npm test             # 22 unit tests + 8 end-to-end tests (no network needed)
```

The test-suite builds **synthetic models in exact Hugging Face format**, serves them over HTTP,
and runs the complete stack — downloader → safetensors → tokenizer → transformer → generated text —
including a proof that KV-cache decoding is numerically identical to full recomputation.

## How the GPU backend works

Tensors live in single `RGBA32F` textures (4 floats per texel, row stride padded to 4).
Every kernel writes zeros into padding, so all math stays exact. Kernels are raw fragment
shaders over a fullscreen triangle — no compute shaders, no extensions beyond
`EXT_color_buffer_float`. Weights too tall for a texture (GPT-2's 50257×768 embedding) are
stored **transposed** at load time and used with dedicated `matmulT` / `gatherT` kernels.
Generation uses a preallocated KV-cache per layer; single-token decode steps never recompute
the prompt.

## Roadmap

- [ ] Quantized (Q8/Q4) weight loading for 4–8× smaller downloads
- [ ] WebGPU backend
- [ ] Llama / RoPE architectures (RMSNorm, SwiGLU, GQA)
- [ ] Encoder–decoder (T5-style) models
- [ ] Whisper speech recognition
- [ ] Batching + padding for multi-sequence inputs

## License

MIT — see [LICENSE](LICENSE).
