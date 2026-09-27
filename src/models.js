/**
 * mini.js — Transformer architectures implemented from scratch with Tensors:
 *   - GPT2 (decoder LM, KV-cache, causal attention, tied/untied lm head)
 *   - BERT (bidirectional encoder + MLM/classification heads)
 *   - DistilBERT (6-layer encoder + heads)
 * All forward passes are explicit sequences of matmul/softmax/layernorm ops.
 */
import { Tensor } from './tensor.js';

function arange(n, start = 0) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = start + i;
  return a;
}

/** Look up a weight with a helpful error. */
function W(weights, name) {
  const t = weights.get(name);
  if (!t) throw new Error('[mini.js] missing weight: ' + name);
  return t;
}

class KVCache {
  constructor(model, capacity) {
    this.capacity = capacity;
    this.len = 0;
    this.layers = [];
    for (let i = 0; i < model.nLayer; i++) {
      this.layers.push({
        k: Tensor.zeros([capacity, model.dim]),
        v: Tensor.zeros([capacity, model.dim]),
      });
    }
  }
  dispose() { for (const l of this.layers) { l.k.dispose(); l.v.dispose(); } }
}

// ---------------------------------------------------------------------------
// GPT-2 family (GPT-2, DistilGPT-2, TinyGPT-2 …)
// ---------------------------------------------------------------------------
export class GPT2Model {
  constructor(weights, config) {
    this.kind = 'gpt2';
    this.wte = W(weights, 'transformer.wte.weight');
    this.wpe = W(weights, 'transformer.wpe.weight');
    this.vocab = this.wte.shape[0];
    this.dim = config.n_embd || this.wte.shape[1];
    this.maxPos = this.wpe.shape[0];
    this.nHead = config.n_head || 4;
    this.headDim = Math.floor(this.dim / this.nHead);
    this.eps = config.layer_norm_epsilon ?? config.layer_norm_eps ?? 1e-5;

    const nLayer = config.n_layer;
    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      const p = `transformer.h.${i}`;
      this.layers.push({
        ln1w: W(weights, `${p}.ln_1.weight`), ln1b: W(weights, `${p}.ln_1.bias`),
        attnW: W(weights, `${p}.attn.c_attn.weight`), attnB: W(weights, `${p}.attn.c_attn.bias`),
        projW: W(weights, `${p}.attn.c_proj.weight`), projB: W(weights, `${p}.attn.c_proj.bias`),
        ln2w: W(weights, `${p}.ln_2.weight`), ln2b: W(weights, `${p}.ln_2.bias`),
        fcW: W(weights, `${p}.mlp.c_fc.weight`), fcB: W(weights, `${p}.mlp.c_fc.bias`),
        mlpW: W(weights, `${p}.mlp.c_proj.weight`), mlpB: W(weights, `${p}.mlp.c_proj.bias`),
      });
    }
    this.lnfw = W(weights, 'transformer.ln_f.weight');
    this.lnfb = W(weights, 'transformer.ln_f.bias');
    // lm head: prefer explicit, fall back to tied wte
    const lm = weights.get('lm_head.weight');
    this.lmHead = lm || this.wte; // [V, d] or transposedStorage [d, V]
    this.eosTokenId = config.eos_token_id ?? config.bos_token_id ?? undefined;
    this.nLayer = nLayer;
  }

  makeCache(capacity) { return new KVCache(this, capacity || this.maxPos); }

  /** Returns logits [s, V] (lastOnly -> [1, V]). */
  forward(ids, { cache = null, lastOnly = false } = {}) {
    const s = ids.length;
    const posStart = cache ? cache.len : 0;
    let x = this.wte.gather(ids);
    const pe = this.wpe.gather(arange(s, posStart));
    x = x.add(pe);

    const t0 = cache ? cache.len : 0;
    const total0 = t0 + s;
    const causal = s > 1 || cache === null; // prefill needs a causal mask; single-token decode doesn't
    let mask = null;
    if (causal) {
      const data = new Float32Array(s * total0);
      for (let r = 0; r < s; r++) {
        for (let c = 0; c < total0; c++) {
          if (t0 + r < c) data[r * total0 + c] = -1e9;
        }
      }
      mask = Tensor.fromArray(data, [s, total0]);
    }

    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      let h = x.layernorm(L.ln1w, L.ln1b, this.eps);
      const qkv = h.matmul(L.attnW).addRowBias(L.attnB); // [s, 3d]
      const q = qkv.sliceCols(0, this.dim);
      const k = qkv.sliceCols(this.dim, this.dim);
      const v = qkv.sliceCols(this.dim * 2, this.dim);

      const t = t0;
      const total = total0;
      if (cache) {
        if (s === 1) {
          cache.layers[li].k.scatterRow(k, t);
          cache.layers[li].v.scatterRow(v, t);
        } else {
          cache.layers[li].k.pasteRows(k, t);
          cache.layers[li].v.pasteRows(v, t);
        }
        cache.len = total;
      }
      const kAll = cache ? cache.layers[li].k : k; // [total, d]
      const vAll = cache ? cache.layers[li].v : v;

      const attnOut = Tensor.zeros([s, this.dim]);
      attnOut.clear();
      for (let hIdx = 0; hIdx < this.nHead; hIdx++) {
        const off = hIdx * this.headDim;
        const qh = q.sliceCols(off, this.headDim);            // [s, hd]
        const kh = kAll.sliceCols(off, this.headDim);          // [total, hd]
        const vh = vAll.sliceCols(off, this.headDim);          // [total, hd]
        let scores = qh.matmul(kh, { transB: true });          // [s, total]
        if (causal && mask) scores = scores.add(mask);
        const probs = scores.softmax(1 / Math.sqrt(this.headDim));
        const ctx = probs.matmul(vh);                          // [s, hd]
        attnOut.scatterCols(ctx, off, this.headDim);
      }
      let o = attnOut.matmul(L.projW).addRowBias(L.projB);
      x = x.add(o);

      let m = x.layernorm(L.ln2w, L.ln2b, this.eps);
      m = m.matmul(L.fcW).addRowBias(L.fcB).gelu();
      m = m.matmul(L.mlpW).addRowBias(L.mlpB);
      x = x.add(m);
    }
    if (mask) mask.dispose();

    x = x.layernorm(this.lnfw, this.lnfb, this.eps);
    let finalX = x;
    if (lastOnly && s > 1) finalX = x.rowSlice(s - 1);
    return finalX.matmul(this.lmHead, {
      transB: !this.lmHead.transposedStorage,
      outShape: lastOnly ? [1, this.vocab] : null,
    });
  }

  disposeCache(cache) { if (cache) cache.dispose(); }
}

// ---------------------------------------------------------------------------
// BERT (bidirectional encoder)
// ---------------------------------------------------------------------------
export class BertModel {
  constructor(weights, config) {
    this.kind = 'bert';
    this.word = W(weights, 'bert.embeddings.word_embeddings.weight');
    this.pos = W(weights, 'bert.embeddings.position_embeddings.weight');
    this.ttype = W(weights, 'bert.embeddings.token_type_embeddings.weight');
    this.lnw = W(weights, 'bert.embeddings.LayerNorm.weight');
    this.lnb = W(weights, 'bert.embeddings.LayerNorm.bias');
    this.vocab = this.word.shape[0];
    this.dim = this.word.shape[1];
    this.maxPos = this.pos.shape[0];
    this.nHead = config.num_attention_heads || 4;
    this.headDim = Math.floor(this.dim / this.nHead);
    this.eps = config.layer_norm_eps ?? config.layer_norm_epsilon ?? 1e-12;

    const nLayer = config.num_hidden_layers;
    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      const p = `bert.encoder.layer.${i}`;
      this.layers.push({
        qw: W(weights, `${p}.attention.self.query.weight`), qb: W(weights, `${p}.attention.self.query.bias`),
        kw: W(weights, `${p}.attention.self.key.weight`), kb: W(weights, `${p}.attention.self.key.bias`),
        vw: W(weights, `${p}.attention.self.value.weight`), vb: W(weights, `${p}.attention.self.value.bias`),
        ow: W(weights, `${p}.attention.output.dense.weight`), ob: W(weights, `${p}.attention.output.dense.bias`),
        olw: W(weights, `${p}.attention.output.LayerNorm.weight`), olb: W(weights, `${p}.attention.output.LayerNorm.bias`),
        i1w: W(weights, `${p}.intermediate.dense.weight`), i1b: W(weights, `${p}.intermediate.dense.bias`),
        i2w: W(weights, `${p}.output.dense.weight`), i2b: W(weights, `${p}.output.dense.bias`),
        o2w: W(weights, `${p}.output.LayerNorm.weight`), o2b: W(weights, `${p}.output.LayerNorm.bias`),
      });
    }
    this.hasPooler = weights.has('bert.pooler.dense.weight');
    if (this.hasPooler) {
      this.pw = W(weights, 'bert.pooler.dense.weight');
      this.pb = W(weights, 'bert.pooler.dense.bias');
    }
    this.nLayer = nLayer;
  }

  /** Returns [hidden [s,d], pooled [1,d]|null] (pooled = tanh(W·h[CLS])). */
  forward(ids, { tokenTypeIds = null, pooled = false } = {}) {
    const s = ids.length;
    let x = this.word.gather(ids)
      .add(this.pos.gather(arange(s)))
      .add(this.ttype.gather(tokenTypeIds || new Array(s).fill(0)));
    x = x.layernorm(this.lnw, this.lnb, this.eps);

    for (const L of this.layers) {
      const q = x.matmul(L.qw).addRowBias(L.qb);
      const k = x.matmul(L.kw).addRowBias(L.kb);
      const v = x.matmul(L.vw).addRowBias(L.vb);
      const attnOut = Tensor.zeros([s, this.dim]);
      attnOut.clear();
      for (let hIdx = 0; hIdx < this.nHead; hIdx++) {
        const off = hIdx * this.headDim;
        const qh = q.sliceCols(off, this.headDim);
        const kh = k.sliceCols(off, this.headDim);
        const vh = v.sliceCols(off, this.headDim);
        const scores = qh.matmul(kh, { transB: true });
        const probs = scores.softmax(1 / Math.sqrt(this.headDim));
        const ctx = probs.matmul(vh);
        attnOut.scatterCols(ctx, off, this.headDim);
      }
      let o = attnOut.matmul(L.ow).addRowBias(L.ob);
      x = x.add(o).layernorm(L.olw, L.olb, this.eps);
      let ff = x.matmul(L.i1w, { transB: true }).addRowBias(L.i1b).gelu();
      ff = ff.matmul(L.i2w, { transB: true }).addRowBias(L.i2b);
      x = x.add(ff).layernorm(L.o2w, L.o2b, this.eps);
    }

    if (!pooled) return [x, null];
    const first = x.firstRow(); // [1, d]
    const p = first.matmul(this.pw, { transB: true }).addRowBias(this.pb).tanh_();
    return [x, p];
  }
}

// ---------------------------------------------------------------------------
// DistilBERT
// ---------------------------------------------------------------------------
export class DistilBertModel {
  constructor(weights, config) {
    this.kind = 'distilbert';
    this.word = W(weights, 'distilbert.embeddings.word_embeddings.weight');
    this.pos = W(weights, 'distilbert.embeddings.position_embeddings.weight');
    this.lnw = W(weights, 'distilbert.embeddings.LayerNorm.weight');
    this.lnb = W(weights, 'distilbert.embeddings.LayerNorm.bias');
    this.vocab = this.word.shape[0];
    this.dim = this.word.shape[1];
    this.maxPos = this.pos.shape[0];
    this.nHead = config.n_heads || 4;
    this.headDim = Math.floor(this.dim / this.nHead);
    this.eps = config.layer_norm_eps ?? config.layer_norm_epsilon ?? 1e-12;
    if (config.sinusoidal_pos_embds === true) {
      throw new Error('[mini.js] this DistilBERT checkpoint uses sinusoidal position embeddings (not supported yet)');
    }
    const nLayer = config.n_layers;
    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      const p = `distilbert.transformer.layer.${i}`;
      this.layers.push({
        qw: W(weights, `${p}.attention.q_lin.weight`), qb: W(weights, `${p}.attention.q_lin.bias`),
        kw: W(weights, `${p}.attention.k_lin.weight`), kb: W(weights, `${p}.attention.k_lin.bias`),
        vw: W(weights, `${p}.attention.v_lin.weight`), vb: W(weights, `${p}.attention.v_lin.bias`),
        ow: W(weights, `${p}.attention.out_lin.weight`), ob: W(weights, `${p}.attention.out_lin.bias`),
        olw: W(weights, `${p}.sa_layer_norm.weight`), olb: W(weights, `${p}.sa_layer_norm.bias`),
        i1w: W(weights, `${p}.ffn.lin1.weight`), i1b: W(weights, `${p}.ffn.lin1.bias`),
        i2w: W(weights, `${p}.ffn.lin2.weight`), i2b: W(weights, `${p}.ffn.lin2.bias`),
        o2w: W(weights, `${p}.output_layer_norm.weight`), o2b: W(weights, `${p}.output_layer_norm.bias`),
      });
    }
    this.nLayer = nLayer;
  }

  forward(ids) {
    const s = ids.length;
    let x = this.word.gather(ids).add(this.pos.gather(arange(s)));
    x = x.layernorm(this.lnw, this.lnb, this.eps);
    for (const L of this.layers) {
      const q = x.matmul(L.qw, { transB: true }).addRowBias(L.qb);
      const k = x.matmul(L.kw, { transB: true }).addRowBias(L.kb);
      const v = x.matmul(L.vw, { transB: true }).addRowBias(L.vb);
      const attnOut = Tensor.zeros([s, this.dim]);
      attnOut.clear();
      for (let hIdx = 0; hIdx < this.nHead; hIdx++) {
        const off = hIdx * this.headDim;
        const qh = q.sliceCols(off, this.headDim);
        const kh = k.sliceCols(off, this.headDim);
        const vh = v.sliceCols(off, this.headDim);
        const scores = qh.matmul(kh, { transB: true });
        const probs = scores.softmax(1 / Math.sqrt(this.headDim));
        const ctx = probs.matmul(vh);
        attnOut.scatterCols(ctx, off, this.headDim);
      }
      let o = attnOut.matmul(L.ow, { transB: true }).addRowBias(L.ob);
      x = x.add(o).layernorm(L.olw, L.olb, this.eps);
      let ff = x.matmul(L.i1w, { transB: true }).addRowBias(L.i1b).gelu();
      ff = ff.matmul(L.i2w, { transB: true }).addRowBias(L.i2b);
      x = x.add(ff).layernorm(L.o2w, L.o2b, this.eps);
    }
    return [x, null];
  }
}

// ---------------------------------------------------------------------------
// Task heads
// ---------------------------------------------------------------------------
export function classificationHead(model, weights, config, hidden) {
  // returns logits [1, numLabels] from pooled/first hidden [1, d]
  const d = hidden.cols;
  const isDistil = model.kind === 'distilbert';
  if (isDistil && weights.has('pre_classifier.weight')) {
    const pw = W(weights, 'pre_classifier.weight'), pb = W(weights, 'pre_classifier.bias');
    let h = hidden.matmul(pw, { transB: true }).addRowBias(pb).relu();
    const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
    return h.matmul(cw, { transB: true }).addRowBias(cb);
  }
  if (!isDistil && weights.has('bert.pooler.dense.weight') && weights.has('classifier.weight')) {
    const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
    return hidden.matmul(cw, { transB: true }).addRowBias(cb);
  }
  // fall back: single linear on first token
  if (weights.has('classifier.weight')) {
    const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
    return hidden.matmul(cw, { transB: true }).addRowBias(cb);
  }
  throw new Error('[mini.js] no classification head found in weights');
}

export function mlmHead(model, weights, config, hidden) {
  // [s, d] -> [s, V] logits
  if (model.kind === 'distilbert') {
    let h = hidden;
    if (weights.has('vocab_transform.weight')) {
      h = h.matmul(W(weights, 'vocab_transform.weight'), { transB: true })
        .addRowBias(W(weights, 'vocab_transform.bias')).gelu();
      if (weights.has('vocab_layer_norm.weight')) {
        h = h.layernorm(W(weights, 'vocab_layer_norm.weight'), W(weights, 'vocab_layer_norm.bias'), model.eps);
      }
    }
    if (weights.has('vocab_projector.weight')) {
      return h.matmul(W(weights, 'vocab_projector.weight'), { transB: true }).addRowBias(W(weights, 'vocab_projector.bias'));
    }
    return h.matmul(model.word, { transB: true });
  }
  if (model.kind === 'bert') {
    let h = hidden;
    if (weights.has('cls.predictions.transform.LayerNorm.weight')) {
      h = h.layernorm(W(weights, 'cls.predictions.transform.LayerNorm.weight'), W(weights, 'cls.predictions.transform.LayerNorm.bias'), model.eps);
    }
    const decoder = weights.get('cls.predictions.decoder.weight') || model.word;
    let t = h.matmul(decoder, { transB: !decoder.transposedStorage, outShape: null });
    if (weights.has('cls.predictions.bias')) t = t.addRowBias(W(weights, 'cls.predictions.bias'));
    return t;
  }
  throw new Error('[mini.js] mlmHead not supported for ' + model.kind);
}
