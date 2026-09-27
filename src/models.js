/**
 * mini.js — Transformer architectures implemented from scratch with Tensors:
 *   - GPT2 (decoder LM, KV-cache, causal attention, tied/untied lm head)
 *   - BERT (bidirectional encoder + MLM/classification heads)
 *   - DistilBERT (6-layer encoder + heads)
 *   - LLAMA FAMILY (Llama / Mistral / Qwen2 / Gemma): RMSNorm, RoPE,
 *     SwiGLU/GeGLU, grouped-query attention (GQA), tied embeddings
 *   - RoBERTa (encoder, dynamic position offsets, classification/MLM heads)
 * Heads: causal LM, classification, token classification, QA (span), MLM.
 * All forward passes are explicit sequences of matmul/softmax/attention ops.
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

/**
 * KV-cache with CHUNKED GROWTH — starts small and doubles as needed, so an
 * 8k-context Llama model does not preallocate gigabytes for a 20-token chat.
 */
class KVCache {
  constructor(model, capacity = 512) {
    this.dim = model.kvDim;
    this.layers = [];
    this.capacity = 0;
    this.len = 0;
    for (let i = 0; i < model.nLayer; i++) this.layers.push({ k: null, v: null });
    this.ensure(capacity);
  }
  ensure(minCapacity) {
    if (minCapacity <= this.capacity) return;
    let cap = Math.max(this.capacity, 256);
    while (cap < minCapacity) cap *= 2;
    for (const l of this.layers) {
      const k = Tensor.zeros([cap, this.dim]);
      const v = Tensor.zeros([cap, this.dim]);
      if (l.k) {
        k.pasteRows(l.k, 0);
        v.pasteRows(l.v, 0);
        k.shape[0] = l.k.shape[0];
        v.shape[0] = l.v.shape[0];
        l.k.dispose(); l.v.dispose();
      } else {
        k.shape[0] = 0; v.shape[0] = 0;
      }
      l.k = k; l.v = v;
    }
    this.capacity = cap;
  }
  dispose() { for (const l of this.layers) { if (l.k) l.k.dispose(); if (l.v) l.v.dispose(); } }
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
    this.kvDim = this.dim;
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
    const lm = weights.get('lm_head.weight');
    this.lmHead = lm || this.wte;
    this.eosTokenId = config.eos_token_id ?? config.bos_token_id ?? undefined;
    this.nLayer = nLayer;
  }

  makeCache(capacity) { return new KVCache(this, capacity || 512); }

  /** Returns logits [s, V] (lastOnly -> [1, V]). */
  forward(ids, { cache = null, lastOnly = false } = {}) {
    const s = ids.length;
    const posStart = cache ? cache.len : 0;
    const total0 = posStart + s;
    let x = this.wte.gather(ids);
    const pe = this.wpe.gather(arange(s, posStart));
    x = x.add(pe);

    const causal = s > 1 || cache === null; // prefill needs a causal mask; single-token decode doesn't
    let mask = null;
    if (causal) {
      const data = new Float32Array(s * total0);
      for (let r = 0; r < s; r++) {
        for (let c = 0; c < total0; c++) {
          if (posStart + r < c) data[r * total0 + c] = -1e9;
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

      if (cache) {
        cache.ensure(total0);
        if (s === 1) {
          cache.layers[li].k.scatterRow(k, posStart);
          cache.layers[li].v.scatterRow(v, posStart);
        } else {
          cache.layers[li].k.pasteRows(k, posStart);
          cache.layers[li].v.pasteRows(v, posStart);
        }
        cache.len = total0;
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
// LLAMA FAMILY — Llama, Mistral, Qwen2, Gemma (and fine-tunes thereof)
// RMSNorm + RoPE + SwiGLU/GeGLU + grouped-query attention (GQA)
// ---------------------------------------------------------------------------
export class LlamaModel {
  constructor(weights, config) {
    this.kind = 'llama';
    const H = config.hidden_size;
    this.dim = H;
    this.embed = W(weights, 'model.embed_tokens.weight'); // [V, H]
    this.vocab = this.embed.shape[0];
    this.maxPos = config.max_position_embeddings || this.embed.shape[0] && 4096 || 4096;
    this.maxPos = config.max_position_embeddings || 4096;
    this.nHead = config.num_attention_heads;
    this.nKV = config.num_key_value_heads || this.nHead;
    this.headDim = config.head_dim || Math.floor(H / this.nHead);
    this.kvDim = this.nKV * this.headDim;
    this.qDim = this.nHead * this.headDim;
    this.eps = config.rms_norm_eps ?? 1e-5;
    this.ropeTheta = config.rope_theta ?? 10000.0;
    this.nLayer = config.num_hidden_layers;
    this.act = (config.hidden_act || 'silu') === 'gelu' ? 'gelu' : 'silu';

    this.layers = [];
    for (let i = 0; i < this.nLayer; i++) {
      const p = `model.layers.${i}`;
      const L = {
        ln1: W(weights, `${p}.input_layernorm.weight`),
        qw: W(weights, `${p}.self_attn.q_proj.weight`),
        kw: W(weights, `${p}.self_attn.k_proj.weight`),
        vw: W(weights, `${p}.self_attn.v_proj.weight`),
        ow: W(weights, `${p}.self_attn.o_proj.weight`),
        ln2: W(weights, `${p}.post_attention_layernorm.weight`),
        gw: W(weights, `${p}.mlp.gate_proj.weight`),
        uw: W(weights, `${p}.mlp.up_proj.weight`),
        dw: W(weights, `${p}.mlp.down_proj.weight`),
      };
      // Qwen2 has q/k/v biases; Llama/Mistral/Gemma do not
      if (weights.has(`${p}.self_attn.q_proj.bias`)) {
        L.qb = W(weights, `${p}.self_attn.q_proj.bias`);
        L.kb = W(weights, `${p}.self_attn.k_proj.bias`);
        L.vb = W(weights, `${p}.self_attn.v_proj.bias`);
      }
      this.layers.push(L);
    }
    this.finalNorm = W(weights, 'model.norm.weight');
    // tied embeddings (SmolLM2, Gemma, some Qwen) or explicit lm_head
    if (config.tie_word_embeddings || !weights.has('lm_head.weight')) {
      this.lmHead = this.embed;
    } else {
      this.lmHead = W(weights, 'lm_head.weight');
    }
  }

  makeCache(capacity) { return new KVCache(this, capacity || 512); }

  forward(ids, { cache = null, lastOnly = false } = {}) {
    const s = ids.length;
    const posStart = cache ? cache.len : 0;
    const total0 = posStart + s;
    const scale = 1 / Math.sqrt(this.headDim);

    let x = this.embed.gather(ids);
    const causal = s > 1 || cache === null;
    let mask = null;
    if (causal) {
      const data = new Float32Array(s * total0);
      for (let r = 0; r < s; r++) {
        for (let c = 0; c < total0; c++) {
          if (posStart + r < c) data[r * total0 + c] = -1e9;
        }
      }
      mask = Tensor.fromArray(data, [s, total0]);
    }

    const headGroups = this.nHead / this.nKV; // q heads per kv head

    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      // ---- attention block ----
      let h = x.rmsnorm(L.ln1, this.eps);
      let q = h.matmul(L.qw, { transB: true });           // [s, qDim]
      let k = h.matmul(L.kw, { transB: true });           // [s, kvDim]
      let v = h.matmul(L.vw, { transB: true });
      if (L.qb) { q = q.addRowBias(L.qb); k = k.addRowBias(L.kb); v = v.addRowBias(L.vb); }
      q = q.rope(posStart, this.headDim, this.ropeTheta);
      k = k.rope(posStart, this.headDim, this.ropeTheta);

      if (cache) {
        cache.ensure(total0);
        if (s === 1) {
          cache.layers[li].k.scatterRow(k, posStart);
          cache.layers[li].v.scatterRow(v, posStart);
        } else {
          cache.layers[li].k.pasteRows(k, posStart);
          cache.layers[li].v.pasteRows(v, posStart);
        }
        cache.len = total0;
      }
      const kAll = cache ? cache.layers[li].k : k;
      const vAll = cache ? cache.layers[li].v : v;

      const attnOut = Tensor.zeros([s, this.qDim]);
      attnOut.clear();
      for (let qi = 0; qi < this.nHead; qi++) {
        const kvHead = Math.floor(qi / headGroups);
        const off = qi * this.headDim;
        const qh = q.sliceCols(off, this.headDim);
        const kh = kAll.sliceCols(kvHead * this.headDim, this.headDim);
        const vh = vAll.sliceCols(kvHead * this.headDim, this.headDim);
        let scores = qh.matmul(kh, { transB: true });
        if (causal && mask) scores = scores.add(mask);
        const probs = scores.softmax(scale);
        const ctx = probs.matmul(vh);
        attnOut.scatterCols(ctx, off, this.headDim);
      }
      let o = attnOut.matmul(L.ow, { transB: true });
      x = x.add(o);

      // ---- SwiGLU / GeGLU block ----
      h = x.rmsnorm(L.ln2, this.eps);
      const gate = h.matmul(L.gw, { transB: true });
      const up = h.matmul(L.uw, { transB: true });
      const acted = this.act === 'gelu' ? gate.gelu().mul(up) : gate.silu().mul(up);
      const mlp = acted.matmul(L.dw, { transB: true });
      x = x.add(mlp);
    }
    if (mask) mask.dispose();

    x = x.rmsnorm(this.finalNorm, this.eps);
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
// RoBERTa — BERT internals with dynamic position offsets (padding_idx=1) and
// no pooler. Weight layout mirrors BERT under the roberta.* prefix.
// ---------------------------------------------------------------------------
export class RobertaModel {
  constructor(weights, config) {
    this.kind = 'roberta';
    this.word = W(weights, 'roberta.embeddings.word_embeddings.weight');
    this.pos = W(weights, 'roberta.embeddings.position_embeddings.weight');
    this.ttype = weights.has('roberta.embeddings.token_type_embeddings.weight')
      ? weights.get('roberta.embeddings.token_type_embeddings.weight') : null;
    this.lnw = W(weights, 'roberta.embeddings.LayerNorm.weight');
    this.lnb = W(weights, 'roberta.embeddings.LayerNorm.bias');
    this.paddingIdx = config.pad_token_id ?? 1;
    this.vocab = this.word.shape[0];
    this.dim = this.word.shape[1];
    this.maxPos = this.pos.shape[0];
    this.nHead = config.num_attention_heads || 4;
    this.headDim = Math.floor(this.dim / this.nHead);
    this.eps = config.layer_norm_eps ?? config.layer_norm_epsilon ?? 1e-5;

    const nLayer = config.num_hidden_layers;
    this.layers = [];
    for (let i = 0; i < nLayer; i++) {
      const p = `roberta.encoder.layer.${i}`;
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
    this.nLayer = nLayer;
  }

  forward(ids) {
    const s = ids.length;
    // roberta position ids: padding_idx + 1 + arange (for unpadded input)
    let x = this.word.gather(ids).add(this.pos.gather(arange(s, this.paddingIdx + 1)));
    if (this.ttype) x = x.add(this.ttype.gather(new Array(s).fill(0)));
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
    return [x, null];
  }
}

// ---------------------------------------------------------------------------
// Task heads
// ---------------------------------------------------------------------------
export function classificationHead(model, weights, config, hidden) {
  // returns logits [1, numLabels] from pooled/first hidden [1, d]
  const d = hidden.cols;
  if (model.kind === 'roberta') {
    if (weights.has('classifier.dense.weight') && weights.has('classifier.out_proj.weight')) {
      let h = hidden.matmul(W(weights, 'classifier.dense.weight'), { transB: true })
        .addRowBias(W(weights, 'classifier.dense.bias')).tanh_();
      return h.matmul(W(weights, 'classifier.out_proj.weight'), { transB: true })
        .addRowBias(W(weights, 'classifier.out_proj.bias'));
    }
    if (weights.has('classifier.weight')) {
      return hidden.matmul(W(weights, 'classifier.weight'), { transB: true })
        .addRowBias(W(weights, 'classifier.bias'));
    }
    throw new Error('[mini.js] no roberta classification head found');
  }
  if (model.kind === 'distilbert') {
    if (weights.has('pre_classifier.weight')) {
      const pw = W(weights, 'pre_classifier.weight'), pb = W(weights, 'pre_classifier.bias');
      let h = hidden.matmul(pw, { transB: true }).addRowBias(pb).relu();
      const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
      return h.matmul(cw, { transB: true }).addRowBias(cb);
    }
  }
  if (!model.kind || model.kind !== 'distilbert') {
    if (weights.has('classifier.weight')) {
      const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
      return hidden.matmul(cw, { transB: true }).addRowBias(cb);
    }
  }
  throw new Error('[mini.js] no classification head found in weights');
}

/** Per-token classification (NER): hidden [s,d] -> logits [s, L]. */
export function tokenClassificationHead(model, weights, hidden) {
  const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
  return hidden.matmul(cw, { transB: true }).addRowBias(cb);
}

/** QA span head: hidden [s,d] -> logits [s,2] (start/end). */
export function qaHead(model, weights, hidden) {
  const prefix = model.kind === 'distilbert' ? '' : '';
  const cw = weights.get('qa_outputs.weight') || weights.get('qa_outputs.weight');
  if (!cw) throw new Error('[mini.js] no qa_outputs head found');
  const cb = weights.get('qa_outputs.bias');
  let t = hidden.matmul(cw, { transB: true });
  if (cb) t = t.addRowBias(cb);
  return t;
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
  if (model.kind === 'bert' || model.kind === 'roberta') {
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
