/*!
 * mini.js v2.4.0 — Run real AI models in your browser. One line of code.
 * A Transformers.js alternative written 100% from scratch: pure HTML/CSS/JS,
 * zero dependencies, no ONNX Runtime, no TensorFlow.js, no WebAssembly blobs.
 *
 *   <script src="mini.js"><\/script>
 *   const gen = await mini.pipeline('text-generation', 'openai-community/gpt2');
 *   const [out] = await gen('Once upon a time');
 *
 * Hand-written components: WebGL2 kernels, CPU kernels, tensor engine,
 * safetensors parser, byte-level BPE + WordPiece tokenizers, GPT-2 / BERT /
 * DistilBERT transformers, KV-cache decoder, HF Hub streaming downloader.
 * MIT License.
 */
(function (globalThis) {
'use strict';
// ───────────────────────────── src/utils.js ─────────────────────────────
/**
 * mini.js — utils: dtype conversion, chunked download helpers, assertions.
 * 100% from scratch. No dependencies.
 */

const MINI_VERSION = '2.4.0';

function assert(cond, msg) {
  if (!cond) throw new Error('[mini.js] ' + msg);
}

function assertDefined(x, name) {
  if (x === undefined || x === null) throw new Error('[mini.js] missing value: ' + name);
  return x;
}

/** Merge a list of Uint8Arrays into one. */
function concatBytes(chunks, totalLen) {
  if (chunks.length === 1) return chunks[0];
  const out = new Uint8Array(totalLen);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

/**
 * Convert raw little-endian bytes of `count` elements to Float32Array.
 * Supports: F32, F16 (IEEE half), BF16 (brain float), F64, I64/I32/I16/I8/U8/U16/U32.
 */
function bytesToFloat32(bytes, dtype, count, byteOffset = 0) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset + byteOffset, bytes.byteLength - byteOffset);
  const out = new Float32Array(count);
  switch (dtype) {
    case 'F32':
      out.set(new Float32Array(bytes.buffer, bytes.byteOffset + byteOffset, count));
      return out;
    case 'F64':
      for (let i = 0; i < count; i++) out[i] = dv.getFloat64(i * 8, true);
      return out;
    case 'F16': {
      // IEEE 754 half -> float32, bit manipulation (from scratch, no Math magic)
      for (let i = 0; i < count; i++) {
        const h = dv.getUint16(i * 2, true);
        out[i] = halfToFloat(h);
      }
      return out;
    }
    case 'BF16': {
      // brain float: just widen 16-bit truncated float32
      for (let i = 0; i < count; i++) {
        const bits = dv.getUint16(i * 2, true) << 16;
        f32Buf[0] = 0; f32Bits[0] = bits;
        out[i] = f32Buf[0];
      }
      return out;
    }
    case 'I64': case 'U64':
      for (let i = 0; i < count; i++) out[i] = Number(dv.getBigInt64(i * 8, true));
      return out;
    case 'I32':
      for (let i = 0; i < count; i++) out[i] = dv.getInt32(i * 4, true);
      return out;
    case 'U32':
      for (let i = 0; i < count; i++) out[i] = dv.getUint32(i * 4, true);
      return out;
    case 'I16':
      for (let i = 0; i < count; i++) out[i] = dv.getInt16(i * 2, true);
      return out;
    case 'U16':
      for (let i = 0; i < count; i++) out[i] = dv.getUint16(i * 2, true);
      return out;
    case 'I8':
      for (let i = 0; i < count; i++) out[i] = dv.getInt8(i);
      return out;
    case 'U8': case 'BOOL':
      for (let i = 0; i < count; i++) out[i] = bytes[i];
      return out;
    default:
      throw new Error('[mini.js] unsupported dtype: ' + dtype);
  }
}

const f32Buf = new Float32Array(1);
const f32Bits = new Uint32Array(f32Buf.buffer);

/** IEEE 754 binary16 -> binary32 via bit twiddling. */
function halfToFloat(h) {
  const sign = (h & 0x8000) << 16;
  let exp = (h & 0x7c00) >> 10;
  let frac = h & 0x03ff;
  if (exp === 0) {
    if (frac === 0) return f32FromBits(sign);                       // zero
    // subnormal half -> normalize
    exp = 127 - 15 + 1;
    while ((frac & 0x0400) === 0) { frac <<= 1; exp--; }
    frac &= 0x03ff;
    return f32FromBits(sign | (exp << 23) | (frac << 13));
  }
  if (exp === 0x1f) return f32FromBits(sign | 0x7f800000 | (frac << 13)); // inf / nan
  return f32FromBits(sign | ((exp - 15 + 127) << 23) | (frac << 13));
}

function f32FromBits(bits) { f32Bits[0] = bits >>> 0; return f32Buf[0]; }

/** Safely parse JSON, with helpful error. */
function parseJSON(text, what) {
  try { return JSON.parse(text); }
  catch (e) { throw new Error('[mini.js] failed to parse ' + what + ': ' + e.message); }
}

/** Tiny event emitter used for progress reporting. */
class ProgressEmitter {
  constructor(callback) { this.cb = typeof callback === 'function' ? callback : null; }
  emit(data) { if (this.cb) { try { this.cb(data); } catch (_) { /* user cb errors must not break loading */ } } }
  progress(file, loaded, total, extra = {}) {
    this.emit(Object.assign({
      status: 'progress', file, loaded, total,
      progress: total > 0 ? Math.min(1, loaded / total) : 0,
    }, extra));
  }
  status(status, file, extra = {}) {
    this.emit(Object.assign({ status, file }, extra));
  }
}

/** Format bytes for humans. */
function formatBytes(n) {
  if (!isFinite(n) || n < 0) return '?';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(n >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
}


// ───────────────────────────── src/cpu-kernels.js ─────────────────────────────
/**
 * mini.js — CPU kernels: hand-written numeric kernels on Float32Array.
 * Blocked / unrolled matmul, softmax, layernorm, gelu, attention helpers.
 * No dependencies.
 */

/** Zero-fill view helper. */
function zeroFill(arr, len) { arr.fill(0, 0, len); }

/**
 * C[M,N] = A[M,K] x B (B stored K,N when transB=false, N,K when transB=true)
 * All row-major contiguous. out must be pre-zeroed or pre-filled with bias.
 */
function cpuMatmul(a, b, out, M, K, N, transB) {
  out.fill(0, 0, M * N);
  if (transB) {
    // B is [N,K]: out[m,n] = dot(a[m,:], b[n,:])
    for (let m = 0; m < M; m++) {
      const aRow = m * K;
      const oRow = m * N;
      for (let n = 0; n < N; n++) {
        const bRow = n * K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        let k = 0;
        const kEnd = K - (K % 4);
        for (; k < kEnd; k += 4) {
          s0 += a[aRow + k] * b[bRow + k];
          s1 += a[aRow + k + 1] * b[bRow + k + 1];
          s2 += a[aRow + k + 2] * b[bRow + k + 2];
          s3 += a[aRow + k + 3] * b[bRow + k + 3];
        }
        let sum = (s0 + s2) + (s1 + s3);
        for (; k < K; k++) sum += a[aRow + k] * b[bRow + k];
        out[oRow + n] = sum;
      }
    }
  } else {
    // B is [K,N]: cache-friendly i-k-j with 4x n unroll
    for (let m = 0; m < M; m++) {
      const aRow = m * K;
      const oRow = m * N;
      for (let k = 0; k < K; k++) {
        const av = a[aRow + k];
        if (av === 0) continue;
        const bRow = k * N;
        let n = 0;
        const nEnd = N - (N % 4);
        for (; n < nEnd; n += 4) {
          out[oRow + n] += av * b[bRow + n];
          out[oRow + n + 1] += av * b[bRow + n + 1];
          out[oRow + n + 2] += av * b[bRow + n + 2];
          out[oRow + n + 3] += av * b[bRow + n + 3];
        }
        for (; n < N; n++) out[oRow + n] += av * b[bRow + n];
      }
    }
  }
}

/** Add bias vector [cols] to every row of x [rows, cols] in place. */
function cpuAddBiasRows(x, bias, rows, cols) {
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let c = 0; c < cols; c++) x[off + c] += bias[c];
  }
}

/** Elementwise add of two same-shape arrays (returns new). */
function cpuAdd(a, b, len) {
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = a[i] + b[i];
  return out;
}

/** Elementwise multiply (broadcast B as single row of len cols across rows). */
function cpuMulRow(a, b, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let c = 0; c < cols; c++) out[off + c] = a[off + c] * b[c];
  }
  return out;
}

function cpuScale(x, s) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * s;
  return out;
}

/** GELU with tanh approximation (the "gelu_new" used by GPT-2). */
function geluScalar(x) {
  const x3 = x * x * x;
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x3)));
}

function cpuGelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = geluScalar(x[i]);
  return out;
}

function cpuRelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

function cpuSilu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / (1 + Math.exp(-x[i]));
  return out;
}

function cpuSigmoid(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
  return out;
}

/** Numerically stable row-wise softmax, in place. rows x cols. Optional scale. */
function cpuSoftmaxRows(x, rows, cols, scale = 1) {
  const buf = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let mx = -Infinity;
    for (let c = 0; c < cols; c++) { const v = x[off + c] * scale; buf[c] = v; if (v > mx) mx = v; }
    let sum = 0;
    for (let c = 0; c < cols; c++) { const e = Math.exp(buf[c] - mx); buf[c] = e; sum += e; }
    const inv = 1 / sum;
    for (let c = 0; c < cols; c++) x[off + c] = buf[c] * inv;
  }
}

/** Row LayerNorm: y = (x-mean)/sqrt(var+eps)*g + b. rows x cols. */
function cpuLayerNorm(x, rows, cols, gamma, beta, eps) {
  const out = new Float32Array(rows * cols);
  const inv = 1 / cols;
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[off + c];
    mean *= inv;
    let variance = 0;
    for (let c = 0; c < cols; c++) { const d = x[off + c] - mean; variance += d * d; }
    variance *= inv;
    const std = 1 / Math.sqrt(variance + eps);
    for (let c = 0; c < cols; c++) out[off + c] = (x[off + c] - mean) * std * gamma[c] + beta[c];
  }
  return out;
}

/** Embedding lookup: out[r,:] = weights[id[r],:]. weights is [vocab, cols] contiguous. */
function cpuEmbedding(weights, cols, ids, rows) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const id = ids[r];
    out.set(weights.subarray(id * cols, id * cols + cols), r * cols);
  }
  return out;
}

/**
 * Extract columns [off, off+len) of src [rows, cols] -> new contiguous array [rows, len].
 */
function cpuSliceCols(src, rows, cols, off, len) {
  const out = new Float32Array(rows * len);
  for (let r = 0; r < rows; r++) {
    out.set(src.subarray(r * cols + off, r * cols + off + len), r * len);
  }
  return out;
}

/**
 * Scatter src [rows, len] into dst [rows, totalCols] at column offset off (dst pre-zeroed
 * or previously written). Used to re-merge attention heads.
 */
function cpuScatterCols(dst, src, rows, totalCols, off, len) {
  for (let r = 0; r < rows; r++) {
    dst.set(src.subarray(r * len, r * len + len), r * totalCols + off);
  }
}

/** Copy one row of src [rows, cols] into dst row `row` of a preallocated [>=rows, cols] buffer. */
function cpuAppendRow(dst, dstStride, src, srcLen, row) {
  dst.set(src.subarray(0, srcLen), row * dstStride);
}

/** Copy a single row of a strided buffer into a contiguous row array. */
function cpuRowOf(strided, stride, row, cols) {
  return strided.subarray(row * stride, row * stride + cols);
}

/** RMSNorm: y = x * g / sqrt(mean(x^2) + eps). rows x cols, no bias. */
function cpuRMSNorm(x, rows, cols, g, eps) {
  const out = new Float32Array(rows * cols);
  const inv = 1 / cols;
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let ss = 0;
    for (let c = 0; c < cols; c++) ss += x[off + c] * x[off + c];
    const rstd = 1 / Math.sqrt(ss * inv + eps);
    for (let c = 0; c < cols; c++) out[off + c] = x[off + c] * rstd * g[c];
  }
  return out;
}

/** Elementwise multiply of two same-length arrays. */
function cpuMul(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

/**
 * Rotary position embedding (RoPE), applied in place to q/k tensors of shape
 * [rows, headDim * nHeads]. Interleaved-half convention used by Llama/Mistral/
 * Qwen/Gemma: pairs (t, t+half) inside each head rotate by angle
 * pos * theta^(-2t/half). Returns a NEW Float32Array.
 */
function cpuRope(x, rows, cols, headDim, pos0, theta) {
  const out = new Float32Array(rows * cols);
  const half = headDim >> 1;
  for (let r = 0; r < rows; r++) {
    const pos = pos0 + r;
    const off = r * cols;
    const nHeads = cols / headDim;
    for (let h = 0; h < nHeads; h++) {
      const hOff = off + h * headDim;
      for (let t = 0; t < half; t++) {
        const invFreq = Math.pow(theta, (-2 * t) / half);
        const ang = pos * invFreq;
        const c = Math.cos(ang), s = Math.sin(ang);
        const a = x[hOff + t], b = x[hOff + t + half];
        out[hOff + t] = a * c - b * s;
        out[hOff + t + half] = b * c + a * s;
      }
    }
  }
  return out;
}


// ───────────────────────────── src/webgl-backend.js ─────────────────────────────
/**
 * mini.js — WebGL2 backend: hand-written GLSL ES 3.00 kernels.
 * Tensors live in single-channel-packed RGBA32F textures (4 floats per texel).
 * No ONNX. No TF.js. Just raw WebGL2.
 *
 * Storage model:
 *  - A tensor is ONE texture: rows on Y, row length padded to multiple of 4 floats (X texels).
 *  - Every kernel writes ZEROS into padded channels so downstream math stays exact.
 *  - Huge-vocab weights (rows > maxTextureSize, e.g. GPT-2 wte 50257x768) are stored
 *    TRANSPOSED [C, R] at load time (flag transposedStorage) and used with the
 *    matmul-T and gather-T kernels, which keeps every texture within GL limits.
 */

const GLSL_HEADER = '#version 300 es\nprecision highp float;\nprecision highp int;\n';

// Fullscreen triangle strip, no vertex buffers needed (WebGL2 gl_VertexID).
const VS_FULLSCREEN = GLSL_HEADER + `
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// helpers shared by kernels: pick channel k (0..7) across two vec4s without dynamic indexing
const GLSL_PICK = `
float pick2(vec4 a, vec4 b, int k) {
  if (k < 4) { return k == 0 ? a.x : k == 1 ? a.y : k == 2 ? a.z : a.w; }
  int k2 = k - 4;
  return k2 == 0 ? b.x : k2 == 1 ? b.y : k2 == 2 ? b.z : b.w;
}`;

function fsKernel(body) { return GLSL_HEADER + GLSL_PICK + '\nout vec4 fragOut;\n' + body; }

const KERNEL_SOURCES = {
  fill: fsKernel(`
    uniform vec4 uVal;
    void main() { fragOut = uVal; }`),

  // A [R,C] + B [R,C]
  add: fsKernel(`
    uniform sampler2D A; uniform sampler2D B; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      vec4 v = texelFetch(A, p, 0) + texelFetch(B, p, 0);
      v = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
      fragOut = v;
    }`),

  // A [R,C] + B [1,C] broadcast
  addRow: fsKernel(`
    uniform sampler2D A; uniform sampler2D B; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      vec4 v = texelFetch(A, p, 0) + texelFetch(B, ivec2(p.x, 0), 0);
      v = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
      fragOut = v;
    }`),

  // unary activation. MODE: 0 gelu-tanh, 1 relu, 2 silu, 3 sigmoid
  unary: fsKernel(`
    uniform sampler2D A; uniform int iC; uniform int uMode;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      vec4 x = texelFetch(A, p, 0);
      vec4 v;
      if (uMode == 0) {
        vec3 x3 = x.x * x.x * x.x, dummy = vec3(0.0);
        float g0 = 0.5 * x.x * (1.0 + tanh(0.7978845608028654 * (x.x + 0.044715 * x.x * x.x * x.x)));
        float g1 = 0.5 * x.y * (1.0 + tanh(0.7978845608028654 * (x.y + 0.044715 * x.y * x.y * x.y)));
        float g2 = 0.5 * x.z * (1.0 + tanh(0.7978845608028654 * (x.z + 0.044715 * x.z * x.z * x.z)));
        float g3 = 0.5 * x.w * (1.0 + tanh(0.7978845608028654 * (x.w + 0.044715 * x.w * x.w * x.w)));
        v = vec4(g0, g1, g2, g3);
      } else if (uMode == 1) {
        v = max(x, vec4(0.0));
      } else if (uMode == 2) {
        v = x / (vec4(1.0) + exp(-x));
      } else if (uMode == 4) {
        v = tanh(x);
      } else {
        v = vec4(1.0) / (vec4(1.0) + exp(-x));
      }
      v = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
      fragOut = v;
    }`),

  scale: fsKernel(`
    uniform sampler2D A; uniform int iC; uniform float uS;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      vec4 v = texelFetch(A, p, 0) * uS;
      v = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
      fragOut = v;
    }`),

  // out[M,N] = A[M,K] x B[K,N] (NN). Padded channels of A/B are zero, so tails are exact.
  matmulNN: fsKernel(`
    uniform sampler2D A; uniform sampler2D B; uniform int iM, iK, iN;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int m = p.y;
      int n0 = p.x * 4;
      if (m >= iM || n0 >= iN) { fragOut = vec4(0.0); return; }
      vec4 acc = vec4(0.0);
      int ktex = (iK + 3) / 4;
      for (int t = 0; t < ktex; t++) {
        vec4 a = texelFetch(A, ivec2(t, m), 0);
        vec4 b0 = texelFetch(B, ivec2(p.x, t * 4 + 0), 0);
        vec4 b1 = texelFetch(B, ivec2(p.x, t * 4 + 1), 0);
        vec4 b2 = texelFetch(B, ivec2(p.x, t * 4 + 2), 0);
        vec4 b3 = texelFetch(B, ivec2(p.x, t * 4 + 3), 0);
        acc += a.x * b0 + a.y * b1 + a.z * b2 + a.w * b3;
      }
      fragOut = n0 + 3 < iN ? acc : vec4(
        n0 + 0 < iN ? acc.x : 0.0, n0 + 1 < iN ? acc.y : 0.0,
        n0 + 2 < iN ? acc.z : 0.0, n0 + 3 < iN ? acc.w : 0.0);
    }`),

  // out[M,N] = A[M,K] x B[N,K]^T (transB). B rows are the N dimension.
  matmulTN: fsKernel(`
    uniform sampler2D A; uniform sampler2D B; uniform int iM, iK, iN;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int m = p.y;
      int n0 = p.x * 4;
      if (m >= iM || n0 >= iN) { fragOut = vec4(0.0); return; }
      vec4 acc = vec4(0.0);
      int ktex = (iK + 3) / 4;
      for (int t = 0; t < ktex; t++) {
        vec4 a = texelFetch(A, ivec2(t, m), 0);
        vec4 b0 = texelFetch(B, ivec2(t, n0 + 0), 0);
        vec4 b1 = texelFetch(B, ivec2(t, n0 + 1), 0);
        vec4 b2 = texelFetch(B, ivec2(t, n0 + 2), 0);
        vec4 b3 = texelFetch(B, ivec2(t, n0 + 3), 0);
        acc.x += a.x * b0.x + a.y * b0.y + a.z * b0.z + a.w * b0.w;
        acc.y += a.x * b1.x + a.y * b1.y + a.z * b1.z + a.w * b1.w;
        acc.z += a.x * b2.x + a.y * b2.y + a.z * b2.z + a.w * b2.w;
        acc.w += a.x * b3.x + a.y * b3.y + a.z * b3.z + a.w * b3.w;
      }
      fragOut = n0 + 3 < iN ? acc : vec4(
        n0 + 0 < iN ? acc.x : 0.0, n0 + 1 < iN ? acc.y : 0.0,
        n0 + 2 < iN ? acc.z : 0.0, n0 + 3 < iN ? acc.w : 0.0);
    }`),

  // out[M,N] = A[M,K] x Wt where Wt is stored TRANSPOSED [K,N] and equals B^T for logical B[N,K].
  // T texel (x, y) channels = T[y][4x..4x+3] = B[4x..4x+3][y] -> 4 consecutive n for ONE k.
  matmulMT: fsKernel(`
    uniform sampler2D A; uniform sampler2D T; uniform int iM, iK, iN;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int m = p.y;
      int n0 = p.x * 4;
      if (m >= iM || n0 >= iN) { fragOut = vec4(0.0); return; }
      vec4 acc = vec4(0.0);
      int ktex = (iK + 3) / 4;
      for (int t = 0; t < ktex; t++) {
        vec4 a = texelFetch(A, ivec2(t, m), 0);
        vec4 t0 = texelFetch(T, ivec2(p.x, t * 4 + 0), 0);
        vec4 t1 = texelFetch(T, ivec2(p.x, t * 4 + 1), 0);
        vec4 t2 = texelFetch(T, ivec2(p.x, t * 4 + 2), 0);
        vec4 t3 = texelFetch(T, ivec2(p.x, t * 4 + 3), 0);
        acc += a.x * t0 + a.y * t1 + a.z * t2 + a.w * t3;
      }
      fragOut = acc;
    }`),

  // Row softmax over A [R, C] with scale. One fragment per output texel; row stats
  // recomputed per texel (cheap, avoids extra passes). Padded channels written as 0.
  softmax: fsKernel(`
    uniform sampler2D A; uniform int iC; uniform float uScale;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int y = p.y;
      int ctex = (iC + 3) / 4;
      float mx = -1e30;
      for (int x = 0; x < ctex; x++) {
        vec4 v = texelFetch(A, ivec2(x, y), 0);
        mx = max(mx, v.x * uScale); mx = max(mx, v.y * uScale);
        mx = max(mx, v.z * uScale); mx = max(mx, v.w * uScale);
      }
      mx = max(mx, -1e30);
      float sum = 0.0;
      for (int x = 0; x < ctex; x++) {
        vec4 v = texelFetch(A, ivec2(x, y), 0) * uScale;
        sum += (x * 4 + 0 < iC ? exp(v.x - mx) : 0.0);
        sum += (x * 4 + 1 < iC ? exp(v.y - mx) : 0.0);
        sum += (x * 4 + 2 < iC ? exp(v.z - mx) : 0.0);
        sum += (x * 4 + 3 < iC ? exp(v.w - mx) : 0.0);
      }
      float inv = 1.0 / max(sum, 1e-30);
      vec4 r = texelFetch(A, p, 0) * uScale;
      vec4 e = exp(r - mx) * inv;
      int base = p.x * 4;
      fragOut = base + 3 < iC ? e : vec4(
        base + 0 < iC ? e.x : 0.0, base + 1 < iC ? e.y : 0.0,
        base + 2 < iC ? e.z : 0.0, base + 3 < iC ? e.w : 0.0);
    }`),

  // Row LayerNorm: gamma [1,C], beta [1,C]
  layernorm: fsKernel(`
    uniform sampler2D A; uniform sampler2D G; uniform sampler2D BE; uniform int iC; uniform float uEps;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int y = p.y;
      int ctex = (iC + 3) / 4;
      float sum = 0.0;
      for (int x = 0; x < ctex; x++) {
        vec4 v = texelFetch(A, ivec2(x, y), 0);
        sum += (x * 4 + 0 < iC ? v.x : 0.0) + (x * 4 + 1 < iC ? v.y : 0.0)
             + (x * 4 + 2 < iC ? v.z : 0.0) + (x * 4 + 3 < iC ? v.w : 0.0);
      }
      float mean = sum / float(iC);
      float vsum = 0.0;
      for (int x = 0; x < ctex; x++) {
        vec4 v = texelFetch(A, ivec2(x, y), 0);
        vec4 d = v - mean;
        vsum += (x * 4 + 0 < iC ? d.x * d.x : 0.0) + (x * 4 + 1 < iC ? d.y * d.y : 0.0)
              + (x * 4 + 2 < iC ? d.z * d.z : 0.0) + (x * 4 + 3 < iC ? d.w * d.w : 0.0);
      }
      float rstd = inversesqrt(vsum / float(iC) + uEps);
      vec4 g = texelFetch(G, ivec2(p.x, 0), 0);
      vec4 b = texelFetch(BE, ivec2(p.x, 0), 0);
      vec4 v = texelFetch(A, p, 0);
      vec4 o = (v - mean) * rstd * g + b;
      int base = p.x * 4;
      fragOut = base + 3 < iC ? o : vec4(
        base + 0 < iC ? o.x : 0.0, base + 1 < iC ? o.y : 0.0,
        base + 2 < iC ? o.z : 0.0, base + 3 < iC ? o.w : 0.0);
    }`),

  // out[r,:] = W[id[r],:] for W [V, C] normal layout. ids: float ids in a [ceil(s/4)] wide texture.
  gather: fsKernel(`
    uniform sampler2D W; uniform sampler2D IDS; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int row = p.y;
      float idf = pick2(texelFetch(IDS, ivec2(row >> 2, 0), 0), texelFetch(IDS, ivec2((row >> 2) + 1, 0), 0), row & 3);
      fragOut = texelFetch(W, ivec2(p.x, int(idf + 0.5)), 0);
    }`),

  // out[r,:] = W[id[r],:] for W stored TRANSPOSED [C, V]: W[n,k]=T[k][n] -> W[id, c] = T[c][id]
  gatherT: fsKernel(`
    uniform sampler2D T; uniform sampler2D IDS; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int row = p.y;
      float idf = pick2(texelFetch(IDS, ivec2(row >> 2, 0), 0), texelFetch(IDS, ivec2((row >> 2) + 1, 0), 0), row & 3);
      int id = int(idf + 0.5);
      int tx = id >> 2;
      int tc = id & 3;
      vec4 t0 = texelFetch(T, ivec2(tx, p.x * 4 + 0), 0);
      vec4 t1 = texelFetch(T, ivec2(tx, p.x * 4 + 1), 0);
      vec4 t2 = texelFetch(T, ivec2(tx, p.x * 4 + 2), 0);
      vec4 t3 = texelFetch(T, ivec2(tx, p.x * 4 + 3), 0);
      vec4 v = vec4(pick2(t0, t1, tc), pick2(t0, t1, tc), pick2(t0, t1, tc), pick2(t0, t1, tc));
      // careful: channels of T row y are W[n..n+3, y]; we need W[id, 4x+c] = T[4x+c][id]
      float c0 = pick2(texelFetch(T, ivec2(tx, p.x * 4 + 0), 0), texelFetch(T, ivec2(tx + 1, p.x * 4 + 0), 0), tc);
      float c1 = pick2(texelFetch(T, ivec2(tx, p.x * 4 + 1), 0), texelFetch(T, ivec2(tx + 1, p.x * 4 + 1), 0), tc);
      float c2 = pick2(texelFetch(T, ivec2(tx, p.x * 4 + 2), 0), texelFetch(T, ivec2(tx + 1, p.x * 4 + 2), 0), tc);
      float c3 = pick2(texelFetch(T, ivec2(tx, p.x * 4 + 3), 0), texelFetch(T, ivec2(tx + 1, p.x * 4 + 3), 0), tc);
      v = vec4(c0, c1, c2, c3);
      int base = p.x * 4;
      fragOut = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
    }`),

  // out[:, 0..len) = A[:, off..off+len)
  sliceCols: fsKernel(`
    uniform sampler2D A; uniform int iOff, iLen;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      int s0 = iOff + base;
      vec4 a = texelFetch(A, ivec2(s0 >> 2, p.y), 0);
      vec4 b = (s0 & 3) == 0 ? a : texelFetch(A, ivec2((s0 >> 2) + 1, p.y), 0);
      int tc = s0 & 3;
      vec4 v = vec4(pick2(a, b, tc), pick2(a, b, tc + 1), pick2(a, b, tc + 2), pick2(a, b, tc + 3));
      v = base + 3 < iLen ? v : vec4(
        base + 0 < iLen ? v.x : 0.0, base + 1 < iLen ? v.y : 0.0,
        base + 2 < iLen ? v.z : 0.0, base + 3 < iLen ? v.w : 0.0);
      fragOut = v;
    }`),

  // dst[:, off..off+len) = A[:, :] ; fragments outside the region are discarded (keep dst)
  scatterCols: fsKernel(`
    uniform sampler2D A; uniform int iOff, iLen;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      if (base >= iLen) discard;
      fragOut = texelFetch(A, ivec2(p.x, p.y), 0);
    }`),

  // dst[row, :] = A[0, :] ; other fragments discarded
  scatterRow: fsKernel(`
    uniform sampler2D A; uniform int iRow;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      if (p.y != iRow) discard;
      fragOut = texelFetch(A, ivec2(p.x, 0), 0);
    }`),

  // add attention mask [R,T] elementwise (A + M)
  maskAdd: fsKernel(`
    uniform sampler2D A; uniform sampler2D M; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      vec4 v = texelFetch(A, p, 0) + texelFetch(M, p, 0);
      fragOut = v;
    }`),

  // out[0,:] = A[iRow, :]
  sliceRows: fsKernel(`
    uniform sampler2D A; uniform int iRow;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      if (p.y != 0) discard;
      fragOut = texelFetch(A, ivec2(p.x, iRow + p.y), 0);
    }`),

  // dst[iRowOff + r, :] = A[r, :] for r < iSrcRows ; other fragments discarded
  pasteRows: fsKernel(`
    uniform sampler2D A; uniform int iRowOff, iSrcRows;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int y = p.y - iRowOff;
      if (y < 0 || y >= iSrcRows) discard;
      fragOut = texelFetch(A, ivec2(p.x, y), 0);
    }`),

  // RMSNorm rows: y = x * rstd * G (no mean subtraction, no bias)
  rmsnorm: fsKernel(`
    uniform sampler2D A; uniform sampler2D G; uniform int iC; uniform float uEps;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int y = p.y;
      int ctex = (iC + 3) / 4;
      float ss = 0.0;
      for (int x = 0; x < ctex; x++) {
        vec4 v = texelFetch(A, ivec2(x, y), 0);
        ss += (x * 4 + 0 < iC ? v.x * v.x : 0.0) + (x * 4 + 1 < iC ? v.y * v.y : 0.0)
            + (x * 4 + 2 < iC ? v.z * v.z : 0.0) + (x * 4 + 3 < iC ? v.w * v.w : 0.0);
      }
      float rstd = inversesqrt(ss / float(iC) + uEps);
      vec4 g = texelFetch(G, ivec2(p.x, 0), 0);
      vec4 v = texelFetch(A, p, 0);
      vec4 o = v * rstd * g;
      int base = p.x * 4;
      fragOut = base + 3 < iC ? o : vec4(
        base + 0 < iC ? o.x : 0.0, base + 1 < iC ? o.y : 0.0,
        base + 2 < iC ? o.z : 0.0, base + 3 < iC ? o.w : 0.0);
    }`),

  // elementwise A * B (same shape)
  mul: fsKernel(`
    uniform sampler2D A; uniform sampler2D B; uniform int iC;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int base = p.x * 4;
      vec4 v = texelFetch(A, p, 0) * texelFetch(B, p, 0);
      v = base + 3 < iC ? v : vec4(
        base + 0 < iC ? v.x : 0.0, base + 1 < iC ? v.y : 0.0,
        base + 2 < iC ? v.z : 0.0, base + 3 < iC ? v.w : 0.0);
      fragOut = v;
    }`),

  // RoPE over [rows, headDim*nHeads]. Rotates pairs (t, t+half) per head by
  // angle = (iPos0 + row) * theta^(-2*(t mod half)/half).
  rope: fsKernel(`
    uniform sampler2D A; uniform int iC; uniform int iHeadDim;
    uniform float uTheta; uniform float uPos0;
    void main() {
      ivec2 p = ivec2(gl_FragCoord.xy);
      int row = p.y;
      int base = p.x * 4;
      int half = iHeadDim / 2;
      float pos = uPos0 + float(row);
      vec4 o = vec4(0.0);
      for (int ch = 0; ch < 4; ch++) {
        int c = base + ch;
        if (c >= iC) break;
        int t = c % iHeadDim;
        int t2 = t < half ? t : t - half;
        float invFreq = pow(uTheta, -2.0 * float(t2) / float(half));
        float ang = pos * invFreq;
        float cs = cos(ang), sn = sin(ang);
        int pc = t < half ? c + half : c - half;
        vec4 pv = texelFetch(A, ivec2(pc >> 2, row), 0);
        float partner = (pc & 3) == 0 ? pv.x : (pc & 3) == 1 ? pv.y : (pc & 3) == 2 ? pv.z : pv.w;
        vec4 av = texelFetch(A, p, 0);
        float v = ch == 0 ? av.x : ch == 1 ? av.y : ch == 2 ? av.z : av.w;
        o[ch] = t < half ? v * cs - partner * sn : v * cs + partner * sn;
      }
      fragOut = base + 3 < iC ? o : vec4(
        base + 0 < iC ? o.x : 0.0, base + 1 < iC ? o.y : 0.0,
        base + 2 < iC ? o.z : 0.0, base + 3 < iC ? o.w : 0.0);
    }`),
};

class GLBackend {
  constructor() {
    if (typeof document === 'undefined') throw new Error('[mini.js] WebGL backend requires a browser (document)');
    const canvas = document.createElement('canvas');
    canvas.width = 1; canvas.height = 1;
    const gl = canvas.getContext('webgl2', {
      alpha: false, depth: false, stencil: false, antialias: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: false,
      desynchronized: true,
    });
    if (!gl) throw new Error('[mini.js] WebGL2 not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('[mini.js] EXT_color_buffer_float not available (needed for float render targets)');
    gl.getExtension('OES_texture_float_linear');
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST); gl.disable(gl.CULL_FACE);
    gl.disable(gl.DITHER); gl.disable(gl.RASTERIZER_DISCARD);
    gl.clearColor(0, 0, 0, 0);
    this.gl = gl;
    this.canvas = canvas;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.fbo = gl.createFramebuffer();
    this.programs = new Map();
    this.quadVao = gl.createVertexArray();
    gl.bindVertexArray(this.quadVao);
    this.liveTextures = 0;
  }

  /** Compile + cache a kernel program. */
  program(name) {
    let p = this.programs.get(name);
    if (p) return p;
    const gl = this.gl;
    const src = KERNEL_SOURCES[name];
    if (!src) throw new Error('[mini.js] unknown kernel: ' + name);
    const compile = (type, source) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, source);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        throw new Error('[mini.js] shader compile error (' + name + '): ' + log);
      }
      return sh;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS_FULLSCREEN));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, src));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('[mini.js] program link error (' + name + '): ' + gl.getProgramInfoLog(prog));
    }
    const uniforms = {};
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(prog, i);
      uniforms[info.name] = { loc: gl.getUniformLocation(prog, info.name), type: info.type };
    }
    p = { prog, uniforms };
    this.programs.set(name, p);
    return p;
  }

  /**
   * Run a kernel writing into `out` texture.
   * samplers: array of [uniformName, texture handle]
   * uniforms: flat object of numbers (dispatched by the uniform's declared GLSL type).
   */
  run(name, outTex, outW, outH, samplers, uniforms = {}, x0 = 0, y0 = 0) {
    const gl = this.gl;
    const p = this.program(name);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (ok !== gl.FRAMEBUFFER_COMPLETE) throw new Error('[mini.js] incomplete framebuffer (' + name + ')');
    gl.viewport(x0, y0, outW, outH);
    gl.useProgram(p.prog);
    gl.bindVertexArray(this.quadVao);
    let unit = 0;
    for (const [uname, handle] of samplers) {
      const u = p.uniforms[uname];
      if (!u) continue;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, handle.tex ? handle.tex : handle);
      gl.uniform1i(u.loc, unit);
      unit++;
    }
    for (const [uname, val] of Object.entries(uniforms)) {
      const u = p.uniforms[uname];
      if (!u) continue;
      if (u.type === gl.FLOAT_VEC4) gl.uniform4f(u.loc, val, val, val, val);
      else if (u.type === gl.FLOAT) gl.uniform1f(u.loc, val);
      else gl.uniform1i(u.loc, val);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Allocate a texture [texW x texH] texels (each texel = 4 floats). Optionally upload padded data. */
  createTexture(texW, texH, paddedData = null) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (paddedData) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, paddedData);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, null);
    }
    this.liveTextures++;
    return tex;
  }

  disposeTexture(tex) {
    if (tex) { this.gl.deleteTexture(tex); this.liveTextures--; }
  }

  /** Build padded (stride multiple of 4) upload buffer from a logical Float32Array. */
  static padData(data, rows, cols) {
    const stride = ((cols + 3) >> 2) << 2;
    const padded = new Float32Array(rows * stride);
    if (stride === cols) {
      padded.set(data);
    } else {
      for (let r = 0; r < rows; r++) {
        padded.set(data.subarray(r * cols, r * cols + cols), r * stride);
      }
    }
    return padded;
  }

  /** Read back the LOGICAL part of a texture (removes row padding). */
  readTexture(tex, rows, cols, texW) {
    const gl = this.gl;
    const stride = texW * 4;
    const raw = new Float32Array(texW * rows * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.readPixels(0, 0, texW, rows, gl.RGBA, gl.FLOAT, raw);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (stride === cols) return raw;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      out.set(raw.subarray(r * stride, r * stride + cols), r * cols);
    }
    return out;
  }

  /** CPU transpose helper used at weight-load time. */
  static transpose(data, rows, cols) {
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      for (let c = 0; c < cols; c++) out[c * rows + r] = data[off + c];
    }
    return out;
  }
}

let _glInstance = null;
function getGL() {
  if (_glInstance === null) {
    _glInstance = new GLBackend(); // may throw; caller decides fallback
  }
  return _glInstance;
}
function resetGL() { _glInstance = null; }


// ───────────────────────────── src/tensor.js ─────────────────────────────
/**
 * mini.js — Tensor: a minimal eager tensor with two from-scratch backends.
 *  - 'cpu'   : Float32Array kernels (always available, incl. Node.js)
 *  - 'webgl' : WebGL2 RGBA32F textures + hand-written GLSL kernels (browser)
 *
 * A tensor is logically [ ...leading rows, cols ]; ops treat it as 2D [rows, cols].
 * Data is always contiguous row-major Float32Array on CPU; on GPU it is a padded
 * texture (row stride rounded up to 4 floats).
 */

const CK = { zeroFill: zeroFill, cpuMatmul: cpuMatmul, cpuAddBiasRows: cpuAddBiasRows, cpuAdd: cpuAdd, cpuMulRow: cpuMulRow, cpuScale: cpuScale, geluScalar: geluScalar, cpuGelu: cpuGelu, cpuRelu: cpuRelu, cpuSilu: cpuSilu, cpuSigmoid: cpuSigmoid, cpuSoftmaxRows: cpuSoftmaxRows, cpuLayerNorm: cpuLayerNorm, cpuEmbedding: cpuEmbedding, cpuSliceCols: cpuSliceCols, cpuScatterCols: cpuScatterCols, cpuAppendRow: cpuAppendRow, cpuRowOf: cpuRowOf, cpuRMSNorm: cpuRMSNorm, cpuMul: cpuMul, cpuRope: cpuRope };

let ACTIVE_BACKEND = null; // 'cpu' | 'webgl'

function setBackend(name) {
  if (name === 'webgl') { getGL(); ACTIVE_BACKEND = 'webgl'; }
  else if (name === 'cpu') ACTIVE_BACKEND = 'cpu';
  else throw new Error('[mini.js] unknown backend: ' + name);
  return ACTIVE_BACKEND;
}

function detectBestBackend() {
  if (typeof document !== 'undefined') {
    try { setBackend('webgl'); return 'webgl'; } catch (_) { /* fall through */ }
  }
  setBackend('cpu');
  return 'cpu';
}

function currentBackend() {
  if (ACTIVE_BACKEND === null) detectBestBackend();
  return ACTIVE_BACKEND;
}

let TENSOR_ID = 0;

// ---------------------------------------------------------------------------
// Arena: automatic disposal of intermediate tensors created inside a scope.
// Prevents GPU memory leaks during long generation loops.
// ---------------------------------------------------------------------------
let ARENA = null;

/**
 * Run fn, auto-disposing every Tensor it created except the returned one(s).
 * If fn returns an array, all Tensors in the array are kept.
 */
function arenaRun(fn) {
  const prev = ARENA;
  ARENA = [];
  try {
    const result = fn();
    const keep = Array.isArray(result) ? new Set(result) : new Set([result]);
    for (const t of ARENA) if (!keep.has(t) && !t.disposed) t.dispose();
    return result;
  } catch (e) {
    for (const t of ARENA) if (!t.disposed) t.dispose();
    throw e;
  } finally {
    ARENA = prev;
  }
}

class Tensor {
  /**
   * shape: array of ints (last dim = cols). data: Float32Array (CPU path) —
   * for webgl you normally use Tensor.zeros + ops, or Tensor.fromArray.
   */
  constructor(shape, data, opts = {}) {
    this.id = ++TENSOR_ID;
    this.shape = shape.slice();
    this.disposed = false;
    this.transposedStorage = !!opts.transposedStorage; // GPU: weight stored [cols, rows]
    this.backend = opts.backend || currentBackend();
    this._ownsTexture = true;
    const size = this.shape.reduce((a, b) => a * b, 1);
    this.size = size;
    if (this.backend === 'cpu') {
      this.data = data instanceof Float32Array ? data : new Float32Array(size);
      if (data && !(data instanceof Float32Array) && data.length !== undefined) this.data.set(data);
    } else {
      this.data = null;
      const gl = getGL();
      const rows = this.rows, cols = this.cols;
      this._checkGLDims(gl, rows, cols);
      if (this.transposedStorage) {
        // logical [rows=R, cols=C] stored as [C, R] texture — caller supplies ALREADY
        // transposed contiguous data of length R*C.
        const src = data || new Float32Array(size);
        const stRows = cols, stCols = rows;
        const stTexW = ((stCols + 3) >> 2);
        const stTexH = stRows;
        this._texW = stTexW; this._texH = stTexH; this._stride = stTexW * 4;
        const padded = GLBackend.padData(src, stRows, stCols);
        this.tex = gl.createTexture(stTexW, stTexH, padded);
      } else {
        const texW = (cols + 3) >> 2;
        this._texW = texW; this._texH = rows; this._stride = texW * 4;
        const padded = GLBackend.padData(data || new Float32Array(size), rows, cols);
        this.tex = gl.createTexture(texW, rows, padded);
      }
    }
  }

  _checkGLDims(gl, rows, cols) {
    if (this.transposedStorage) {
      // storage: [cols, rows]
      if (cols > gl.maxTex || ((rows + 3) >> 2) > gl.maxTex) {
        throw new Error(`[mini.js] tensor ${rows}x${cols} too large for GPU (max ${gl.maxTex})`);
      }
    } else if (rows > gl.maxTex || ((cols + 3) >> 2) > gl.maxTex) {
      throw new Error(`[mini.js] tensor ${rows}x${cols} too large for GPU (max ${gl.maxTex}); backend fallback needed`);
    }
  }

  get rows() { let r = 1; for (let i = 0; i < this.shape.length - 1; i++) r *= this.shape[i]; return r; }
  get cols() { return this.shape[this.shape.length - 1]; }

  static fromArray(data, shape) {
    return new Tensor(shape, data instanceof Float32Array ? data : Float32Array.from(data));
  }
  static zeros(shape) { return new Tensor(shape, null); }

  reshape(newShape) {
    const size = newShape.reduce((a, b) => a * b, 1);
    if (size !== this.size) throw new Error(`[mini.js] reshape ${this.shape} -> ${newShape} size mismatch`);
    if (this.backend === 'cpu') {
      const t = Object.create(Tensor.prototype);
      t.id = ++TENSOR_ID; t.shape = newShape.slice(); t.size = size; t.data = this.data;
      t.backend = 'cpu'; t.disposed = false; t.transposedStorage = false;
      return t;
    }
    // GPU: metadata reshape only if it preserves the padded layout (only last-dim growth allowed to differ)
    const rows = newShape.reduce((a, b) => a * b, 1);
    const cols = newShape[newShape.length - 1];
    const t = Object.create(Tensor.prototype);
    t.id = ++TENSOR_ID; t.shape = newShape.slice(); t.size = size; t.data = null;
    t.backend = 'webgl'; t.disposed = false; t.transposedStorage = false;
    t.tex = this.tex; t._texW = (cols + 3) >> 2; t._texH = rows; t._stride = t._texW * 4;
    t._ownsTexture = false;
    if (t._stride !== this._stride && this.rows !== 1 && rows !== 1) {
      throw new Error('[mini.js] GPU reshape with different stride not supported for multi-row tensors');
    }
    return t;
  }

  /** Upload CPU data into a GPU tensor's texture (used after CPU-side prep). */
  _upload(paddedData) {
    const gl = getGL();
    const glc = gl.gl;
    glc.bindTexture(glc.TEXTURE_2D, this.tex);
    glc.texSubImage2D(glc.TEXTURE_2D, 0, 0, 0, this._texW, this._texH, glc.RGBA, glc.FLOAT, paddedData);
  }

  /** Full CPU copy of logical data. */
  read() {
    if (this.disposed) throw new Error('[mini.js] read() on disposed tensor');
    if (this.backend === 'cpu') return this.data;
    const gl = getGL();
    return gl.readTexture(this.tex, this._texH, this.cols, this._texW);
  }
  toArray() { return Array.from(this.read()); }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.backend === 'webgl') {
      getGL().disposeTexture(this.tex);
      this.tex = null;
    }
    this.data = null;
  }

  _cpu() {
    if (this.backend === 'cpu') return this.data;
    throw new Error('[mini.js] internal: expected CPU tensor');
  }

  // ------------------------------------------------------------------ ops

  /** out = this @ B ; transB=true means B stored [cols, rowsOfB] (y = x@W^T). */
  matmul(B, { transB = false, outShape = null } = {}) {
    if (B.transposedStorage) {
      // logical B [N,K] stored [K,N]; compute this[M,K] @ B[N,K]^T via matmulMT
      const M = this.rows, K = this.cols;
      const N = B.shape[0];
      const out = Tensor.zeros(outShape || [M, N]);
      if (this.backend === 'webgl') {
        getGL().run('matmulMT', out.tex, out._texW, out._texH,
          [['A', this.tex], ['T', B.tex]],
          { iM: M, iK: K, iN: N });
      } else {
        // CPU: storage is B^T [K,N] -> plain NN matmul
        CK.cpuMatmul(this.data, B.data, out.data, M, K, N, false);
      }
      return out;
    }
    const M = this.rows, K = this.cols;
    const N = transB ? B.shape[0] : B.shape[B.shape.length - 1];
    const out = Tensor.zeros(outShape || [M, N]);
    if (this.backend === 'webgl') {
      const gl = getGL();
      gl.run(transB ? 'matmulTN' : 'matmulNN', out.tex, out._texW, out._texH,
        [['A', this.tex], ['B', B.tex]],
        { iM: M, iK: K, iN: N });
    } else {
      CK.cpuMatmul(this.data, B.data, out.data, M, K, N, transB);
    }
    return out;
  }

  add(o) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('add', out.tex, out._texW, out._texH,
        [['A', this.tex], ['B', o.tex]], { iC: this.cols });
    } else {
      out.data.set(CK.cpuAdd(this.data, o.data, this.size));
    }
    return out;
  }

  /** Broadcast add bias [cols] over rows. */
  addRowBias(bias) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('addRow', out.tex, out._texW, out._texH,
        [['A', this.tex], ['B', bias.tex]], { iC: this.cols });
    } else {
      out.data.set(this.data);
      CK.cpuAddBiasRows(out.data, bias.data, this.rows, this.cols);
    }
    return out;
  }

  _unary(mode) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('unary', out.tex, out._texW, out._texH,
        [['A', this.tex]], { iC: this.cols, uMode: mode });
    } else {
      const f = mode === 0 ? CK.cpuGelu : mode === 1 ? CK.cpuRelu : mode === 2 ? CK.cpuSilu : CK.cpuSigmoid;
      out.data.set(f(this.data));
    }
    return out;
  }
  gelu() { return this._unary(0); }
  relu() { return this._unary(1); }
  silu() { return this._unary(2); }
  sigmoid() { return this._unary(3); }
  tanh_() {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('unary', out.tex, out._texW, out._texH, [['A', this.tex]], { iC: this.cols, uMode: 4 });
    } else {
      const d = this.data, o = out.data;
      for (let i = 0; i < d.length; i++) o[i] = Math.tanh(d[i]);
    }
    return out;
  }

  /** Extract one row -> new [1, cols] tensor. */
  rowSlice(r) {
    const out = Tensor.zeros([1, this.cols]);
    if (this.backend === 'webgl') {
      getGL().run('sliceRows', out.tex, out._texW, out._texH,
        [['A', this.tex]], { iRow: r });
    } else {
      out.data.set(this.data.subarray(r * this.cols, r * this.cols + this.cols));
    }
    return out;
  }
  firstRow() { return this.rowSlice(0); }

  /** dst op: copy all rows of src into dst starting at row iRowOff. */
  pasteRows(src, iRowOff) {
    if (this.backend === 'webgl') {
      getGL().run('pasteRows', this.tex, this._texW, this._texH,
        [['A', src.tex]], { iRowOff, iSrcRows: src.rows });
    } else {
      this.data.set(src.data, iRowOff * this.cols);
    }
    this.shape[0] = iRowOff + src.rows; // logical rows = valid prefix length
    return this;
  }

  scale(s) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('scale', out.tex, out._texW, out._texH,
        [['A', this.tex]], { iC: this.cols, uS: s });
    } else {
      out.data.set(CK.cpuScale(this.data, s));
    }
    return out;
  }

  /** Row-wise softmax (optionally scaled pre-softmax). */
  softmax(scale = 1) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('softmax', out.tex, out._texW, out._texH,
        [['A', this.tex]], { iC: this.cols, uScale: scale });
    } else {
      out.data.set(this.data);
      CK.cpuSoftmaxRows(out.data, this.rows, this.cols, scale);
    }
    return out;
  }

  layernorm(gamma, beta, eps = 1e-5) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('layernorm', out.tex, out._texW, out._texH,
        [['A', this.tex], ['G', gamma.tex], ['BE', beta.tex]],
        { iC: this.cols, uEps: eps });
    } else {
      out.data.set(CK.cpuLayerNorm(this.data, this.rows, this.cols, gamma.data, beta.data, eps));
    }
    return out;
  }

  /** RMSNorm (Llama family): y = x * g / sqrt(mean(x²)+eps). */
  rmsnorm(gamma, eps = 1e-5) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('rmsnorm', out.tex, out._texW, out._texH,
        [['A', this.tex], ['G', gamma.tex]],
        { iC: this.cols, uEps: eps });
    } else {
      out.data.set(CK.cpuRMSNorm(this.data, this.rows, this.cols, gamma.data, eps));
    }
    return out;
  }

  /** Elementwise multiply (same shape). */
  mul(o) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('mul', out.tex, out._texW, out._texH,
        [['A', this.tex], ['B', o.tex]], { iC: this.cols });
    } else {
      out.data.set(CK.cpuMul(this.data, o.data));
    }
    return out;
  }

  /**
   * Rotary position embedding (Llama family). headDim = dim per head;
   * this tensor is [rows, headDim * nHeads]; pos0 = position of first row.
   * theta = rope base (e.g. 10000, 500000).
   */
  rope(pos0, headDim, theta = 10000.0) {
    const out = Tensor.zeros(this.shape.slice());
    if (this.backend === 'webgl') {
      getGL().run('rope', out.tex, out._texW, out._texH,
        [['A', this.tex]],
        { iC: this.cols, iHeadDim: headDim, uTheta: theta, uPos0: pos0 });
    } else {
      out.data.set(CK.cpuRope(this.data, this.rows, this.cols, headDim, pos0, theta));
    }
    return out;
  }

  /** Add attention-mask matrix (same shape) — used before softmax. */
  addMask(mask) { return this.add(mask); }

  /**
   * Embedding gather: this is the weight [V, C]; ids = int array length s.
   * Returns [s, C].
   */
  gather(ids) {
    const s = ids.length;
    const C = this.cols;
    const out = Tensor.zeros([s, C]);
    if (this.backend === 'webgl') {
      const gl = getGL();
      const idData = new Float32Array(s);
      for (let i = 0; i < s; i++) idData[i] = ids[i];
      const idsTex = gl.createTexture((s + 3) >> 2, 1, GLBackend.padData(idData, 1, s));
      try {
        if (this.transposedStorage) {
          gl.run('gatherT', out.tex, out._texW, out._texH,
            [['T', this.tex], ['IDS', idsTex]], { iC: C });
        } else {
          gl.run('gather', out.tex, out._texW, out._texH,
            [['W', this.tex], ['IDS', idsTex]], { iC: C });
        }
      } finally {
        gl.disposeTexture(idsTex);
      }
    } else {
      out.data.set(CK.cpuEmbedding(this.data, C, ids, s));
    }
    return out;
  }

  /** out = this[:, off : off+len]  -> [rows, len] */
  sliceCols(off, len) {
    const out = Tensor.zeros([this.rows, len]);
    if (this.backend === 'webgl') {
      getGL().run('sliceCols', out.tex, out._texW, out._texH,
        [['A', this.tex]], { iOff: off, iLen: len });
    } else {
      out.data.set(CK.cpuSliceCols(this.data, this.rows, this.cols, off, len));
    }
    return out;
  }

  /** Clear to zeros (GPU needs explicit clear before partial scatter). */
  clear() {
    if (this.backend === 'webgl') {
      getGL().run('fill', this.tex, this._texW, this._texH, [], { uVal: 0 });
    } else {
      this.data.fill(0);
    }
    return this;
  }

  /** dst op: write src [rows, len] into columns [off, off+len). Preserves the rest. */
  scatterCols(src, off, len) {
    if (this.backend === 'webgl') {
      getGL().run('scatterCols', this.tex, this._texW, this._texH,
        [['A', src.tex]], { iOff: off, iLen: len });
    } else {
      CK.cpuScatterCols(this.data, src.data, this.rows, this.cols, off, len);
    }
    return this;
  }

  /** dst op: write src row 0 into row `row` of dst (dst is a growing KV-cache). */
  scatterRow(src, row) {
    if (this.backend === 'webgl') {
      getGL().run('scatterRow', this.tex, this._texW, this._texH,
        [['A', src.tex]], { iRow: row });
    } else {
      CK.cpuAppendRow(this.data, this.cols, src.data, this.cols, row);
    }
    // set logical rows to the valid prefix length
    this.shape[0] = row + 1;
    return this;
  }
}


// ───────────────────────────── src/safetensors.js ─────────────────────────────
/**
 * mini.js — safetensors file parser, implemented from the published format spec:
 *   8 bytes  : little-endian u64 header length N
 *   N bytes  : JSON header: { "tensor.name": {dtype, shape, data_offsets:[start,end]}, __metadata__ }
 *   rest     : raw buffer data
 * No dependencies.
 */


const DTYPES = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  I64: 8, I32: 4, I16: 2, I8: 1,
  U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

function parseSafetensors(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 8) throw new Error('[mini.js] safetensors file too small');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  if (headerLen <= 0 || 8 + headerLen > bytes.length) {
    throw new Error('[mini.js] safetensors header length invalid: ' + headerLen);
  }
  const headerBytes = bytes.subarray(8, 8 + headerLen);
  const headerText = new TextDecoder('utf-8').decode(headerBytes);
  const header = parseJSON(headerText, 'safetensors header');

  const tensors = new Map();
  const dataStart = 8 + headerLen;
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const { dtype, shape, data_offsets } = info;
    if (!DTYPES[dtype]) throw new Error(`[mini.js] unsupported dtype "${dtype}" for tensor ${name}`);
    const [s, e] = data_offsets;
    const count = shape.reduce((a, b) => a * b, 1);
    const absStart = dataStart + s;
    const absEnd = dataStart + e;
    if (absEnd > bytes.length) throw new Error(`[mini.js] tensor ${name} out of bounds`);
    const f32 = bytesToFloat32(bytes, dtype, count, absStart);
    tensors.set(name, { dtype, shape, data: f32, byteStart: absStart, byteEnd: absEnd });
  }
  return { header, tensors };
}

/** Write a safetensors file from {name: {shape, Float32Array}} — used by tests/tools. */
function writeSafetensors(entries, metadata = null) {
  const enc = new TextEncoder();
  const header = {};
  let offset = 0;
  const blobs = [];
  for (const [name, e] of entries) {
    const bytes = new Uint8Array(e.data.buffer, e.data.byteOffset, e.data.byteLength);
    header[name] = { dtype: 'F32', shape: e.shape.slice(), data_offsets: [offset, offset + bytes.length] };
    blobs.push(bytes);
    offset += bytes.length;
  }
  if (metadata) header.__metadata__ = metadata;
  const headerText = JSON.stringify(header);
  const headerBytes = enc.encode(headerText);
  // pad header to 8-byte alignment with spaces (per spec recommendation)
  let pad = (8 - ((8 + headerBytes.length) % 8)) % 8;
  const out = new Uint8Array(8 + headerBytes.length + pad + offset);
  const dv = new DataView(out.buffer);
  dv.setBigUint64(0, BigInt(headerBytes.length + pad), true);
  out.set(headerBytes, 8);
  for (let i = 0; i < pad; i++) out[8 + headerBytes.length + i] = 0x20;
  let pos = 8 + headerBytes.length + pad;
  for (const b of blobs) { out.set(b, pos); pos += b.length; }
  return out;
}


// ───────────────────────────── src/tokenizer.js ─────────────────────────────
/**
 * mini.js — Tokenizers written from scratch:
 *   1. Byte-level BPE (GPT-2 / GPT-Neo / RoBERTa family) from tokenizer.json
 *   2. WordPiece (BERT / DistilBERT family) from tokenizer.json
 * Supports special-token parsing, added_tokens, encode + decode round trips.
 * No dependencies. No tokenizers.rs. No sentencepiece. Pure JS.
 */


// ---------------------------------------------------------------------------
// GPT-2 byte <-> unicode map (the famous printable-rotation trick)
// ---------------------------------------------------------------------------
function buildByteEncoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);   // ! .. ~
  for (let i = 161; i <= 172; i++) bs.push(i);  // ¡ .. ¬
  for (let i = 174; i <= 255; i++) bs.push(i);  // ® .. ÿ
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const encoder = new Map();
  for (let i = 0; i < bs.length; i++) encoder.set(bs[i], String.fromCharCode(cs[i]));
  return encoder;
}
const BYTE_ENCODER = buildByteEncoder();

function textToByteChars(text) {
  const bytes = new TextEncoder().encode(text);
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += BYTE_ENCODER.get(bytes[i]);
  return s;
}

/** Reverse of the byte map, used at decode time. */
const BYTE_DECODER = (() => {
  const m = new Map();
  for (const [b, c] of BYTE_ENCODER) m.set(c, b);
  return m;
})();

function byteCharsToText(chars) {
  const bytes = new Uint8Array(chars.length);
  for (let i = 0; i < chars.length; i++) bytes[i] = BYTE_DECODER.get(chars[i]) ?? 63;
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
}

// ---------------------------------------------------------------------------
// Pre-tokenizer regexes (from the tokenizer.json spec, compiled natively)
// ---------------------------------------------------------------------------
const GPT2_SPLIT_PATTERN =
  /( 's|'t|'re|'ve|'m|'ll|'d)|('s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
// note: GPT-2's exact pattern (contractions anchored without leading space alternation issues)
const GPT2_PATTERN_EXACT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function splitWithRegex(text, pattern) {
  const out = [];
  let last = 0;
  pattern.lastIndex = 0;
  let m;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(m[0]);
    last = m.index + m[0].length;
    if (m[0].length === 0) pattern.lastIndex++; // safety
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** BERT basic tokenizer: clean, lowercase, strip accents, split punct & CJK. */
function bertBasicTokenize(text, { lowercase = true, stripAccents = true } = {}) {
  // NFD + remove combining marks
  let t = text.normalize('NFD');
  if (stripAccents) t = t.replace(/\p{M}+/gu, '');
  if (lowercase) t = t.toLowerCase();
  t = t.replace(/\s+/g, ' ').trim();
  if (!t) return [];
  const out = [];
  // split on whitespace, then isolate punctuation and CJK chars
  for (const chunk of t.split(' ')) {
    if (!chunk) continue;
    let cur = '';
    for (const ch of chunk) {
      const cp = ch.codePointAt(0);
      const isCJK =
        (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
        (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x3040 && cp <= 0x30ff) ||
        (cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0xf900 && cp <= 0xfaff);
      const isPunct = /[\p{P}\p{S}]/u.test(ch);
      if (isCJK || isPunct) {
        if (cur) { out.push(cur); cur = ''; }
        out.push(ch);
      } else {
        cur += ch;
      }
    }
    if (cur) out.push(cur);
  }
  return out;
}

// ---------------------------------------------------------------------------
// BPE merge engine
// ---------------------------------------------------------------------------
class BPE {
  constructor(vocab, merges) {
    this.vocab = vocab;                 // Map token string -> id
    this.ranks = new Map();             // "a\u0000b" -> rank
    for (let i = 0; i < merges.length; i++) {
      const parts = merges[i].split(' ');
      if (parts.length === 2) this.ranks.set(parts[0] + '\u0000' + parts[1], i);
    }
  }

  /** Merge one word (string of byte-chars) into vocab tokens. */
  encodeWord(word) {
    if (word.length === 0) return [];
    let parts = Array.from(word);
    if (parts.length === 1) {
      const id = this.vocab.get(parts[0]);
      return id === undefined ? [] : [id];
    }
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + '\u0000' + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx === -1) break;
      parts = parts.slice(0, bestIdx).concat([parts[bestIdx] + parts[bestIdx + 1]], parts.slice(bestIdx + 2));
    }
    const ids = [];
    for (const p of parts) {
      const id = this.vocab.get(p);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }
}

// ---------------------------------------------------------------------------
// WordPiece engine
// ---------------------------------------------------------------------------
class WordPiece {
  constructor(vocab, { unkToken = '[UNK]', maxChars = 100, prefix = '##' } = {}) {
    this.vocab = vocab;
    this.unk = unkToken in vocab ? vocab[unkToken] : (vocab.get ? undefined : 0);
    this.unkId = vocab instanceof Map ? (vocab.get(unkToken) ?? 0) : (vocab[unkToken] ?? 0);
    this.maxChars = maxChars;
    this.prefix = prefix;
  }

  encodeWord(word) {
    if (word.length > this.maxChars) return [this.unkId];
    const ids = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let curId = null;
      while (start < end) {
        let sub = word.slice(start, end);
        if (start > 0) sub = this.prefix + sub;
        const id = this.vocab instanceof Map ? this.vocab.get(sub) : this.vocab[sub];
        if (id !== undefined) { curId = id; break; }
        end--;
      }
      if (curId === null) return [this.unkId];
      ids.push(curId);
      start = end;
    }
    return ids;
  }
}

// ---------------------------------------------------------------------------
// Unified tokenizer
// ---------------------------------------------------------------------------
class Tokenizer {
  /**
   * Pass either a parsed tokenizer.json object, or the raw JSON text.
   * kind: 'bpe' (byte-level) or 'wordpiece' — auto-detected from the file if absent.
   */
  constructor(tokenizerJson, tokenizerConfig = {}, kind = null) {
    const tj = typeof tokenizerJson === 'string'
      ? parseJSON(tokenizerJson, 'tokenizer.json')
      : tokenizerJson;
    this.json = tj;
    this.config = tokenizerConfig || {};

    const modelType = (tj.model && tj.model.type) || '';
    if (!kind) {
      if (modelType === 'BPE' || modelType === 'WordPiece') {
        kind = modelType === 'BPE' ? 'bpe' : 'wordpiece';
      } else {
        kind = 'bpe';
      }
    }
    this.kind = kind;

    // vocab: Map(token -> id)
    const rawVocab = tj.vocab || (tj.model && tj.model.vocab);
    if (!rawVocab) throw new Error('[mini.js] tokenizer.json has no vocab');
    let vocabObj;
    if (Array.isArray(rawVocab)) {
      vocabObj = {};
      rawVocab.forEach((tok, i) => { vocabObj[tok] = i; });
    } else {
      vocabObj = rawVocab;
    }
    this.vocab = new Map(Object.entries(vocabObj).map(([k, v]) => [k, v]));
    this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));

    if (kind === 'bpe') {
      const merges = (tj.model && tj.model.merges) || [];
      const norm = merges.map(m => Array.isArray(m) ? m.join(' ') : m);
      this.bpe = new BPE(this.vocab, norm);
    } else {
      const mc = (tj.model && tj.model) || {};
      this.wordpiece = new WordPiece(this.vocab, {
        unkToken: mc.unk_token || this.config.unk_token || '[UNK]',
        maxChars: mc.max_input_chars_per_word || 100,
        prefix: mc.continuing_subword_prefix || '##',
      });
    }

    // added / special tokens: id -> info
    this.specialIds = new Set();
    this.specialTokenIds = {};  // name -> id
    const added = tj.added_tokens || [];
    this.addedTokens = new Map(); // content -> id
    for (const at of added) {
      this.addedTokens.set(at.content, at.id);
      if (at.special) this.specialIds.add(at.id);
    }
    // config-declared specials
    for (const key of ['unk_token', 'bos_token', 'eos_token', 'pad_token', 'sep_token', 'cls_token', 'mask_token']) {
      const t = this.config[key];
      const content = typeof t === 'string' ? t : (t && t.content);
      if (content && this.addedTokens.has(content)) {
        const id = this.addedTokens.get(content);
        this.specialTokenIds[key] = id;
        this.specialIds.add(id);
      } else if (content && this.vocab.has(content)) {
        this.specialTokenIds[key] = this.vocab.get(content);
      }
    }
    // normalizer info for wordpiece — read structured fields, not string sniffing
    const normz = (tj.normalizer && typeof tj.normalizer === 'object') ? tj.normalizer : {};
    if (this.config.do_lower_case !== undefined) this.lowercase = !!this.config.do_lower_case;
    else if (normz.lowercase !== undefined) this.lowercase = !!normz.lowercase;
    else this.lowercase = this.kind === 'wordpiece'; // BERT-family default: uncased
    this.stripAccents = this.config.strip_accents !== undefined
      ? !!this.config.strip_accents
      : true;
  }

  get bosTokenId() { return this.specialTokenIds.bos_token; }
  get eosTokenId() { return this.specialTokenIds.eos_token; }
  get clsTokenId() { return this.specialTokenIds.cls_token; }
  get sepTokenId() { return this.specialTokenIds.sep_token; }
  get padTokenId() { return this.specialTokenIds.pad_token; }
  get maskTokenId() { return this.specialTokenIds.mask_token; }
  get unkTokenId() { return this.specialTokenIds.unk_token; }

  /** Split text on special tokens, returning [piece, isSpecial] segments. */
  _splitSpecial(text) {
    const segments = [];
    let rest = text;
    const specials = [...this.addedTokens.keys()].sort((a, b) => b.length - a.length);
    while (rest.length) {
      let found = null, foundIdx = -1;
      for (const s of specials) {
        const i = rest.indexOf(s);
        if (i !== -1 && (found === null || i < foundIdx)) { found = s; foundIdx = i; }
      }
      if (found === null) { segments.push([rest, false]); break; }
      if (foundIdx > 0) segments.push([rest.slice(0, foundIdx), false]);
      segments.push([found, true]);
      rest = rest.slice(foundIdx + found.length);
    }
    return segments;
  }

  encode(text, { addSpecialTokens = false } = {}) {
    const ids = [];
    for (const [seg, isSpecial] of this._splitSpecial(text)) {
      if (isSpecial) {
        ids.push(this.addedTokens.get(seg));
        continue;
      }
      if (this.kind === 'bpe') {
        const mapped = textToByteChars(seg);
        for (const word of splitWithRegex(mapped, GPT2_PATTERN_EXACT)) {
          for (const id of this.bpe.encodeWord(word)) ids.push(id);
        }
      } else {
        for (const word of bertBasicTokenize(seg, { lowercase: this.lowercase, stripAccents: this.stripAccents })) {
          for (const id of this.wordpiece.encodeWord(word)) ids.push(id);
        }
      }
    }
    if (addSpecialTokens && this.kind === 'wordpiece') {
      if (this.clsTokenId !== undefined && this.sepTokenId !== undefined) {
        return [this.clsTokenId, ...ids, this.sepTokenId];
      }
    }
    return ids;
  }

  decode(ids, { skipSpecialTokens = true } = {}) {
    if (this.kind === 'bpe') {
      let chars = '';
      for (const id of ids) {
        if (skipSpecialTokens && this.specialIds.has(id)) continue;
        const tok = this.idToToken.get(id);
        if (tok === undefined) continue;
        chars += tok;
      }
      return byteCharsToText(chars);
    }
    // wordpiece
    const words = [];
    let cur = '';
    for (const id of ids) {
      if (skipSpecialTokens && this.specialIds.has(id)) continue;
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      if (tok.startsWith('##')) cur += tok.slice(2);
      else {
        if (cur) words.push(cur);
        cur = tok;
      }
    }
    if (cur) words.push(cur);
    return words.join(' ');
  }

  /** tokens as strings (for display) */
  tokenize(text) {
    return this.encode(text).map(id => this.idToToken.get(id) ?? '<unk>');
  }

  /**
   * Apply the model's chat template to a conversation.
   * messages: [{role:'system'|'user'|'assistant', content}, ...]
   * Uses the repo's real Jinja template (subset renderer) when present,
   * otherwise a builtin ChatML / Llama-2 / Llama-3 / Gemma template.
   */
  applyChatTemplate(messages, { addGenerationPrompt = true, modelType = null } = {}) {
    return buildChatPrompt(messages, {
      template: this.chatTemplate,
      modelType: modelType || this.modelType,
      addGenerationPrompt,
    });
  }
}

/** Convenience: build from fetched files. */
function createTokenizer(tokenizerJsonText, tokenizerConfigObj, modelType) {
  let kind = null;
  const mt = (modelType || '').toLowerCase();
  if (mt === 'gpt2' || mt === 'gpt_neox' || mt === 'gptj' || mt === 'codegen' || mt === 'roberta' || mt === 'bloom') kind = 'bpe';
  else if (mt === 'bert' || mt === 'distilbert' || mt === 'albert') kind = 'wordpiece';
  return new Tokenizer(tokenizerJsonText, tokenizerConfigObj, kind);
}


// ───────────────────────────── src/chat-template.js ─────────────────────────────
/**
 * mini.js — chat templates, from scratch.
 * Renders the Jinja2 `chat_template` that real models ship inside
 * tokenizer_config.json (subset: for/if/elif/else, set, comparisons,
 * is defined / is not defined, not/and/or, string concat with +,
 * loop.first/last/index0/index1, path & index access, |length, |tojson,
 * whitespace control with {%- -%} / {{- -}}).
 * Falls back to builtin family templates (ChatML, Llama-2, Llama-3, Gemma)
 * when a repo has no template or the template uses unsupported syntax.
 */

class TemplateError extends Error {}

// ─────────────────────────────── expression parser ───────────────────────────
class ExprParser {
  constructor(text) { this.s = text; this.i = 0; }
  ws() { while (this.i < this.s.length && /\s/.test(this.s[this.i])) this.i++; }
  peek() { this.ws(); return this.s[this.i]; }
  eat(tok) {
    this.ws();
    if (this.s.startsWith(tok, this.i)) { this.i += tok.length; return true; }
    return false;
  }
  expect(tok) { if (!this.eat(tok)) throw new TemplateError(`expected "${tok}" in "${this.s}"`); }

  parse() {
    let left = this.parseOr();
    return left;
  }
  parseOr() {
    let left = this.parseAnd();
    while (true) {
      this.ws();
      if (this.s.startsWith('or', this.i) && /[\s(]/.test(this.s[this.i + 2] || ' ')) {
        this.i += 2;
        const right = this.parseAnd();
        left = { t: 'or', a: left, b: right };
      } else return left;
    }
  }
  parseAnd() {
    let left = this.parseNot();
    while (true) {
      this.ws();
      if (this.s.startsWith('and', this.i) && /[\s(]/.test(this.s[this.i + 3] || ' ')) {
        this.i += 3;
        const right = this.parseNot();
        left = { t: 'and', a: left, b: right };
      } else return left;
    }
  }
  parseNot() {
    this.ws();
    if (this.s.startsWith('not ', this.i)) {
      this.i += 4;
      return { t: 'not', a: this.parseNot() };
    }
    return this.parseCompare();
  }
  parseCompare() {
    let left = this.parseAdd();
    while (true) {
      this.ws();
      let op = null;
      for (const cand of ['==', '!=', '>=', '<=', '>', '<']) {
        if (this.s.startsWith(cand, this.i)) { op = cand; break; }
      }
      if (!op) return left;
      this.i += op.length;
      const right = this.parseAdd();
      left = { t: 'cmp', op, a: left, b: right };
    }
  }
  parseAdd() {
    let left = this.parseTest();
    while (true) {
      this.ws();
      if (this.s[this.i] === '+') {
        this.i++;
        left = { t: 'concat', a: left, b: this.parseTest() };
      } else return left;
    }
  }
  parseTest() {
    let left = this.parsePostfix();
    this.ws();
    const m = /^is\s+(not\s+)?(defined|string|number|mapping)/.exec(this.s.slice(this.i));
    if (m) {
      this.i += m[0].length;
      return { t: 'test', neg: !!m[1], kind: m[2], a: left };
    }
    return left;
  }
  parsePostfix() {
    let base = this.parsePrimary();
    while (true) {
      if (this.s[this.i] === '.') {
        this.i++;
        const name = this.ident();
        base = { t: 'get', a: base, b: { t: 'lit', v: name } };
      } else if (this.s[this.i] === '[') {
        this.i++;
        const idx = this.parse();
        this.expect(']');
        base = { t: 'get', a: base, b: idx };
      } else if (this.s[this.i] === '|') {
        this.i++;
        const f = this.ident();
        base = { t: 'filter', f, a: base };
      } else return base;
    }
  }
  ident() {
    this.ws();
    const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(this.s.slice(this.i));
    if (!m) throw new TemplateError(`expected identifier at "${this.s.slice(this.i, this.i + 12)}"`);
    this.i += m[0].length;
    return m[0];
  }
  parsePrimary() {
    this.ws();
    const c = this.s[this.i];
    if (c === "'" || c === '"') {
      const q = c; this.i++;
      let out = '';
      while (this.i < this.s.length && this.s[this.i] !== q) {
        if (this.s[this.i] === '\\') { this.i++; }
        out += this.s[this.i++];
      }
      this.i++; // closing quote
      return { t: 'lit', v: out };
    }
    if (/[0-9]/.test(c)) {
      const m = /^[0-9]+(\.[0-9]+)?/.exec(this.s.slice(this.i));
      this.i += m[0].length;
      return { t: 'lit', v: Number(m[0]) };
    }
    if (this.s.startsWith('true', this.i)) { this.i += 4; return { t: 'lit', v: true }; }
    if (this.s.startsWith('false', this.i)) { this.i += 5; return { t: 'lit', v: false }; }
    if (this.s.startsWith('none', this.i)) { this.i += 4; return { t: 'lit', v: null }; }
    return { t: 'var', name: this.ident() };
  }
}

function parseExpr(text) { return new ExprParser(text).parse(); }

// ─────────────────────────────── evaluator ───────────────────────────────────
function evalExpr(node, ctx) {
  switch (node.t) {
    case 'lit': return node.v;
    case 'var': {
      if (!(node.name in ctx)) throw new TemplateError('undefined variable: ' + node.name);
      return ctx[node.name];
    }
    case 'get': {
      const obj = evalExpr(node.a, ctx);
      const key = evalExpr(node.b, ctx);
      if (obj === null || obj === undefined) throw new TemplateError('cannot index ' + obj);
      return obj[key];
    }
    case 'concat': {
      const a = evalExpr(node.a, ctx), b = evalExpr(node.b, ctx);
      return String(a ?? '') + String(b ?? '');
    }
    case 'cmp': {
      const a = evalExpr(node.a, ctx), b = evalExpr(node.b, ctx);
      switch (node.op) {
        case '==': return a === b || (a == null && b == null);
        case '!=': return !(a === b || (a == null && b == null));
        case '>': return a > b; case '<': return a < b;
        case '>=': return a >= b; case '<=': return a <= b;
      }
      return false;
    }
    case 'and': return !!(evalExpr(node.a, ctx)) && !!(evalExpr(node.b, ctx));
    case 'or': return !!(evalExpr(node.a, ctx)) || !!(evalExpr(node.b, ctx));
    case 'not': return !evalExpr(node.a, ctx);
    case 'test': {
      let v;
      try { v = evalExpr(node.a, ctx); } catch (_) { v = undefined; }
      let res;
      if (node.kind === 'defined') res = v !== undefined;
      else if (node.kind === 'string') res = typeof v === 'string';
      else if (node.kind === 'number') res = typeof v === 'number';
      else res = v !== null && typeof v === 'object';
      return node.neg ? !res : res;
    }
    case 'filter': {
      const v = evalExpr(node.a, ctx);
      if (node.f === 'length') return v == null ? 0 : (v.length ?? Object.keys(v).length);
      if (node.f === 'tojson') return JSON.stringify(v);
      if (node.f === 'trim') return String(v).trim();
      if (node.f === 'lower') return String(v).toLowerCase();
      if (node.f === 'upper') return String(v).toUpperCase();
      throw new TemplateError('unsupported filter: ' + node.f);
    }
    default: throw new TemplateError('unknown expr node ' + node.t);
  }
}

// ─────────────────────────────── template parser ─────────────────────────────
/**
 * Parses a jinja-ish template into a node tree:
 *   {t:'text', v} {t:'out', expr} {t:'for', varName, list, body}
 *   {t:'if', clauses:[{cond, body}], else: body|null} {t:'set', name, expr}
 *   {t:'raise', msg}
 */
function parseTemplate(raw) {
  // normalize whitespace control: {%- / -%} / {{- / -}}
  const src = raw.replace(/\{%-/g, '{%').replace(/-%\}/g, '%}')
                 .replace(/\{\{-/g, '{{').replace(/-\}\}/g, '}}');
  const TAG_RE = /\{%([\s\S]+?)%\}|\{\{([\s\S]+?)\}\}/g;

  // find next tag at or after `from`
  function nextTag(from) {
    TAG_RE.lastIndex = from;
    const m = TAG_RE.exec(src);
    if (!m) return null;
    return {
      start: m.index,
      end: m.index + m[0].length,
      type: m[2] !== undefined ? 'out' : 'tag',
      inner: (m[2] !== undefined ? m[2] : m[1]).trim(),
    };
  }

  function parseNodes(stopTags) {
    const body = [];
    while (true) {
      const tag = nextTag(i);
      if (!tag) {
        if (stopTags) throw new TemplateError('unclosed block, expected ' + stopTags.join('/'));
        if (i < src.length) body.push({ t: 'text', v: src.slice(i) });
        i = src.length;
        return { body, stopTag: null };
      }
      if (tag.start > i) body.push({ t: 'text', v: src.slice(i, tag.start) });
      i = tag.end;
      if (tag.type === 'out') {
        body.push({ t: 'out', expr: parseExpr(tag.inner) });
        continue;
      }
      const word = (/^[A-Za-z_][A-Za-z0-9_]*/.exec(tag.inner) || [''])[0];
      if (stopTags && stopTags.includes(word)) {
        return { body, stopTag: tag.inner };
      }
      if (word === 'for') {
        const fm = /^for\s+([A-Za-z_]\w*)\s+in\s+([\s\S]+)$/.exec(tag.inner);
        if (!fm) throw new TemplateError('bad for tag: ' + tag.inner);
        const sub = parseNodes(['endfor']);
        body.push({ t: 'for', varName: fm[1], list: parseExpr(fm[2]), body: sub.body });
      } else if (word === 'if') {
        const cond = parseExpr(tag.inner.slice(2).trim());
        const sub = parseNodes(['elif', 'else', 'endif']);
        const clauses = [{ cond, body: sub.body }];
        let elseBody = null;
        let cur = sub;
        while (cur.stopTag && cur.stopTag.startsWith('elif')) {
          const cond2 = parseExpr(cur.stopTag.slice(4).trim());
          const sub2 = parseNodes(['elif', 'else', 'endif']);
          clauses.push({ cond: cond2, body: sub2.body });
          cur = sub2;
        }
        if (cur.stopTag === 'else') {
          const sub2 = parseNodes(['endif']);
          elseBody = sub2.body;
        }
        body.push({ t: 'if', clauses, else: elseBody });
      } else if (word === 'set') {
        const sm = /^set\s+([A-Za-z_]\w*)\s*=\s*([\s\S]+)$/.exec(tag.inner);
        if (!sm) throw new TemplateError('bad set tag: ' + tag.inner);
        body.push({ t: 'set', name: sm[1], expr: parseExpr(sm[2]) });
      } else if (word === 'raise_exception') {
        const rm = /^raise_exception\(([\s\S]+)\)$/.exec(tag.inner);
        body.push({ t: 'raise', msg: rm ? rm[1].replace(/^['"]|['"]$/g, '') : 'template error' });
      } else {
        throw new TemplateError('unsupported tag: ' + word);
      }
    }
  }

  let i = 0;
  const tree = parseNodes(null);
  return tree.body;
}

function runNodes(nodes, ctx, out) {
  for (const n of nodes) {
    if (n.t === 'text') out.push(n.v);
    else if (n.t === 'out') out.push(String(evalExpr(n.expr, ctx) ?? ''));
    else if (n.t === 'set') ctx[n.name] = evalExpr(n.expr, ctx);
    else if (n.t === 'raise') throw new TemplateError(n.msg);
    else if (n.t === 'for') {
      const list = evalExpr(n.list, ctx);
      if (!Array.isArray(list)) continue;
      const nItems = list.length;
      for (let i = 0; i < nItems; i++) {
        ctx[n.varName] = list[i];
        ctx.loop = { first: i === 0, last: i === nItems - 1, index0: i, index1: i + 1 };
        runNodes(n.body, ctx, out);
      }
    } else if (n.t === 'if') {
      let taken = false;
      for (const cl of n.clauses) {
        if (evalExpr(cl.cond, ctx)) { runNodes(cl.body, ctx, out); taken = true; break; }
      }
      if (!taken && n.else) runNodes(n.else, ctx, out);
    }
  }
}

/** Main entry: build the final prompt string for a chat conversation.
 * Expressions must be fully consumed — a trailing unparsed fragment (like an
 * unsupported function call) is a subset miss → TemplateError → builtin fallback.
 */
function renderChatTemplate(template, vars) {
  const tree = parseTemplate(template);
  const ctx = Object.assign({}, vars);
  const out = [];
  runNodes(tree, ctx, out);
  return out.join('');
}

// ─────────────────────────────── builtin families ────────────────────────────
function detectChatFamily(templateStr, modelType) {
  const t = templateStr || '';
  if (t.includes('<|start_header_id|>')) return 'llama3';
  if (t.includes('<|im_start|>')) return 'chatml';
  if (t.includes('<start_of_turn>')) return 'gemma';
  if (t.includes('[INST]')) return 'llama2';
  const mt = (modelType || '').toLowerCase();
  if (mt === 'qwen2' || mt === 'qwen3') return 'chatml';
  if (mt === 'llama') return 'llama3';
  if (mt === 'gemma' || mt === 'gemma2' || mt === 'gemma3') return 'gemma';
  if (mt === 'mistral') return 'llama2';
  return 'chatml';
}

function builtinTemplate(family) {
  switch (family) {
    case 'llama3':
      return "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
    case 'llama2':
      return "{% for m in messages %}{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]{% elif m['role'] == 'assistant' %}{{ m['content'] }}</s>{% elif m['role'] == 'system' %}<<SYS>> {{ m['content'] }} <</SYS>>{% endif %}{% endfor %}";
    case 'gemma':
      return "{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";
    case 'chatml':
    default:
      return "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
  }
}

/**
 * Main entry: build the final prompt string for a chat conversation.
 * messages: [{role: 'system'|'user'|'assistant', content: string}, ...]
 */
function buildChatPrompt(messages, { template = null, modelType = null, addGenerationPrompt = true } = {}) {
  const family = detectChatFamily(template, modelType);
  const vars = {
    messages,
    add_generation_prompt: addGenerationPrompt,
    bos_token: '', eos_token: '',
    development: 'none',
  };
  if (template) {
    try {
      return renderChatTemplate(template, vars);
    } catch (e) {
      // template used syntax outside our subset → family builtin
      console.warn('[mini.js] chat template subset miss (' + e.message + ') — using builtin ' + family + ' template');
    }
  }
  return renderChatTemplate(builtinTemplate(family), vars);
}


// ───────────────────────────── src/models.js ─────────────────────────────
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
class GPT2Model {
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
class LlamaModel {
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
class BertModel {
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
class DistilBertModel {
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
class RobertaModel {
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
function classificationHead(model, weights, config, hidden) {
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
function tokenClassificationHead(model, weights, hidden) {
  const cw = W(weights, 'classifier.weight'), cb = W(weights, 'classifier.bias');
  return hidden.matmul(cw, { transB: true }).addRowBias(cb);
}

/** QA span head: hidden [s,d] -> logits [s,2] (start/end). */
function qaHead(model, weights, hidden) {
  const prefix = model.kind === 'distilbert' ? '' : '';
  const cw = weights.get('qa_outputs.weight') || weights.get('qa_outputs.weight');
  if (!cw) throw new Error('[mini.js] no qa_outputs head found');
  const cb = weights.get('qa_outputs.bias');
  let t = hidden.matmul(cw, { transB: true });
  if (cb) t = t.addRowBias(cb);
  return t;
}

function mlmHead(model, weights, config, hidden) {
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


// ───────────────────────────── src/generation.js ─────────────────────────────
/**
 * mini.js — Text generation engine: greedy / temperature / top-k / top-p sampling,
 * repetition penalty, EOS stopping, stop-callbacks, KV-cache stepping.
 * `generateTokens` is fully synchronous (Node-friendly); `generateAsync`
 * yields to the event loop between tokens (keeps browser UIs responsive).
 */


function softmaxTopNothing(logits) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); sum += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

function applyRepetitionPenalty(logits, generatedIds, penalty) {
  const seen = new Set(generatedIds);
  for (const id of seen) {
    const v = logits[id];
    logits[id] = v > 0 ? v / penalty : v * penalty;
  }
}

function sampleTopKTopP(probs, { topK = 0, topP = 0 } = {}) {
  const V = probs.length;
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  let cand = idx;
  if (topK && topK > 0 && topK < V) cand = idx.slice(0, topK);
  if (topP && topP < 1) {
    let acc = 0, cut = cand.length;
    for (let i = 0; i < cand.length; i++) {
      acc += probs[cand[i]];
      if (acc >= topP) { cut = i + 1; break; }
    }
    cand = cand.slice(0, Math.max(1, cut));
  }
  let total = 0;
  for (const i of cand) total += probs[i];
  let r = Math.random() * total;
  for (const i of cand) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return cand[cand.length - 1];
}

function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

function normalizeOptions(options) {
  return {
    maxNewTokens: options.maxNewTokens ?? options.max_new_tokens ?? 50,
    doSample: options.doSample ?? options.do_sample ?? true,
    temperature: options.temperature ?? 1.0,
    topK: options.topK ?? options.top_k ?? 50,
    topP: options.topP ?? options.top_p ?? 0.95,
    repetitionPenalty: options.repetitionPenalty ?? options.repetition_penalty ?? 1.0,
    eosTokenId: options.eosTokenId ?? options.eos_token_id ?? null,
    onToken: options.onToken || null,
  };
}

/** Pick the next token from last-token logits. Returns [tokenId, keep]. */
function pickToken(lastLogits, generated, inputIds, opts) {
  const work = Float32Array.from(lastLogits);
  if (opts.repetitionPenalty && opts.repetitionPenalty !== 1.0 && generated.length) {
    applyRepetitionPenalty(work, inputIds.concat(generated), opts.repetitionPenalty);
  }
  if (!opts.doSample || opts.temperature <= 0.0001) return [argmax(work), true];
  if (opts.temperature !== 1.0) {
    for (let i = 0; i < work.length; i++) work[i] /= opts.temperature;
  }
  return [sampleTopKTopP(softmaxTopNothing(work), { topK: opts.topK, topP: opts.topP }), true];
}

function lastTokenLogits(model, inputIds, cache) {
  const logits = arenaRun(() => model.forward(inputIds, { cache, lastOnly: true }));
  let last = logits.read();
  if (logits.rows !== 1) last = last.subarray((logits.rows - 1) * logits.cols);
  else if (last.length !== logits.size) last = last.subarray(0, logits.cols);
  logits.dispose();
  return last;
}

/**
 * Synchronous generation. options: maxNewTokens, doSample, temperature, topK,
 * topP, repetitionPenalty, eosTokenId, onToken(id, n) — returning `false` from
 * onToken stops generation. Returns array of generated token ids.
 */
function generateTokens(model, inputIds, options = {}) {
  const opts = normalizeOptions(options);
  const cache = model.makeCache();
  try {
    const generated = [];
    let lastLogits = lastTokenLogits(model, inputIds, cache);
    for (let step = 0; step < opts.maxNewTokens; step++) {
      const [nextId] = pickToken(lastLogits, generated, inputIds, opts);
      generated.push(nextId);
      if (opts.onToken && opts.onToken(nextId, step + 1) === false) break;
      if (opts.eosTokenId !== null && nextId === opts.eosTokenId) break;
      lastLogits = lastTokenLogits(model, [nextId], cache);
    }
    return generated;
  } finally {
    model.disposeCache(cache);
  }
}

/**
 * Async generation — identical API, but yields to the event loop between
 * tokens (via `tick`, default setTimeout 0) so browser UIs stay responsive
 * and onToken can update the DOM mid-generation.
 */
async function generateAsync(model, inputIds, options = {}) {
  const opts = normalizeOptions(options);
  const tick = options.tick || (typeof window !== 'undefined'
    ? () => new Promise(r => setTimeout(r, 0))
    : null);
  const cache = model.makeCache();
  try {
    const generated = [];
    let lastLogits = lastTokenLogits(model, inputIds, cache);
    for (let step = 0; step < opts.maxNewTokens; step++) {
      const [nextId] = pickToken(lastLogits, generated, inputIds, opts);
      generated.push(nextId);
      if (opts.onToken && opts.onToken(nextId, step + 1) === false) break;
      if (opts.eosTokenId !== null && nextId === opts.eosTokenId) break;
      if (tick) await tick();
      lastLogits = lastTokenLogits(model, [nextId], cache);
    }
    return generated;
  } finally {
    model.disposeCache(cache);
  }
}


// ───────────────────────────── src/hub.js ─────────────────────────────
/**
 * mini.js — Model Hub: streams files from the Hugging Face Hub with progress,
 * caches them in the browser Cache API (repeat runs are instant), no deps.
 * Supports sharded safetensors models (model.safetensors.index.json).
 */



const HF_BASE = 'https://huggingface.co';

const env = {
  hfBase: HF_BASE,
  useBrowserCache: true,   // cache downloaded files (Cache API)
  cacheName: 'mini-js-models-v1',
};

function fileUrl(repo, filename, revision = 'main') {
  return `${env.hfBase}/${repo}/resolve/${revision}/${filename}`;
}

async function openCache() {
  if (!env.useBrowserCache) return null;
  try {
    if (typeof caches !== 'undefined' && caches.open) return await caches.open(env.cacheName);
  } catch (_) { /* private mode etc. */ }
  return null;
}

/**
 * Download (or fetch from cache) one file as Uint8Array.
 * progressCallback receives {status, file, loaded, total, progress}.
 */
async function downloadFile(repo, filename, { progressCallback = null, revision = 'main' } = {}) {
  const emitter = new ProgressEmitter(progressCallback);
  const url = fileUrl(repo, filename, revision);
  emitter.status('initiate', filename, { url });

  const cache = await openCache();
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        emitter.status('cached', filename, { loaded: buf.length, total: buf.length });
        return buf;
      }
    } catch (_) { /* ignore cache errors */ }
  }

  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) {
    throw new Error(`[mini.js] failed to download ${repo}/${filename} — HTTP ${res.status}` +
      (res.status === 401 ? ' (repository is gated; skipping)' : ''));
  }

  let bytes;
  if (res.body && res.body.getReader) {
    const total = Number(res.headers.get('content-length') || 0);
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    let lastEmit = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      const now = Date.now();
      if (now - lastEmit > 100 || loaded === total) { // throttle to ~10 events/sec
        emitter.progress(filename, loaded, total || 0);
        lastEmit = now;
      }
    }
    emitter.progress(filename, loaded, loaded);
    bytes = concatBytes(chunks, loaded);
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    emitter.progress(filename, bytes.length, bytes.length);
  }

  if (cache) {
    try {
      await cache.put(url, new Response(bytes.slice(), {
        headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(bytes.length) },
      }));
    } catch (_) { /* quota exceeded — ignore */ }
  }
  emitter.status('done', filename, { loaded: bytes.length, total: bytes.length });
  return bytes;
}

async function fetchJSON(repo, filename) {
  const bytes = await downloadFile(repo, filename);
  return JSON.parse(new TextDecoder('utf-8').decode(bytes));
}

/**
 * Download everything a repo needs: config, tokenizer files, weights.
 * Supports single `model.safetensors` AND sharded models via
 * `model.safetensors.index.json` (weight_map → multiple shard files).
 * Returns {config, tokenizerConfig, tokenizerJson, tensors: Map, weightFiles}
 */
async function downloadModel(repo, { progressCallback = null, weightFile = null } = {}) {
  const files = {};
  const grabJSON = async (name) => {
    try { files[name] = await fetchJSON(repo, name); } catch (_) { files[name] = null; }
  };
  await Promise.all([
    grabJSON('config.json'),
    grabJSON('tokenizer_config.json'),
    grabJSON('special_tokens_map.json'),
  ]);

  const dlOpts = { progressCallback };

  // ── sharded safetensors? ──
  let index = null;
  try { index = await fetchJSON(repo, 'model.safetensors.index.json'); } catch (_) { /* single-file */ }

  let tensors = null;
  const weightFiles = [];
  if (index && index.weight_map) {
    const shardNames = [...new Set(Object.values(index.weight_map))];
    for (const shard of shardNames) {
      const bytes = await downloadFile(repo, shard, dlOpts);
      const parsed = parseSafetensors(bytes);
      if (!tensors) tensors = parsed.tensors;
      else for (const [k, v] of parsed.tensors) tensors.set(k, v);
      weightFiles.push(shard);
    }
    // sanity: every mapped tensor must exist
    for (const name of Object.keys(index.weight_map)) {
      if (!tensors.has(name)) throw new Error(`[mini.js] shard index references missing tensor "${name}"`);
    }
  } else {
    const candidates = weightFile ? [weightFile] : ['model.safetensors'];
    let lastErr = null;
    for (const f of candidates) {
      try {
        const bytes = await downloadFile(repo, f, dlOpts);
        tensors = parseSafetensors(bytes).tensors;
        weightFiles.push(f);
        break;
      } catch (e) { lastErr = e; }
    }
    if (!tensors) {
      throw lastErr || new Error('[mini.js] no safetensors weights found in ' + repo);
    }
  }

  // tokenizer.json (may be missing on very old repos)
  try { files['tokenizer.json'] = await fetchJSON(repo, 'tokenizer.json'); }
  catch (_) { files['tokenizer.json'] = null; }

  return {
    config: files['config.json'],
    tokenizerConfig: files['tokenizer_config.json'] || files['special_tokens_map.json'] || {},
    tokenizerJson: files['tokenizer.json'],
    tensors,
    weightFiles,
  };
}

async function clearCache() {
  const cache = await openCache();
  if (cache) await cache.keys().then(keys => Promise.all(keys.map(k => cache.delete(k))));
  return true;
}


// ───────────────────────────── src/pipelines.js ─────────────────────────────
/**
 * mini.js — High-level pipelines, Transformers.js-style but 100% from scratch:
 *   const gen = await mini.pipeline('text-generation', 'HuggingFaceTB/SmolLM2-135M-Instruct');
 *   const chat = await gen.chat([{ role: 'user', content: 'Hello!' }]);
 *
 * Tasks: text-generation (+chat), feature-extraction, text-classification,
 * fill-mask, question-answering, token-classification.
 */







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
class TextGenerationPipeline extends PipelineBase {
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
class FeatureExtractionPipeline extends PipelineBase {
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
class TextClassificationPipeline extends PipelineBase {
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
class TokenClassificationPipeline extends PipelineBase {
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
class FillMaskPipeline extends PipelineBase {
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
class QuestionAnsweringPipeline extends PipelineBase {
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
async function pipeline(task, repo, options = {}) {
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
function ready() {
  detectBestBackend();
  return currentBackend();
}

function disposePipeline(task, repo) {
  const key = `${task}::${repo}`;
  const p = pipelineCache.get(key);
  if (p) { p.dispose(); pipelineCache.delete(key); }
}


// ───────────────────────────── src/selftest.js ─────────────────────────────
/**
 * mini.js — runtime GPU self-test.
 * Runs every WebGL kernel on small known inputs and compares against the
 * hand-written CPU kernels. If ANY check fails, callers should fall back to
 * the CPU backend — so users never see silently-wrong AI output.
 */
const CK2 = { zeroFill: zeroFill, cpuMatmul: cpuMatmul, cpuAddBiasRows: cpuAddBiasRows, cpuAdd: cpuAdd, cpuMulRow: cpuMulRow, cpuScale: cpuScale, geluScalar: geluScalar, cpuGelu: cpuGelu, cpuRelu: cpuRelu, cpuSilu: cpuSilu, cpuSigmoid: cpuSigmoid, cpuSoftmaxRows: cpuSoftmaxRows, cpuLayerNorm: cpuLayerNorm, cpuEmbedding: cpuEmbedding, cpuSliceCols: cpuSliceCols, cpuScatterCols: cpuScatterCols, cpuAppendRow: cpuAppendRow, cpuRowOf: cpuRowOf, cpuRMSNorm: cpuRMSNorm, cpuMul: cpuMul, cpuRope: cpuRope };


function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randF32(n, seed, scale = 2) {
  const r = rng(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (r() * 2 - 1) * scale;
  return out;
}

function close(a, b, tol = 2e-4) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const s = Math.abs(a[i]) + Math.abs(b[i]) + 1;
    if (d > tol * s) return false;
  }
  return true;
}

/**
 * Verify the active backend. Returns
 *   { backend, ok, failed: [names], checks: n }
 * If backend === 'webgl' and !ok, do: mini.setBackend('cpu').
 */
function selfTest() {
  const backend = currentBackend();
  const failed = [];
  let checks = 0;
  const test = (name, fn) => {
    checks++;
    try { if (!fn()) failed.push(name); }
    catch (e) { failed.push(name + ' (' + e.message + ')'); }
  };

  const t = (shape, seed, opts) => new Tensor(shape, randF32(shape.reduce((a, b) => a * b, 1), seed), opts);
  const readOf = (fn) => arenaRun(fn).read();

  if (backend === 'cpu') {
    // reference sanity: CPU kernels against each other (fast)
    test('matmul', () => {
      const A = randF32(15, 1), B = randF32(18, 2);
      const out = new Float32Array(15);
      CK2.cpuMatmul(A, B, out, 3, 6, 3, false);
      return out.every(Number.isFinite);
    });
    return { backend, ok: failed.length === 0, failed, checks };
  }

  // ---- matmul NN (unaligned + aligned) ----
  test('matmulNN', () => {
    const M = 3, Kd = 5, N = 6;
    const A = t([M, Kd], 1), B = t([Kd, N], 2);
    const got = readOf(() => A.matmul(B));
    const ref = new Float32Array(M * N);
    CK2.cpuMatmul(A.read(), B.read(), ref, M, Kd, N, false);
    return close(got, ref);
  });

  test('matmulTN', () => {
    const M = 2, Kd = 7, N = 5;
    const A = t([M, Kd], 3), B = t([N, Kd], 4);
    const got = readOf(() => A.matmul(B, { transB: true }));
    const ref = new Float32Array(M * N);
    CK2.cpuMatmul(A.read(), B.read(), ref, M, Kd, N, true);
    return close(got, ref);
  });

  test('matmulMT (transposed storage)', () => {
    const M = 2, Kd = 6, N = 9;
    const A = t([M, Kd], 5);
    const logical = randF32(N * Kd, 6); // B [N, K]
    const transposed = new Float32Array(N * Kd);
    for (let n = 0; n < N; n++) for (let k = 0; k < Kd; k++) transposed[k * N + n] = logical[n * Kd + k];
    const Bt = new Tensor([N, Kd], transposed, { transposedStorage: true });
    const got = readOf(() => A.matmul(Bt));
    const ref = new Float32Array(M * N);
    CK2.cpuMatmul(A.read(), logical, ref, M, Kd, N, true);
    return close(got, ref);
  });

  test('softmax (scaled)', () => {
    const rows = 3, cols = 10;
    const A = t([rows, cols], 7);
    const got = readOf(() => A.softmax(0.5));
    const ref = A.read().slice();
    CK2.cpuSoftmaxRows(ref, rows, cols, 0.5);
    return close(got, ref, 5e-4);
  });

  test('layernorm', () => {
    const rows = 3, cols = 10;
    const A = t([rows, cols], 8);
    const g = new Tensor([cols], randF32(cols, 9).map(v => 1 + v * 0.2));
    const b = new Tensor([cols], randF32(cols, 10).map(v => v * 0.1));
    const got = readOf(() => A.layernorm(g, b, 1e-5));
    const ref = CK2.cpuLayerNorm(A.read(), rows, cols, g.read(), b.read(), 1e-5);
    return close(got, ref, 5e-4);
  });

  test('gelu', () => {
    const A = t([2, 7], 11);
    const got = readOf(() => A.gelu());
    return close(got, CK2.cpuGelu(A.read()), 5e-4);
  });

  test('rmsnorm (Llama family)', () => {
    const rows = 3, cols = 10;
    const A = t([rows, cols], 30);
    const g = new Tensor([cols], randF32(cols, 31).map(v => 1 + v * 0.2));
    const got = readOf(() => A.rmsnorm(g, 1e-5));
    const ref = CK2.cpuRMSNorm(A.read(), rows, cols, g.read(), 1e-5);
    return close(got, ref, 5e-4);
  });

  test('mul (elementwise)', () => {
    const A = t([2, 9], 32), B = t([2, 9], 33);
    const got = readOf(() => A.mul(B));
    const ref = CK2.cpuMul(A.read(), B.read());
    return close(got, ref);
  });

  test('rope (Llama RoPE)', () => {
    const A = t([3, 12], 34); // 3 tokens, 12 cols = 2 heads x headDim 6
    const got = readOf(() => A.rope(5, 6, 10000.0));
    const ref = CK2.cpuRope(A.read(), 3, 12, 6, 5, 10000.0);
    return close(got, ref, 5e-4);
  });

  test('add / addRowBias / scale', () => {
    const A = t([2, 6], 12), B = t([2, 6], 13);
    const bias = new Tensor([6], randF32(6, 14));
    const g1 = readOf(() => A.add(B));
    const r1 = CK2.cpuAdd(A.read(), B.read(), 12);
    const g2 = readOf(() => A.addRowBias(bias));
    const aCopy = A.read().slice();
    CK2.cpuAddBiasRows(aCopy, bias.read(), 2, 6);
    const g3 = readOf(() => A.scale(1.5));
    const r3 = CK2.cpuScale(A.read(), 1.5);
    return close(g1, r1) && close(g2, aCopy) && close(g3, r3);
  });

  test('gather + gatherT (embedding)', () => {
    const V = 11, C = 6;
    const W = t([V, C], 15);
    const ids = [3, 10, 0, 7];
    const got = readOf(() => W.gather(ids));
    const ref = CK2.cpuEmbedding(W.read(), C, ids, 4);
    if (!close(got, ref)) return false;
    const logical = randF32(V * C, 16);
    const transposed = new Float32Array(V * C);
    for (let v = 0; v < V; v++) for (let c = 0; c < C; c++) transposed[c * V + v] = logical[v * C + c];
    const WT = new Tensor([V, C], transposed, { transposedStorage: true });
    const got2 = readOf(() => WT.gather(ids));
    return close(got2, ref);
  });

  test('sliceCols (unaligned)', () => {
    const A = t([3, 8], 17);
    const got = readOf(() => A.sliceCols(1, 5));
    const ref = CK2.cpuSliceCols(A.read(), 3, 8, 1, 5);
    return close(got, ref);
  });

  test('scatterCols / scatterRow / pasteRows', () => {
    const dst = Tensor.zeros([3, 9]);
    const src = t([3, 4], 18);
    arenaRun(() => { dst.clear(); dst.scatterCols(src, 5, 4); });
    const d1 = dst.read();
    const s1 = src.read();
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
      if (d1[r * 9 + 5 + c] !== s1[r * 4 + c]) return false;
    }
    const row = t([1, 9], 19);
    arenaRun(() => dst.scatterRow(row, 2));
    const d2 = dst.read();
    const rowRef = row.read();
    for (let c = 0; c < 9; c++) if (d2[2 * 9 + c] !== rowRef[c]) return false;
    const block = t([2, 9], 20);
    arenaRun(() => dst.pasteRows(block, 1));
    const d3 = dst.read();
    const bRef = block.read();
    for (let c = 0; c < 18; c++) if (d3[9 + c] !== bRef[c]) return false;
    return true;
  });

  test('rowSlice', () => {
    const A = t([4, 7], 21);
    const got = readOf(() => A.rowSlice(2));
    const ref = A.read().slice(2 * 7, 3 * 7);
    return close(got, ref);
  });

  test('tanh', () => {
    const A = t([2, 5], 22);
    const got = readOf(() => A.tanh_());
    const a = A.read(), ref = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) ref[i] = Math.tanh(a[i]);
    return close(got, ref, 5e-4);
  });

  return { backend, ok: failed.length === 0, failed, checks };
}

/** Pick the best backend, verifying WebGL numerics; fall back to CPU if bad. */
function initBackend() {
  if (typeof document === 'undefined') { setBackend('cpu'); return 'cpu'; }
  try {
    setBackend('webgl');
    const res = selfTest();
    if (res.ok) return 'webgl';
    console.warn('[mini.js] GPU self-test failed (' + res.failed.join(', ') + ') — falling back to CPU');
  } catch (e) {
    console.warn('[mini.js] WebGL2 unavailable (' + (e && e.message) + ') — using CPU backend');
  }
  setBackend('cpu');
  return 'cpu';
}


// ───────────────────────────── src/index.js ─────────────────────────────
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

var __EXPORT_DEFAULT__ = mini;

// expose single entry object (index.js already assigns globalThis.mini too)
if (typeof __EXPORT_DEFAULT__ !== 'undefined' && __EXPORT_DEFAULT__) {
  globalThis.mini = __EXPORT_DEFAULT__;
  if (typeof module !== 'undefined' && module.exports) module.exports = __EXPORT_DEFAULT__;
}
})(typeof globalThis !== 'undefined' ? globalThis : this);
