/**
 * mini.js — CPU kernels: hand-written numeric kernels on Float32Array.
 * Blocked / unrolled matmul, softmax, layernorm, gelu, attention helpers.
 * No dependencies.
 */

/** Zero-fill view helper. */
export function zeroFill(arr, len) { arr.fill(0, 0, len); }

/**
 * C[M,N] = A[M,K] x B (B stored K,N when transB=false, N,K when transB=true)
 * All row-major contiguous. out must be pre-zeroed or pre-filled with bias.
 */
export function cpuMatmul(a, b, out, M, K, N, transB) {
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
export function cpuAddBiasRows(x, bias, rows, cols) {
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let c = 0; c < cols; c++) x[off + c] += bias[c];
  }
}

/** Elementwise add of two same-shape arrays (returns new). */
export function cpuAdd(a, b, len) {
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = a[i] + b[i];
  return out;
}

/** Elementwise multiply (broadcast B as single row of len cols across rows). */
export function cpuMulRow(a, b, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let c = 0; c < cols; c++) out[off + c] = a[off + c] * b[c];
  }
  return out;
}

export function cpuScale(x, s) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * s;
  return out;
}

/** GELU with tanh approximation (the "gelu_new" used by GPT-2). */
export function geluScalar(x) {
  const x3 = x * x * x;
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x3)));
}

export function cpuGelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = geluScalar(x[i]);
  return out;
}

export function cpuRelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

export function cpuSilu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / (1 + Math.exp(-x[i]));
  return out;
}

export function cpuSigmoid(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
  return out;
}

/** Numerically stable row-wise softmax, in place. rows x cols. Optional scale. */
export function cpuSoftmaxRows(x, rows, cols, scale = 1) {
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
export function cpuLayerNorm(x, rows, cols, gamma, beta, eps) {
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
export function cpuEmbedding(weights, cols, ids, rows) {
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
export function cpuSliceCols(src, rows, cols, off, len) {
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
export function cpuScatterCols(dst, src, rows, totalCols, off, len) {
  for (let r = 0; r < rows; r++) {
    dst.set(src.subarray(r * len, r * len + len), r * totalCols + off);
  }
}

/** Copy one row of src [rows, cols] into dst row `row` of a preallocated [>=rows, cols] buffer. */
export function cpuAppendRow(dst, dstStride, src, srcLen, row) {
  dst.set(src.subarray(0, srcLen), row * dstStride);
}

/** Copy a single row of a strided buffer into a contiguous row array. */
export function cpuRowOf(strided, stride, row, cols) {
  return strided.subarray(row * stride, row * stride + cols);
}

/** RMSNorm: y = x * g / sqrt(mean(x^2) + eps). rows x cols, no bias. */
export function cpuRMSNorm(x, rows, cols, g, eps) {
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
export function cpuMul(a, b) {
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
export function cpuRope(x, rows, cols, headDim, pos0, theta) {
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
