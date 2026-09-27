/**
 * mini.js — Tensor: a minimal eager tensor with two from-scratch backends.
 *  - 'cpu'   : Float32Array kernels (always available, incl. Node.js)
 *  - 'webgl' : WebGL2 RGBA32F textures + hand-written GLSL kernels (browser)
 *
 * A tensor is logically [ ...leading rows, cols ]; ops treat it as 2D [rows, cols].
 * Data is always contiguous row-major Float32Array on CPU; on GPU it is a padded
 * texture (row stride rounded up to 4 floats).
 */
import { getGL, GLBackend } from './webgl-backend.js';
import * as CK from './cpu-kernels.js';

let ACTIVE_BACKEND = null; // 'cpu' | 'webgl'

export function setBackend(name) {
  if (name === 'webgl') { getGL(); ACTIVE_BACKEND = 'webgl'; }
  else if (name === 'cpu') ACTIVE_BACKEND = 'cpu';
  else throw new Error('[mini.js] unknown backend: ' + name);
  return ACTIVE_BACKEND;
}

export function detectBestBackend() {
  if (typeof document !== 'undefined') {
    try { setBackend('webgl'); return 'webgl'; } catch (_) { /* fall through */ }
  }
  setBackend('cpu');
  return 'cpu';
}

export function currentBackend() {
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
export function arenaRun(fn) {
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

export class Tensor {
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
