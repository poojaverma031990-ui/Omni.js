/**
 * mini.js — runtime GPU self-test.
 * Runs every WebGL kernel on small known inputs and compares against the
 * hand-written CPU kernels. If ANY check fails, callers should fall back to
 * the CPU backend — so users never see silently-wrong AI output.
 */
import * as CK2 from './cpu-kernels.js';
import { Tensor, arenaRun, currentBackend, setBackend } from './tensor.js';

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
export function selfTest() {
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
export function initBackend() {
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
