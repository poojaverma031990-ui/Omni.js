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

export class GLBackend {
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
export function getGL() {
  if (_glInstance === null) {
    _glInstance = new GLBackend(); // may throw; caller decides fallback
  }
  return _glInstance;
}
export function resetGL() { _glInstance = null; }
