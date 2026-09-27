/**
 * mini.js — utils: dtype conversion, chunked download helpers, assertions.
 * 100% from scratch. No dependencies.
 */

export const MINI_VERSION = '2.4.0';

export function assert(cond, msg) {
  if (!cond) throw new Error('[mini.js] ' + msg);
}

export function assertDefined(x, name) {
  if (x === undefined || x === null) throw new Error('[mini.js] missing value: ' + name);
  return x;
}

/** Merge a list of Uint8Arrays into one. */
export function concatBytes(chunks, totalLen) {
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
export function bytesToFloat32(bytes, dtype, count, byteOffset = 0) {
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
export function halfToFloat(h) {
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

export function f32FromBits(bits) { f32Bits[0] = bits >>> 0; return f32Buf[0]; }

/** Safely parse JSON, with helpful error. */
export function parseJSON(text, what) {
  try { return JSON.parse(text); }
  catch (e) { throw new Error('[mini.js] failed to parse ' + what + ': ' + e.message); }
}

/** Tiny event emitter used for progress reporting. */
export class ProgressEmitter {
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
export function formatBytes(n) {
  if (!isFinite(n) || n < 0) return '?';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(n >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
}
