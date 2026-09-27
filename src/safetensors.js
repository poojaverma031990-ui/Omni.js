/**
 * mini.js — safetensors file parser, implemented from the published format spec:
 *   8 bytes  : little-endian u64 header length N
 *   N bytes  : JSON header: { "tensor.name": {dtype, shape, data_offsets:[start,end]}, __metadata__ }
 *   rest     : raw buffer data
 * No dependencies.
 */
import { bytesToFloat32, parseJSON } from './utils.js';

const DTYPES = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  I64: 8, I32: 4, I16: 2, I8: 1,
  U64: 8, U32: 4, U16: 2, U8: 1, BOOL: 1,
};

export function parseSafetensors(buffer) {
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
export function writeSafetensors(entries, metadata = null) {
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
