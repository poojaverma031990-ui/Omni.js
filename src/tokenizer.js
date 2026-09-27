/**
 * mini.js — Tokenizers written from scratch:
 *   1. Byte-level BPE (GPT-2 / GPT-Neo / RoBERTa family) from tokenizer.json
 *   2. WordPiece (BERT / DistilBERT family) from tokenizer.json
 * Supports special-token parsing, added_tokens, encode + decode round trips.
 * No dependencies. No tokenizers.rs. No sentencepiece. Pure JS.
 */
import { parseJSON } from './utils.js';

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
export class Tokenizer {
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
export function createTokenizer(tokenizerJsonText, tokenizerConfigObj, modelType) {
  let kind = null;
  const mt = (modelType || '').toLowerCase();
  if (mt === 'gpt2' || mt === 'gpt_neox' || mt === 'gptj' || mt === 'codegen' || mt === 'roberta' || mt === 'bloom') kind = 'bpe';
  else if (mt === 'bert' || mt === 'distilbert' || mt === 'albert') kind = 'wordpiece';
  return new Tokenizer(tokenizerJsonText, tokenizerConfigObj, kind);
}
