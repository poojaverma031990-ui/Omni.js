#!/usr/bin/env node
/**
 * mini.js build script — a tiny from-scratch ES-module bundler (no deps).
 * Concatenates src/*.js (in dependency order) into one IIFE file: mini.js,
 * which exposes `window.mini` / `globalThis.mini` and works as a
 * CommonJS require in Node.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const SRC = join(ROOT, 'src');
const OUT = join(ROOT, 'mini.js');

const ORDER = [
  'utils.js',
  'cpu-kernels.js',
  'webgl-backend.js',
  'tensor.js',
  'safetensors.js',
  'tokenizer.js',
  'chat-template.js',
  'models.js',
  'generation.js',
  'hub.js',
  'pipelines.js',
  'selftest.js',
  'index.js',
];

/** Extract exported top-level names of a module (raw source). */
function exportsOf(code) {
  const names = new Set();
  const re = /export\s+(?:async\s+)?(?:function|class)\s+(\w+)|export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g;
  let m;
  while ((m = re.exec(code)) !== null) names.add(m[1] || m[2]);
  return names;
}

function transform(code, file, exportsByFile) {
  const importRe = /import\s+(?:\{([^}]*)\}|\*\s*as\s+(\w+)|(\w+))\s*from\s*['"]([^'"]+)['"];?/g;
  const imported = new Set();
  code = code.replace(importRe, (_, braces, ns, def, src) => {
    if (braces) {
      for (const part of braces.split(',')) {
        const p = part.trim();
        if (!p) continue;
        const asMatch = p.match(/(\w+)\s+as\s+(\w+)/);
        imported.add(asMatch ? asMatch[2] : p);
      }
    }
    if (ns) {
      const srcFile = src.replace('./', '');
      const names = exportsByFile.get(srcFile);
      if (!names || names.size === 0) throw new Error(`cannot resolve namespace import * as ${ns} from ${src}`);
      imported.add(ns);
      return `const ${ns} = { ${[...names].map(n => `${n}: ${n}`).join(', ')} };`;
    }
    if (def) imported.add(def);
    return '';
  });

  code = code.replace(/export\s+default\s+/g, 'var __EXPORT_DEFAULT__ = ');
  code = code.replace(/export\s*\*\s*from\s*['"][^'"]+['"];?/g, '');
  code = code.replace(/export\s*\*;?/g, '');
  code = code.replace(/export\s+\{[^}]*\}\s*from\s*['"][^'"]+['"];?/g, '');
  code = code.replace(/export\s+\{[^}]*\};?/g, '');
  code = code.replace(/export\s+(async\s+)?(function|class|const|let|var)\s+/g, '$1$2 ');

  return { code, imported, file };
}

const exportsByFile = new Map();
for (const f of ORDER) exportsByFile.set(f, exportsOf(readFileSync(join(SRC, f), 'utf8')));

const parts = [];
const allImported = new Map();
const allDefined = new Map();

for (const f of ORDER) {
  const raw = readFileSync(join(SRC, f), 'utf8');
  const { code, imported, file } = transform(raw, f, exportsByFile);
  for (const name of imported) allImported.set(name, file);
  const defRe = /^(?:async\s+)?(?:function|class)\s+(\w+)|^(?:const|let|var)\s+([A-Za-z_$][\w$]*)/gm;
  let m;
  while ((m = defRe.exec(code)) !== null) {
    const name = m[1] || m[2];
    if (allDefined.has(name)) throw new Error(`top-level name collision: "${name}" defined in ${allDefined.get(name)} and ${file}`);
    allDefined.set(name, file);
  }
  parts.push(`// ───────────────────────────── src/${file} ─────────────────────────────\n${code}`);
}

for (const [name, file] of allImported) {
  if (!allDefined.has(name)) throw new Error(`import "${name}" in ${file} is never defined`);
}

const banner = `/*!
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
`;

const footer = `
// expose single entry object (index.js already assigns globalThis.mini too)
if (typeof __EXPORT_DEFAULT__ !== 'undefined' && __EXPORT_DEFAULT__) {
  globalThis.mini = __EXPORT_DEFAULT__;
  if (typeof module !== 'undefined' && module.exports) module.exports = __EXPORT_DEFAULT__;
}
})(typeof globalThis !== 'undefined' ? globalThis : this);
`;

let bundle = banner + parts.join('\n\n') + footer;
// make the engine safe to inline into HTML: neutralize literal script-closing tags
// (a JS comment containing "<\/script" is inert, but ends nothing in HTML)
bundle = bundle.replace(/<\/script/g, '<\\/script');
writeFileSync(OUT, bundle);
const kb = (Buffer.byteLength(bundle) / 1024).toFixed(1);
console.log(`✓ built mini.js (${kb} KB, ${ORDER.length} modules)`);
