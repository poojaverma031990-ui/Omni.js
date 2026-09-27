#!/usr/bin/env node
/**
 * Builds demo.html — a TRUE single-file page: the entire mini.js engine is
 * inlined into the HTML, so this one file (opened anywhere, even via file://)
 * runs real transformer models with zero other files needed.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const template = readFileSync(join(ROOT, 'demo.template.html'), 'utf8');
const engine = readFileSync(join(ROOT, 'mini.js'), 'utf8');

if (!template.includes('/*__MINI_ENGINE__*/')) {
  throw new Error('demo.template.html is missing the /*__MINI_ENGINE__*/ placeholder');
}
// protect the inlined engine from containing the literal script-closing tag
if (engine.includes('</script')) throw new Error('engine contains </script — cannot inline safely');

const out = template.replace('/*__MINI_ENGINE__*/', () => engine);
writeFileSync(join(ROOT, 'demo.html'), out);
const kb = (Buffer.byteLength(out) / 1024).toFixed(1);
console.log(`✓ built demo.html (${kb} KB, single file with engine inlined)`);
