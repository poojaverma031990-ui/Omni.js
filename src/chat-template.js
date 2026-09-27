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
export function renderChatTemplate(template, vars) {
  const tree = parseTemplate(template);
  const ctx = Object.assign({}, vars);
  const out = [];
  runNodes(tree, ctx, out);
  return out.join('');
}

// ─────────────────────────────── builtin families ────────────────────────────
export function detectChatFamily(templateStr, modelType) {
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
export function buildChatPrompt(messages, { template = null, modelType = null, addGenerationPrompt = true } = {}) {
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
