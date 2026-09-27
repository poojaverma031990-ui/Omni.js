/* mini.js playground — vanilla JS, no frameworks */
(function () {
  'use strict';

  const $ = (s) => document.querySelector(s);

  // ─── task configs ───────────────────────────────────────────────
  const TASKS = {
    'text-generation': {
      label: 'Text generation',
      models: [
        { id: 'openai-community/gpt2', name: 'GPT-2 (124M) — the real deal', size: '~548 MB' },
        { id: 'distilbert/distilgpt2', name: 'DistilGPT-2 (82M)', size: '~355 MB' },
        { id: 'hf-internal-testing/tiny-random-gpt2', name: 'TinyGPT-2 — instant test', size: '<1 MB' },
      ],
      input: 'Once upon a time, in a land far away,',
      samples: [
        'Once upon a time, in a land far away,',
        'The best way to learn artificial intelligence is',
        'In the year 2030, humans will',
      ],
      defaults: { maxTokens: 40, temperature: 0.8, doSample: true },
    },
    'text-classification': {
      label: 'Sentiment',
      models: [
        { id: 'distilbert/distilbert-base-uncased-finetuned-sst-2-english', name: 'DistilBERT SST-2 (sentiment)', size: '~268 MB' },
      ],
      input: 'I absolutely loved this movie — the acting was incredible!',
      samples: [
        'I absolutely loved this movie — the acting was incredible!',
        'This was a complete waste of time and money.',
        'The plot was thin but the soundtrack made up for it.',
      ],
      defaults: {},
    },
    'feature-extraction': {
      label: 'Embeddings',
      models: [
        { id: 'sentence-transformers/all-MiniLM-L6-v2', name: 'all-MiniLM-L6-v2 (384-dim)', size: '~90 MB' },
      ],
      input: 'The quick brown fox jumps over the lazy dog',
      samples: [
        'The quick brown fox jumps over the lazy dog',
        'Artificial intelligence will reshape everything',
        'A good breakfast is the start of a good day',
      ],
      defaults: {},
    },
    'chat': {
      label: 'Chat',
      isChat: true,
      models: [
        { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', name: 'SmolLM2-135M-Instruct — real instruct LLM', size: '~269 MB' },
        { id: 'Qwen/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5-0.5B-Instruct', size: '~988 MB' },
      ],
      input: 'Say hello and tell me what you are.',
      samples: [
        'Say hello and tell me what you are.',
        'Explain gravity to me like I am five.',
        'Give me three quick tips for learning JavaScript.',
      ],
      defaults: {},
    },
    'question-answering': {
      label: 'QA',
      models: [
        { id: 'distilbert/distilbert-base-uncased-distilled-squad', name: 'DistilBERT SQuAD (extractive QA)', size: '~268 MB' },
      ],
      input: 'Which name is also used to describe the Amazon rainforest in English?\n---\nThe Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America.',
      samples: [
        'Which name is also used to describe the Amazon rainforest in English?\n---\nThe Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America.',
        'Who wrote the theory of general relativity?\n---\nAlbert Einstein developed the theory of relativity, which transformed theoretical physics and astronomy during the 20th century.',
      ],
      defaults: {},
    },
    'fill-mask': {
      label: 'Fill mask',
      models: [
        { id: 'distilbert/distilbert-base-uncased', name: 'DistilBERT (MLM)', size: '~268 MB' },
        { id: 'google-bert/bert-base-uncased', name: 'BERT base (MLM)', size: '~440 MB' },
      ],
      input: 'The capital of France is [MASK].',
      samples: [
        'The capital of France is [MASK].',
        'She went to the [MASK] to buy some milk.',
        'The doctor wrote me a [MASK].',
      ],
      defaults: {},
    },
  };

  let currentTask = 'text-generation';
  let loadedPipe = null;   // { key, pipe }
  let running = false;
  let tokenBuf = [];
  let convo = [];          // chat history
  let chatTranscript = ''; // rendered transcript text
  let stopFlag = false;

  const el = {
    tabs: $('#taskTabs'),
    modelSelect: $('#modelSelect'),
    inputText: $('#inputText'),
    runBtn: $('#runBtn'),
    sampleBtn: $('#sampleBtn'),
    status: $('#status'),
    progressWrap: $('#progressWrap'),
    progressTitle: $('#progressTitle'),
    progressFill: $('#progressFill'),
    progressMeta: $('#progressMeta'),
    outputArea: $('#outputArea'),
    outputTitle: $('#outputTitle'),
    outputBody: $('#outputBody'),
    timingBadge: $('#timingBadge'),
    maxTokField: $('#maxTokField'),
    tempField: $('#tempField'),
    sampleField: $('#sampleField'),
    maxTokens: $('#maxTokens'),
    temperature: $('#temperature'),
    doSample: $('#doSample'),
    copyOut: $('#copyOut'),
  };

  function fmtBytes(n) {
    if (!n || n <= 0) return '';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(n >= 100 || i === 0 ? 0 : 1) + ' ' + u[i];
  }

  function setStatus(msg, cls) {
    el.status.textContent = msg;
    el.status.className = 'status' + (cls ? ' ' + cls : '');
  }

  function renderModelOptions() {
    const cfg = TASKS[currentTask];
    el.modelSelect.innerHTML = cfg.models
      .map(m => `<option value="${m.id}">${m.name} · ${m.size}</option>`)
      .join('');
    const hasGenOpts = currentTask === 'text-generation';
    el.maxTokField.style.display = hasGenOpts ? '' : 'none';
    el.tempField.style.display = hasGenOpts ? '' : 'none';
    el.sampleField.style.display = hasGenOpts ? '' : 'none';
    if (hasGenOpts) {
      el.maxTokens.value = cfg.defaults.maxTokens;
      el.temperature.value = cfg.defaults.temperature;
      el.doSample.checked = cfg.defaults.doSample;
    }
    const labels = {
      'text-generation': 'Prompt',
      'text-classification': 'Text to classify',
      'feature-extraction': 'Text to embed',
      'fill-mask': 'Text with [MASK]',
      'chat': 'First message (conversation continues below)',
      'question-answering': 'Question, then a line with ---, then the context',
    };
    $('#inputLabel').textContent = labels[currentTask];
    el.inputText.value = cfg.input;
    hideOutput();
  }

  function hideOutput() {
    el.outputArea.hidden = true;
    el.outputBody.innerHTML = '';
    el.timingBadge.textContent = '';
  }

  // ─── progress UI ────────────────────────────────────────────────
  const fileState = new Map(); // file -> {loaded, total}
  function onProgress(p) {
    if (p.status === 'initiate') fileState.set(p.file, { loaded: 0, total: 0 });
    if (p.status === 'progress' || p.status === 'done' || p.status === 'cached') {
      fileState.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
    }
    let loaded = 0, total = 0, bigName = '';
    let bigLoaded = 0, bigTotal = 0;
    for (const [f, st] of fileState) {
      loaded += st.loaded; total += st.total;
      if (st.total > bigTotal) { bigTotal = st.total; bigLoaded = st.loaded; bigName = f; }
    }
    const pct = bigTotal > 0 ? Math.min(100, (bigLoaded / bigTotal) * 100) : 0;
    el.progressWrap.hidden = false;
    el.progressFill.style.width = pct.toFixed(1) + '%';
    const speed = p.speed ? ' · ' + fmtBytes(p.speed) + '/s' : '';
    el.progressTitle.textContent = total > 0
      ? `Downloading ${bigName} — ${pct.toFixed(0)}%`
      : 'Fetching model files…';
    el.progressMeta.textContent =
      `${fmtBytes(bigLoaded)} / ${fmtBytes(bigTotal)}${speed} — cached in your browser for next time`;
    setStatus('Downloading model… (one-time; then it is cached & offline)');
  }

  // ─── pipeline loading ───────────────────────────────────────────
  async function getPipe() {
    const repo = el.modelSelect.value;
    const key = currentTask + '::' + repo;
    if (loadedPipe && loadedPipe.key === key) return loadedPipe.pipe;
    if (loadedPipe) {
      try { loadedPipe.pipe.dispose(); } catch (_) { /* noop */ }
      loadedPipe = null;
      fileState.clear();
      el.progressWrap.hidden = false;
      el.progressFill.style.width = '0%';
    }
    const t0 = performance.now();
    const pipe = await mini.pipeline(currentTask, repo, { progress_callback: onProgress });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    loadedPipe = { key, pipe };
    el.progressWrap.hidden = true;
    setStatus(`Model ready in ${secs}s (cached for next time).`, 'ok');
    return pipe;
  }

  // ─── runners ────────────────────────────────────────────────────
  function showOutput(title) {
    el.outputArea.hidden = false;
    el.outputTitle.textContent = title;
    el.outputBody.innerHTML = '';
  }

  async function runGeneration(pipe, text) {
    showOutput('Generated text');
    const maxTokens = parseInt(el.maxTokens.value, 10) || 40;
    const temperature = parseFloat(el.temperature.value) || 1.0;
    const doSample = el.doSample.checked;
    const promptIds = pipe.tokenizer.encode(text);
    const ids = [];
    const t0 = performance.now();
    const [out] = await pipe(text, {
      max_new_tokens: maxTokens,
      do_sample: doSample,
      temperature,
      top_k: 50,
      top_p: 0.95,
      onToken: (id) => {
        ids.push(id);
        el.outputBody.textContent = pipe.tokenizer.decode(
          promptIds.concat(ids), { skipSpecialTokens: true });
        const cur = document.createElement('span');
        cur.className = 'cursor';
        el.outputBody.appendChild(cur);
      },
    });
    const secs = (performance.now() - t0) / 1000;
    const n = ids.length;
    el.outputBody.textContent = out.generated_text;
    el.timingBadge.textContent = `${n} tokens · ${secs.toFixed(1)}s · ${(n / Math.max(secs, 0.001)).toFixed(1)} tok/s`;
  }

  async function runChat(pipe, userMsg) {
    if (el.outputArea.hidden) { chatTranscript = ''; convo = []; }
    el.outputArea.hidden = false;
    el.outputTitle.textContent = 'Chat — ' + (pipe.repo || 'LLM');
    chatTranscript += (chatTranscript ? '\n\n' : '') + 'You: ' + userMsg + '\n\nAssistant: ';
    el.outputBody.textContent = chatTranscript;
    const cur = document.createElement('span');
    cur.className = 'cursor';
    el.outputBody.appendChild(cur);
    stopFlag = false;
    const t0 = performance.now();
    let nTokens = 0;
    const history = convo.concat([{ role: 'user', content: userMsg }]);
    const result = await pipe.chat(history, {
      max_new_tokens: parseInt(el.maxTokens.value, 10) || 120,
      temperature: parseFloat(el.temperature.value) || 0.7,
      onText: (delta, full) => {
        if (stopFlag) return false;
        nTokens++;
        el.outputBody.textContent = chatTranscript + full;
        el.outputBody.appendChild(cur);
      },
    });
    cur.remove();
    const reply = result[0].assistant_message;
    convo = history.concat([{ role: 'assistant', content: reply }]);
    chatTranscript += reply;
    el.outputBody.textContent = chatTranscript;
    const secs = (performance.now() - t0) / 1000;
    el.timingBadge.textContent = `${nTokens} tok · ${secs.toFixed(1)}s · ${(nTokens / Math.max(secs, 0.001)).toFixed(1)} tok/s`;
  }

  async function runQA(pipe, text) {
    const sep = text.indexOf('\n---\n');
    if (sep === -1) throw new Error('Enter the question, then a line with ---, then the context.');
    const question = text.slice(0, sep).trim();
    const context = text.slice(sep + 5).trim();
    showOutput('Answer');
    const t0 = performance.now();
    const [r] = await pipe({ question, context });
    const ms = performance.now() - t0;
    el.outputBody.innerHTML = '';
    const ans = document.createElement('div');
    ans.style.fontWeight = '700';
    ans.style.fontSize = '16px';
    ans.textContent = r.answer || '(no answer found)';
    const kv = document.createElement('div');
    kv.className = 'kv';
    kv.textContent = 'confidence: ' + (r.score * 100).toFixed(1) + '%';
    el.outputBody.appendChild(ans);
    el.outputBody.appendChild(kv);
    el.timingBadge.textContent = ms.toFixed(0) + ' ms';
  }

  async function runClassification(pipe, text) {
    showOutput('Classification');
    const t0 = performance.now();
    const results = await pipe(text);
    const ms = (performance.now() - t0).toFixed(0);
    const wrap = document.createElement('div');
    wrap.className = 'score-bars';
    for (const r of results.slice(0, 5)) {
      const row = document.createElement('div');
      row.className = 'score-row';
      row.innerHTML = `<div class="score-label"></div>
        <div class="score-track"><div class="score-fill" style="width:${(r.score * 100).toFixed(1)}%">${(r.score * 100).toFixed(2)}%</div></div>`;
      row.querySelector('.score-label').textContent = r.label;
      wrap.appendChild(row);
    }
    el.outputBody.appendChild(wrap);
    el.timingBadge.textContent = ms + ' ms';
  }

  async function runEmbedding(pipe, text) {
    showOutput('Embedding');
    const t0 = performance.now();
    const out = await pipe(text, { pooling: 'mean', normalize: true });
    const ms = (performance.now() - t0).toFixed(0);
    const v = out.data;
    // mini chart of first 64 dims
    const strip = document.createElement('div');
    strip.className = 'vec-strip';
    const maxAbs = Math.max(...Array.from(v.slice(0, 64)).map(Math.abs)) || 1;
    for (let i = 0; i < Math.min(64, v.length); i++) {
      const bar = document.createElement('div');
      bar.className = 'vec-bar ' + (v[i] >= 0 ? 'pos' : 'neg');
      bar.style.height = Math.max(3, (Math.abs(v[i]) / maxAbs) * 74) + 'px';
      bar.title = `dim ${i}: ${v[i].toFixed(4)}`;
      strip.appendChild(bar);
    }
    const meta = document.createElement('div');
    meta.className = 'kv';
    let norm = 0;
    for (const x of v) norm += x * x;
    meta.textContent = `dims: ${v.length} · L2 norm: ${Math.sqrt(norm).toFixed(4)} · first 64 dims shown`;
    el.outputBody.appendChild(strip);
    el.outputBody.appendChild(meta);
    el.timingBadge.textContent = ms + ' ms';
  }

  async function runFillMask(pipe, text) {
    if (!text.includes('[MASK]')) throw new Error('Input must contain [MASK]');
    showOutput('Mask predictions');
    const t0 = performance.now();
    const results = await pipe(text, { top_k: 5 });
    const ms = (performance.now() - t0).toFixed(0);
    const wrap = document.createElement('div');
    wrap.className = 'score-bars';
    for (const r of results) {
      const row = document.createElement('div');
      row.className = 'score-row';
      row.innerHTML = `<div class="score-label"></div>
        <div class="score-track"><div class="score-fill" style="width:${(r.score * 100).toFixed(1)}%">${(r.score * 100).toFixed(2)}%</div></div>`;
      row.querySelector('.score-label').textContent = r.token;
      row.title = r.sequence;
      wrap.appendChild(row);
    }
    el.outputBody.appendChild(wrap);
    const best = document.createElement('div');
    best.className = 'kv';
    best.textContent = 'best: ' + (results[0] ? results[0].sequence : '—');
    el.outputBody.appendChild(best);
    el.timingBadge.textContent = ms + ' ms';
  }

  async function run() {
    if (running) return;
    running = true;
    el.runBtn.disabled = true;
    el.runBtn.textContent = '⏳ Running…';
    hideOutput();
    try {
      const text = el.inputText.value.trim();
      if (!text) throw new Error('Please enter some text first.');
      const pipe = await getPipe();
      setStatus('Running on your device…', 'ok');
      if (currentTask === 'chat') await runChat(pipe, text);
      else if (currentTask === 'question-answering') await runQA(pipe, text);
      else if (currentTask === 'text-generation') await runGeneration(pipe, text);
      else if (currentTask === 'text-classification') await runClassification(pipe, text);
      else if (currentTask === 'feature-extraction') await runEmbedding(pipe, text);
      else if (currentTask === 'fill-mask') await runFillMask(pipe, text);
      setStatus('Done — computed 100% locally. ✔', 'ok');
    } catch (e) {
      console.error(e);
      setStatus((e && e.message ? e.message : String(e)), 'error');
    } finally {
      running = false;
      el.runBtn.disabled = false;
      el.runBtn.textContent = '▶ Run';
    }
  }

  // ─── events ─────────────────────────────────────────────────────
  el.tabs.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab');
    if (!btn) return;
    el.tabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    currentTask = btn.dataset.task;
    renderModelOptions();
  });
  el.runBtn.addEventListener('click', run);
  el.sampleBtn.addEventListener('click', () => {
    const cfg = TASKS[currentTask];
    el.inputText.value = cfg.samples[Math.floor(Math.random() * cfg.samples.length)];
  });
  el.inputText.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') run();
    if (e.key === 'Escape') stopFlag = true; // stops a streaming chat reply
  });
  el.copyOut.addEventListener('click', () => {
    navigator.clipboard.writeText(el.outputBody.textContent).then(() => {
      el.copyOut.textContent = 'Copied!';
      setTimeout(() => { el.copyOut.textContent = 'Copy'; }, 1200);
    });
  });
  document.querySelectorAll('[data-copy]').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(btn.dataset.copy).then(() => {
        const old = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = old; }, 1200);
      });
    });
  });

  // ─── backend badge + boot ───────────────────────────────────────
  async function boot() {
    const badge = $('#backendBadge');
    try {
      const backend = mini.initBackend();
      if (backend === 'webgl') {
        badge.textContent = '⚡ WebGL2 (GPU) — self-test passed';
        badge.classList.add('good');
      } else {
        badge.textContent = '🖥 CPU backend';
        badge.classList.add('good');
      }
    } catch (e) {
      badge.textContent = '⚠ backend error — see console';
      badge.classList.add('bad');
      console.error(e);
    }
    renderModelOptions();
    setStatus('Ready — pick a model and press Run. First run downloads the model once.');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
