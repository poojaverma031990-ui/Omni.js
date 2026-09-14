import { Omni } from './omni.js';

// 1. Instantiate the engine
const omni = await Omni.create({
  dModel: 32,
  nHeads: 2,
  nLayers: 1,
  maxSeqLen: 64
});

// 2. Train on sample sentences
const corpus = `hello world
how are you
omni transformer`;

omni.initVocab(corpus);

// 3. Train Step (Adam optimizer updates weights locally)
const tokens = omni.tokenizer.encode("hello world\n");
const inputs = tokens.slice(0, -1);
const targets = tokens.slice(1);

for (let i = 0; i < 100; i++) {
  const loss = omni.model.trainStep(inputs, targets, 0.02);
  if (i % 20 === 0) console.log(`Step ${i} Loss:`, loss);
}

// 4. Generate Text
const output = omni.model.generate("he", 20, omni.tokenizer, 0.5);
console.log("Output:", output);