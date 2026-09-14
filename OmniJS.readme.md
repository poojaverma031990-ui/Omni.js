# omni.js - 100% From Scratch Transformer in Pure JS

A complete, self-contained generative Transformer built entirely in vanilla JavaScript with **ZERO dependencies**.

- **No ONNX Runtime** (`onnxruntime-web`)
- **No Hugging Face** transformers / CDNs
- **No PyTorch / TensorFlow**
- **No Import Maps or external downloads**
- **Runs anywhere**: Desktop browsers, mobile Chrome, Node.js, and static hosts like GitHub Pages.

Every single layer, matrix multiplication, softmax vector pass, causal multi-head attention routine, RMSNorm layer, and backpropagation step is coded directly from raw math.

---

## Architecture Overview

1. **`Matrix`**:
   - Continuous flat `Float32Array` backend.
   - Cache-friendly linear indexing for transposed and standard dot-products (`Matrix.matmul`).
   - Numerically stabilized row-wise `Matrix.softmax` (subtracting max logit to avoid overflow).
   - Box-Muller Gaussian normal initialization (`Matrix.randn`).

2. **`Tokenizer`**:
   - Zero-dependency character-level tokenizer.
   - Dynamically inspects any arbitrary input corpus and builds token-to-id and id-to-token bi-directional maps.

3. **`MultiHeadAttention`**:
   - True multi-head causal self-attention.
   - Projections: $Q = XW_q$, $K = XW_k$, $V = XW_v$.
   - Split heads with scaled dot-product attention: $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$ where $M_{i,j} = -\infty$ for $j > i$.
   - Output projection $W_o$ and analytical backpropagation for all head weights.

4. **`Transformer`**:
   - Learned Token Embeddings + Learned Positional Embeddings.
   - RMSNorm (Root Mean Square Normalization) with learnable scale parameters ($\gamma$).
   - Multi-Head Causal Attention with residual connections.
   - 2-Layer FeedForward Network with ReLU non-linearity.
   - Linear Language Modeling head producing vocabulary logits.
   - Full analytical backward pass and built-in **Adam optimizer** ($\beta_1=0.9, \beta_2=0.999$) running directly in the browser.

5. **`Omni`**:
   - Factory and persistence coordinator.
   - Save and load model weights to/from JSON strings without external binaries.

---

## Quickstart

### In the Browser

No build step required. Just serve `index.html` on any static HTTP server or GitHub Pages:

```bash
npx serve .
