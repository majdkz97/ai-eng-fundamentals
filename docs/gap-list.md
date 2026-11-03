# Day 1 Diagnostic Gap List (Nov 3, 2026)

## Strong (Verified by Quiz Evidence)
- **None demonstrated in Block 1 quiz:** While experienced with production LLM agents at the application layer, low-level mathematical and PyTorch implementation fundamentals were not retrievable under direct questioning.
- **Environment & Tooling Execution (Block 2):** Quickly configured local CUDA environment (Windows 11, Python 3.13.7, PyTorch 2.14.1+cu126 on GTX 1660 Ti) and cloud GPU environment (Kaggle Tesla T4, PyTorch 2.11.0+cu130) with checkpoint persistence.

## Shaky (Verified by Quiz Evidence)
- **High-Level Conceptual Familiarity Only:** Recognizes terminology from LLM engineering, but cannot currently translate concepts into tensor shapes, equations, or PyTorch code without reference material.

## Missing (Verified by Quiz Evidence)
1. **Tensors, Shapes & Broadcasting:** Failed 3D broadcasting prediction ((4,1,6) + (3,1)), memory contiguity/strides (.transpose() breaking .view()), and 3D batched matrix multiplication shapes ((B, T, C) @ (C, H)).
2. **Autograd & Chain Rule:** Could not write the chain rule formula, compute a scalar polynomial derivative (z = 2x^2 + 1), or explain how torch.no_grad() prevents activation memory leaks in VRAM.
3. **Backprop by Hand:** Skipped computing partial derivatives of a single linear neuron (y = wx + b) and explaining why stacked linear layers collapse without non-linear activations.
4. **Loss Functions, Optimizers & MLP Loop:** Skipped loss mechanics, learning rate divergence/stalling, mini-batch gradient noise vs. throughput trade-offs, and the 4-step PyTorch training loop.
5. **Initialization, Normalization & Overfitting:** Skipped train/validation loss curve diagnostics and LayerNorm vs. BatchNorm dimension mechanics.
6. **Tokenization, Embeddings & Self-Attention Math:** Skipped BPE subword mechanics, token ID vs. embedding lookup table distinction, Q/K/V projections, 1/sqrt(d_k) scaling, and softmax/cross-entropy math.

## Top 3 Priorities to Fix This Week (Mapped to Nov 4 – Nov 9 Schedule)

1. **Priority 1 — Tensor Manipulation, Shapes, Strides & Autograd Mechanics**
   - Why first: Every line of deep learning code requires tracking tensor shapes ((B, T, C)), broadcasting without silent bugs, and knowing how PyTorch tracks gradients.
   - Schedule mapping:
     - **Nov 4 (Tensors/Autograd):** Master indexing, .view() vs .reshape(), .transpose(), right-to-left broadcasting rules, matmul (@) shapes, and requires_grad / torch.no_grad().
2. **Priority 2 — Chain Rule & Backpropagation by Hand**
   - Why second: You cannot debug vanishing/exploding gradients or dead neurons if .backward() is a black box.
   - Schedule mapping:
     - **Nov 5 (Micrograd Backprop):** Build a scalar autograd engine from scratch, manually derive local gradients for addition, multiplication, and tanh/ReLU, and verify against PyTorch .backward().
3. **Priority 3 — End-to-End MLP Training Loop, Stability & Language Modeling**
   - Why third: Connects tensors and gradients into an actual learning system with embeddings, cross-entropy loss, and optimization.
   - Schedule mapping:
     - **Nov 6 (MLP Training Loop):** Write batch slicing, forward pass, cross-entropy loss, optimizer.zero_grad(), loss.backward(), and optimizer.step() from scratch.
     - **Nov 7 (Init/Norm/LR/Overfitting):** Break and fix an MLP using bad weight initialization, high/low learning rates, and LayerNorm/BatchNorm; plot train vs. val loss.
     - **Nov 8 (Makemore Bigram + MLP):** Combine embeddings, matrix multiplications, and cross-entropy to train a character-level language model.
     - **Nov 9 (Redo Weakest Topic + Note):** Re-take today's missed quiz questions on Tensors, Broadcasting, and Autograd from memory and write a 1-page technical summary note.