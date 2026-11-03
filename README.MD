# AI Engineering Fundamentals (4-Week Plan)

Hands-on deep learning and PyTorch fundamentals built from scratch.

## 4-Week Plan Overview
- **Week 1 (Nov 3 - Nov 9): Core Mechanics & Backprop**
  - Nov 3: Setup & Gap Check (src/test_env.py, docs/gap-list.md)
  - Nov 4: Tensors, shapes, broadcasting, strides, and autograd
  - Nov 5: Build micrograd from scratch (manual backpropagation & computation graphs)
  - Nov 6: MLP training loop (zero_grad, backward, step, mini-batches, SGD vs Adam)
  - Nov 7: Weight initialization, LayerNorm vs BatchNorm, learning rate schedules, overfitting
  - Nov 8: makemore (Bigram + MLP character-level language model)
  - Nov 9: Redo weakest topic from Week 1 + write technical summary note
- **Week 2:** Activations, Gradients, BatchNorm internals, and diagnostic tools
- **Week 3:** Tokenization (BPE), Embeddings, and Self-Attention / Transformer from scratch (nanoGPT)
- **Week 4:** Fine-tuning, optimization, quantization, and production inference mechanics

## Repository Structure
- src/ - Python scripts and modular implementations
- notebooks/ - Jupyter / Kaggle notebooks for experiments
- docs/ - Diagnostic notes and learning logs (docs/gap-list.md)

## How to Run Locally (Windows 11 + CUDA)
1. Create and activate the virtual environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies (PyTorch CUDA 12.6 + core tools):
   pip install torch torchvision torchaudio --extra-index-url download.pytorch.org/whl/cu126
   pip install numpy matplotlib jupyter

3. Run the environment verification script:
   python src/test_env.py