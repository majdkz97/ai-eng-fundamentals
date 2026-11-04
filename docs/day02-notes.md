# Day 2: Tensors, Broadcasting, and Autograd Mechanics

## Key Takeaways

1. **Broadcasting is Right-to-Left:**
   - PyTorch aligns dimensions on the right margin.
   - Dimensions of size 1 are stretched to match the other tensor.
   - **The Error Trap:** `(4, 3) + (4,)` fails because `3 != 4`. No left-alignment occurs.
   - **The Silent Bug:** `(3,) == (3, 1)` broadcasts to `(3, 3)`. Always assert shapes before loss functions.

2. **Memory, Strides, and Contiguity:**
   - `x.transpose()` is $O(1)$. It does not move memory; it only swaps strides.
   - `x.view()` requires contiguous memory (strides in descending order ending in 1). 
   - `x.reshape()` is safer: it uses `.view()` if contiguous, or copies to new memory if not.

3. **Batched Matmul (`@`):**
   - Retains batch dimensions, applies standard 2D matmul on the last two dimensions.
   - Attention score shape: `(B, T, C) @ (B, C, T) -> (B, T, T)`.

4. **Autograd Engine:**
   - `.backward()` accumulates (adds) gradients to the `.grad` attribute. You must call `optimizer.zero_grad()` to clear them between batches.
   - `with torch.no_grad():` turns off the computation graph. Skipping this during inference/validation leads to catastrophic OOM errors because the graph prevents intermediate tensors from being garbage collected.
   - Neural network backpropagation is just the chain rule applied sequentially through the graph.