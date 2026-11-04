import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name:     {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tiny autograd test: z = 2 * (x^2) + 1 at x = 3.0 -> dz/dx = 4*x = 12.0
x = torch.tensor(3.0, requires_grad=True, device=device)
y = x ** 2
z = 2 * y + 1
z.backward()

print(f"Tensor device:   {z.device}")
print(f"Forward out (z): {z.item()}")
print(f"Gradient (x.grad): {x.grad.item()}")