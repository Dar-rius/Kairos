import torch
import time

device = "cuda"

N = 8192

a = torch.rand((N, N), device=device)
b = torch.rand((N, N), device=device)

torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    c = a @ b  # this uses FMA heavily
torch.cuda.synchronize()

end = time.time()

# Each matrix multiply does ~2 * N^3 FLOPs
flops = 10 * 2 * (N ** 3)
tflops = flops / (end - start) / 1e12

print("Measured TFLOPS:", round(tflops, 2))

