from pynvml import *
import torch

# Init NVML
nvmlInit()

handle = nvmlDeviceGetHandleByIndex(0)
clock_mhz = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
clock_khz = clock_mhz * 1000

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)

# Estimation des cœurs CUDA (Ampere)
cuda_cores = props.multi_processor_count * 128  

gflops = cuda_cores * 2 * clock_khz * 1e-6

# VRAM
mem_info = nvmlDeviceGetMemoryInfo(handle)
vram_total = mem_info.total / (1024**3)  # en GB
vram_used = mem_info.used / (1024**3)
vram_free = mem_info.free / (1024**3)

print("GPU :", props.name)
print("SM Count :", props.multi_processor_count)
print("CUDA Cores :", cuda_cores)
print("Clock :", clock_khz, "kHz")
print("VRAM total :", round(vram_total, 2), "GB")
print("VRAM used :", round(vram_used, 2), "GB")
print("VRAM free :", round(vram_free, 2), "GB")
print(f"GFLOPS theorics in f32 : {gflops:.2f}")
print(f"TFLOPS theorics in f32 : {gflops/1000:.2f}")

nvmlShutdown()
