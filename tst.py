import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available. GPU detected: {gpu_name}")
else:
    print("CUDA is not available. No GPU detected.")
