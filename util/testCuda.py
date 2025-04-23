import torch

print("\n===== PYTORCH & CUDA INFO =====")
print("\nIs CUDA available?", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA version (from PyTorch):", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print("Name:", torch.cuda.get_device_name(i))
        print("Total memory (MB):", torch.cuda.get_device_properties(i).total_memory // (1024**2))
        print(f"CUDA driver version: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}\n")
        
else:
    print("CUDA is not available")