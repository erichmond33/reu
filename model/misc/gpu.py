import torch

# Ensure CUDA is available
if torch.cuda.is_available():
    # Select a GPU device
    device = torch.device("cuda")

    # Ensure it is working
    torch.cuda.current_device()
    print(f"Selected CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device index: {torch.cuda.current_device()}")

    # Get current GPU memory usage
    allocated = torch.cuda.memory_allocated(device)
    print(f"Current GPU memory allocated: {allocated / 1024**3} GB")

    # Get maximum GPU memory usage
    max_allocated = torch.cuda.max_memory_allocated(device)
    print(f"Maximum GPU memory allocated: {max_allocated / 1024**3} GB")
else:
    print("No CUDA")