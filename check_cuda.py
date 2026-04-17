import torch

print("="*60)
print("CUDA AVAILABILITY CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("\n✓ GPU is available - experiments will run FASTER")
else:
    print("\n✗ GPU NOT available - experiments will run on CPU (slower)")
print("="*60)
