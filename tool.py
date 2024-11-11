import torch

# PyTorch 버전 확인
print("Torch version:", torch.__version__)

# CUDA 사용 가능 여부 및 버전 확인
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
