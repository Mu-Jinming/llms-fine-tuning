import torch

# 打印 PyTorch 版本
print("PyTorch Version:", torch.__version__)

# 检测是否可以使用 GPU
if torch.cuda.is_available():
    print("CUDA is available. GPU(s) can be used.")
    # 打印 GPU 数量和型号
    num_gpus = torch.cuda.device_count()
    print("Number of GPU(s) available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. GPU cannot be used.")
