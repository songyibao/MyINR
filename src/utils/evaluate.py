from typing import Dict

import numpy as np
import piq
import torch
from torch._C import dtype

from src.configs.config import MyConfig

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def calculate_bpp(image: torch.Tensor, model: torch.nn.Module):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels

def binary_to_float(binary_tensor):
    # 获取输入张量的设备
    device = binary_tensor.device

    # 获取最后一维的大小
    last_dim_size = binary_tensor.shape[-1]

    # 确保最后一维是24，表示3个通道，每个通道8位
    if last_dim_size % 8 != 0:
        raise ValueError("The last dimension size must be a multiple of 8 (e.g., 24 for RGB).")

    # 计算通道数（24位的二进制表示每8位为一个通道）
    num_channels = last_dim_size // 8

    # 逐通道还原，从二进制表示还原为整数
    # 在与输入张量相同的设备上创建张量
    restored_tensor = torch.zeros(*binary_tensor.shape[:-1], num_channels, dtype=torch.int32, device=device)

    # 对每个通道的8位二进制进行处理，恢复为整数
    for i in range(8):
        restored_tensor += (binary_tensor[..., i::8] << (7 - i))

    # 将整数缩放到[0, 1]之间的浮点数
    restored_float_tensor = restored_tensor.float() / 255.0

    return restored_float_tensor

def evaluate_tensor_h_w_3(original_image: torch.Tensor, compressed_image: torch.Tensor) -> dict:
    if MyConfig.get_instance().net.use_binary_pixels:
        original_image = binary_to_float(original_image.int())
        compressed_image = binary_to_float(torch.round(compressed_image).int())
    # 转换成 [1,C,H,W]
    x = original_image.permute(2, 0, 1).unsqueeze(0)
    y = compressed_image.permute(2, 0, 1).unsqueeze(0)
    mse = torch.mean((x - y) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    msssim = piq.multi_scale_ssim(x, y, data_range=1.)
    return {
        'PSNR': round(psnr.item(), 2),
        'MS-SSIM': round(msssim.item(), 2)
    }
