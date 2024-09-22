from typing import Dict

import numpy as np
import piq
import torch
from torch._C import dtype

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


def evaluate_tensor_h_w_3(original_image: torch.Tensor, compressed_image: torch.Tensor) -> dict:
    # 转换成 [1,C,H,W]
    x = original_image.permute(2, 0, 1).unsqueeze(0)
    y = compressed_image.permute(2, 0, 1).unsqueeze(0)
    psnr = piq.psnr(x, y, data_range=1.)
    msssim = piq.multi_scale_ssim(x, y, data_range=1.)
    return {
        'PSNR': round(psnr.item(), 2),
        'MS-SSIM': round(msssim.item(), 2)
    }
