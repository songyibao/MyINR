from typing import Dict
import piq
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch._C import dtype
import imageio.v3 as iio

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


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


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


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)


def get_original_image_numpy(original_image_path: str) -> np.ndarray:
    """
    从文件路径加载原始图像

    参数:
        original_image_path (str): 原始图像的文件路径

    返回:
        np.ndarray: 原始图像的 numpy 数组 [h, w, c]
    """
    return iio.imread(original_image_path)


def get_original_image_tensor(original_image_path: str) -> torch.Tensor:
    """
    从文件路径加载原始图像

    参数:
        original_image_path (str): 原始图像的文件路径

    返回:
        torch.Tensor: 原始图像的 tensor [h, w, c]
    """
    original_image = torch.tensor(iio.imread(original_image_path))
    return original_image


def calculate_psnr_ndarray(original_image: np.ndarray, compressed_image: np.ndarray) -> float:
    """
    计算峰值信噪比 (PSNR)

    参数:
        original_image (np.ndarray): 原始图像（灰度图或RGB图像） [h,w,3]
        compressed_image (np.ndarray): 压缩后的图像（灰度图或RGB图像）[h,w,3]

    返回:
        float: PSNR 值，单位为 dB
    """
    # mse = np.mean((original_image - compressed_image) ** 2)
    # if mse == 0:  # 图像完全相同
    #     return float('inf')
    #
    # max_pixel_value = 255.0
    # res = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    # return res
    x = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(compressed_image).permute(2, 0, 1).unsqueeze(0)

    psnr_index = piq.psnr(x, y, data_range=255., reduction='none')
    return psnr_index.item()


def calculate_psnr_tensor(original_image: torch.Tensor, compressed_image: torch.Tensor) -> float:
    """
    计算峰值信噪比 (PSNR)

    参数:
        original_image (torch.Tensor): 原始图像（灰度图或RGB图像） [3,h,w]
        compressed_image (torch.Tensor): 压缩后的图像（灰度图或RGB图像）[3,h,w]

    返回:
        float: PSNR 值，单位为 dB
    """
    x = original_image.permute(2, 0, 1).unsqueeze(0)
    y = compressed_image.permute(2, 0, 1).unsqueeze(0)
    psnr_index = piq.psnr(x, y, data_range=1.,reduction='none')
    return psnr_index.item()


def calculate_msssim_ndarray(original_image: np.ndarray, compressed_image: np.ndarray, is_rgb: bool = True) -> float:
    """
    计算多尺度结构相似性 (MS-SSIM)

    参数:
        original_image (np.ndarray): 原始图像（灰度图或RGB图像） [h, w, c]
        compressed_image (np.ndarray): 压缩后的图像（灰度图或RGB图像） [h, w, c]
        is_rgb (bool): 如果输入的是RGB图像，则设为True

    返回:
        float: MS-SSIM 值
    """
    if is_rgb:
        # 对 RGB 图像的处理
        original_image_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0).float()
        compressed_image_tensor = torch.from_numpy(compressed_image).permute(2, 0, 1).unsqueeze(0).float()
    else:
        # 对灰度图像的处理
        original_image_tensor = torch.from_numpy(original_image).unsqueeze(0).unsqueeze(0).float()
        compressed_image_tensor = torch.from_numpy(compressed_image).unsqueeze(0).unsqueeze(0).float()

    # 计算 MS-SSIM
    msssim_value = ms_ssim(original_image_tensor, compressed_image_tensor, data_range=255)

    return msssim_value.item()


def calculate_msssim_tensor(original_image: torch.Tensor, compressed_image: torch.Tensor, is_rgb: bool = True,
                            data_range: float = 1.0) -> float:
    """
    计算多尺度结构相似性 (MS-SSIM)

    参数:
        original_image (torch.Tensor): 原始图像（灰度图或RGB图像） [c,h,w]
        compressed_image (torch.Tensor): 压缩后的图像（灰度图或RGB图像） [c,h,w]
        is_rgb (bool): 如果输入的是RGB图像，则设为True

    返回:
        float: MS-SSIM 值
    """
    # 如果original_image和compressed_image的device不同，就抛出一个错误
    if original_image.device != compressed_image.device:
        raise ValueError("original_image and compressed_image must be on the same device")
    if is_rgb:
        # 对 RGB 图像的处理
        original_image_tensor = original_image.permute(2, 0, 1).unsqueeze(0).float()
        compressed_image_tensor = compressed_image.permute(2, 0, 1).unsqueeze(0).float()
    else:
        # 对灰度图像的处理
        original_image_tensor = original_image.unsqueeze(0).unsqueeze(0).float()
        compressed_image_tensor = compressed_image.unsqueeze(0).unsqueeze(0).float()

    # 计算 MS-SSIM
    msssim_value = ms_ssim(original_image_tensor, compressed_image_tensor, data_range=data_range)

    return msssim_value.item()


def evaluate_ndarray(original_image: np.ndarray, compressed_image: np.ndarray) -> dict:
    """
    计算压缩图像的 PSNR 和 MS-SSIM

    参数:
        original_image (np.ndarray): 原始图像（灰度图或 RGB 图像）
        compressed_image (np.ndarray): 压缩后的图像（灰度图或 RGB 图像）

    返回:
        dict: 包含 PSNR 和 MS-SSIM 值的字典
    """
    psnr = calculate_psnr_ndarray(original_image, compressed_image)
    msssim = calculate_msssim_ndarray(original_image, compressed_image)

    return {'PSNR': psnr, 'MS-SSIM': msssim}


def evaluate_tensor(original_image: torch.Tensor, compressed_image: torch.Tensor) -> dict:
    """
    计算压缩图像的 PSNR 和 MS-SSIM

    参数:
        original_image (torch.Tensor): 原始图像（灰度图或 RGB 图像）
        compressed_image (torch.Tensor): 压缩后的图像（灰度图或 RGB 图像）

    返回:
        dict: 包含 PSNR 和 MS-SSIM 值的字典
    """
    # 判断源图像值的范围是否在 [0, 1] 之间,如果是,转化为 [0, 255] 之间的整数值,否则判断值是否为 [0, 255] 之间的整数值
    # if original_image.min() >= 0 and original_image.max() <= 1:
    #     original_image = (original_image * 255).to(torch.uint8).float()
    # else:
    #     raise ValueError("original_image values must be in the range [0, 1]")
    #
    # # 判断压缩图像值的范围是否在 [0, 1] 之间,如果是,转化为 [0, 255] 之间的整数值,否则判断值是否为 [0, 255] 之间的整数值
    # if compressed_image.min() >= 0 and compressed_image.max() <= 1:
    #     compressed_image = (compressed_image * 255).to(torch.uint8).float()
    # else:
    #     return {'PSNR': None, 'MS-SSIM': None}

    # 输出两张图像的数值范围
    # print(f"original_image min: {original_image.min()}, max: {original_image.max()}")
    # print(f"compressed_image min: {compressed_image.min()}, max: {compressed_image.max()}")
    psnr = calculate_psnr_tensor(original_image, compressed_image)
    msssim = calculate_msssim_tensor(original_image, compressed_image,data_range=1.)

    return {
        'PSNR': f'{psnr:.4f}',  # 固定保留4位小数
        'MS-SSIM': f'{msssim:.6f}'  # 固定保留6位小数
    }
def evaluate_tensor_h_w_3(original_image: torch.Tensor, compressed_image: torch.Tensor) -> dict:
    # 转换成 [1,C,H,W]
    x = original_image.permute(2, 0, 1).unsqueeze(0)
    y = compressed_image.permute(2, 0, 1).unsqueeze(0)
    psnr = piq.psnr(x, y, data_range=1.)
    msssim = piq.multi_scale_ssim(x, y, data_range=1.)
    return {
        'PSNR': f'{psnr:.2f}',  # 固定保留4位小数
        'MS-SSIM': f'{msssim:.2f}'  # 固定保留6位小数
    }