# 初始化数据集和数据加载器
# dataset = ImageDataset(config['data_path'])
# dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
# 实现上面的ImageDataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.configs.config import MyConfig
from src.models.inputs import positional_encoding
from sklearn.preprocessing import PolynomialFeatures


def upsample_image(img, scale_factor=2):
    h, w, c = img.shape
    upsampled_img = np.zeros((scale_factor * h, scale_factor * w, c), dtype=img.dtype)
    upsampled_img[::scale_factor, ::scale_factor, :] = img
    return upsampled_img


def downsample_image(img, scale_factor=2):
    h, w, c = img.shape
    downsampled_img = img[::scale_factor, ::scale_factor, :]
    return downsampled_img


def poly_fit(coords, degree=3):
    poly = PolynomialFeatures(degree=degree)
    return torch.tensor(poly.fit_transform(coords), dtype=torch.float32)

# def positional_encoding(coords, num_freqs=10):
#     """
#     对输入坐标进行位置编码。
#
#     参数:
#         coords (torch.Tensor): 输入坐标，形状为 (N, 2)，包含 N 个二维坐标。
#         num_freqs (int): 编码频率数量，控制编码的维度大小。
#
#     返回:
#         encoded_coords (torch.Tensor): 编码后的坐标，形状为 (N, 4 * num_freqs)。
#     """
#     # 生成频率序列
#     frequencies = torch.tensor([2 ** i for i in range(num_freqs)], dtype=torch.float32)
#
#     # 对 x 和 y 分别编码
#     x, y = coords[:, 0], coords[:, 1]
#     x_encoded = []
#     y_encoded = []
#
#     for freq in frequencies:
#         # 对 x 和 y 进行编码，并保留每个编码的维度
#         x_encoded.append(torch.sin(freq * x).unsqueeze(-1))  # (N, 1)
#         x_encoded.append(torch.cos(freq * x).unsqueeze(-1))  # (N, 1)
#         y_encoded.append(torch.sin(freq * y).unsqueeze(-1))  # (N, 1)
#         y_encoded.append(torch.cos(freq * y).unsqueeze(-1))  # (N, 1)
#
#     # 将 x 和 y 的编码结果分别拼接
#     x_encoded = torch.cat(x_encoded, dim=-1)  # 形状: (N, 2 * num_freqs)
#     y_encoded = torch.cat(y_encoded, dim=-1)  # 形状: (N, 2 * num_freqs)
#
#     # 合并 x 和 y 的编码结果，得到最终的编码表示
#     encoded_coords = torch.cat([x_encoded, y_encoded], dim=-1)  # 形状: (N, 4 * num_freqs)
#
#     return encoded_coords


def float_to_binary(tensor):
    # 将[0,1]之间的浮点数映射到[0,255]的整数
    scaled_tensor = (tensor * 255).int()

    # 将每个通道的像素值转换为二进制字符串，保持8位长
    binary_tensor = torch.cat([((scaled_tensor >> i) & 1).unsqueeze(-1) for i in range(7, -1, -1)], dim=-1)

    # 将结果转换为24位的形式（每个像素的3个通道展开为24个二进制位）
    return binary_tensor.view(-1, 24)

def find_sin_cos_intersections(points, k, frequency):
    N = points.shape[0]  # 点的数量
    intersections = torch.zeros((N, 4 * k))  # 初始化返回张量

    # 对于 y = sin(x) 和 y = a 的交点
    a_mask = (-1 <= points[:, 0]) & (points[:, 0] <= 1)  # 判断 a 是否在 [-1, 1] 范围内
    valid_a = points[a_mask, 0]  # 获取有效的 a 值
    if valid_a.shape[0] > 0:
        x_sol_sin = torch.arcsin(valid_a)  # arcsin 解
        sin_solutions_pos = x_sol_sin[:, None] + 2 * torch.pi * torch.arange(-k, k, device=points.device)[None, :]  # 正向解
        sin_solutions_neg = -sin_solutions_pos  # 对称解
        sin_solutions = torch.cat((sin_solutions_pos, sin_solutions_neg), dim=1)  # 合并正向和对称解
        sin_solutions = torch.sort(sin_solutions, dim=1)[0][:, :2 * k]  # 取前 2k 个解
        intersections[a_mask, :2 * k] = sin_solutions

    # 对于 y = cos(x) 和 y = b 的交点
    b_mask = (-1 <= points[:, 1]) & (points[:, 1] <= 1)  # 判断 b 是否在 [-1, 1] 范围内
    valid_b = points[b_mask, 1]  # 获取有效的 b 值
    if valid_b.shape[0] > 0:
        x_sol_cos = torch.arccos(valid_b)  # arccos 解
        cos_solutions_pos = x_sol_cos[:, None] + 2 * torch.pi * torch.arange(-k, k, device=points.device)[None, :]  # 正向解
        cos_solutions_neg = -cos_solutions_pos  # 对称解
        cos_solutions = torch.cat((cos_solutions_pos, cos_solutions_neg), dim=1)  # 合并正向和对称解
        cos_solutions = torch.sort(cos_solutions, dim=1)[0][:, :2 * k]  # 取前 2k 个解
        intersections[b_mask, 2 * k:] = cos_solutions

    # 通过另一个频率的三角函数映射到 [-1, 1] 范围
    mapped_intersections = torch.sin(frequency * intersections)  # 使用正弦函数进行映射

    return mapped_intersections


def cartesian_to_polar(coords, scale=1.0):
    """
    将笛卡尔坐标转换为极坐标，并将其与原直角坐标拼接，同时结果归一化到 [-1, 1]

    参数:
    - coords: [H*W, 2] 的张量，笛卡尔坐标系下的 (x, y) 坐标
    - scale: 控制转换后的 r 的缩放尺度，默认为 1.0

    返回:
    - [H*W, 4] 的张量，包含归一化的 (x, y, r, theta)
    """
    # 提取 x 和 y 坐标
    x = coords[:, 0]
    y = coords[:, 1]

    # 计算极径 r，并进行缩放
    r = torch.sqrt(x**2 + y**2) * scale

    # 计算极角 theta
    theta = torch.atan2(y, x)

    # 归一化 r
    r_max = r.max()
    r = (r / r_max) * 2 - 1  # 将 r 归一化到 [-1, 1]

    # 归一化 theta 到 [-1, 1]
    theta = theta / torch.pi  # 将 theta 归一化到 [-1, 1]

    # 将原始的 (x, y) 和计算出的 (r, theta) 拼接
    cartesian_polar_coords = torch.cat(( r.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)

    return cartesian_polar_coords


class FFM():
    def __init__(self, in_features: int, out_features: int, scale: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

    # 创建 B 矩阵
    def create_fixed_B(self, in_features, out_features, scale=10, seed=3407):
        torch.manual_seed(seed)
        B = torch.randn((in_features, out_features)) * scale
        torch.nn.init.uniform_(B, a=0, b=1)
        return B
        # 定义傅里叶特征映射函数

    def fourier_feature_mapping(self, x, B):
        x_proj = 60 * x @ B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fmap(self, x):
        B = self.create_fixed_B(self.in_features, self.out_features, self.scale)
        return self.fourier_feature_mapping(x, B)
def get_coords(h, w):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, h, dtype=torch.float32),
        torch.linspace(-1, 1, w, dtype=torch.float32), indexing='ij'
    )
    res = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    return res
def partition_with_info(tensor, splits):
    """
    将张量切分并返回每个块的位置信息，支持不能整除的情况

    Args:
        tensor: 输入张量，形状为[H, W, C]
        splits: 包含三个整数的元组或列表 (a, b)，表示在H、W、T维度上分别切分的份数

    Returns:
        tuple: (blocks, positions)
            - blocks: 切分后的张量列表
            - positions: 每个块的位置和大小信息列表
    """
    H, W, C = tensor.shape
    a, b = splits

    # 确保分割数不大于维度大小
    a = min(a, H)
    b = min(b, W)

    def get_split_points(dim_size, num_splits):
        base_size = dim_size // num_splits
        remainder = dim_size % num_splits
        points = []
        sizes = []
        current = 0

        for i in range(num_splits):
            current_size = base_size + (1 if i < remainder else 0)
            current += current_size
            if i < num_splits - 1:
                points.append(current)
            sizes.append(current_size)

        return points, sizes

    h_points, h_sizes = get_split_points(H, a)
    w_points, w_sizes = get_split_points(W, b)

    blocks = []
    positions = []

    h_start = 0
    for i, h_end in enumerate(h_points + [H]):
        w_start = 0
        for j, w_end in enumerate(w_points + [W]):
            block = tensor[h_start:h_end, w_start:w_end, :]
            blocks.append(block)

            positions.append({
                'pos': [h_start, w_start],
                'size': [h_end - h_start, w_end - w_start, C]
            })

            w_start = w_end
        h_start = h_end

    return blocks, positions

def reconstruct_tensor(blocks, positions, original_shape=None):
    """
    将切分的张量块重组为原始张量

    Args:
        blocks: 张量块列表
        positions: 每个块的位置和大小信息列表，每个元素为字典，包含'pos'和'size'键
        original_shape: 可选，原始张量的形状 (H, W, C)。如果不提供，将从positions推断

    Returns:
        reconstructed: 重组后的张量

    Raises:
        ValueError: 如果块的数量与位置信息不匹配，或位置信息无效
    """
    if len(blocks) != len(positions):
        raise ValueError("块的数量与位置信息数量不匹配")

    # 如果没有提供原始形状，从positions中推断
    if original_shape is None:
        H = max(pos['pos'][0] + pos['size'][0] for pos in positions)
        W = max(pos['pos'][1] + pos['size'][1] for pos in positions)
        channels = blocks[0].shape[-1]
        original_shape = (H, W, channels)

    # 创建输出张量
    device = blocks[0].device if torch.is_tensor(blocks[0]) else None
    dtype = blocks[0].dtype if torch.is_tensor(blocks[0]) else None

    if device is not None:
        reconstructed = torch.zeros(original_shape, dtype=dtype, device=device)
    else:
        reconstructed = np.zeros(original_shape, dtype=dtype)

    # 将每个块放回原位置
    for block, pos_info in zip(blocks, positions):
        h_start, w_start = pos_info['pos']
        h_size, w_size,_ = pos_info['size']
        h_end = h_start + h_size
        w_end = w_start + w_size

        reconstructed[h_start:h_end, w_start:w_end, :] = block

    return reconstructed
def get_coords_with_config(h,w,config:MyConfig):
    coords = get_coords(h, w)
    if config.net.layers[0].type == 'LearnableEmbedding':
        coords = torch.arange(h * w).long()
    elif config.net.num_frequencies is not None:
        coords = positional_encoding(coords, num_frequencies=config.net.num_frequencies)
    elif config.net.degree is not None:
        coords = poly_fit(coords, degree=config.net.degree)
    elif config.net.ffm_out_features is not None:
        coords = FFM(in_features=coords.shape[-1], out_features=config.net.ffm_out_features).fmap(
            coords)
    elif config.net.use_polar_coords:
        coords = cartesian_to_polar(coords)
    return coords
class ImageCompressionDataset(Dataset):
    def __init__(self, config: MyConfig, mode: str = 'test'):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        self.config = config
        # 加载图像
        # self.img = Image.fromarray(skimage.data.camera()) # 读取示例图像
        self.img = Image.open(config.train.image_path)
        self.channels = -1
        # 判断图像的通道数
        if self.img.mode == 'RGB':
            self.channels = 3
        elif self.img.mode == 'L':
            self.channels = 1
        else:
            self.channels = len(self.img.getbands())
            if self.channels == 4:  # 如果是RGBA，转换为RGB
                self.img = self.img.convert('RGB')
                self.channels = 3
            else:
                raise ValueError(f"Unsupported image mode: {self.img.mode}")
        self.img_tensor = ToTensor()(self.img)  # 转换为 PyTorch 张量，形状为 (3, H, W)
        self.img = self.img_tensor.permute(1, 2, 0)
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]
        self.coords = get_coords_with_config(self.h, self.w, config)  # 转换为 (h * w, 2)
        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img_tensor.permute(1, 2, 0).view(-1, self.channels)





    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return 1

    def __getitem__(self, idx):
        """
        根据索引获取样本。

        参数:
        - idx (int): 索引值

        返回:
        - 坐标网格（形状为 (h * w, 2)）
        - 图像像素值（形状为 (h * w, 3)）
        """
        return self.coords, self.pixels, self.h, self.w, self.channels

class ImgDataset1dBlock(Dataset):
    def __init__(self, config: MyConfig, mode: str = 'test'):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        self.config = config
        # 加载图像
        # self.img = Image.fromarray(skimage.data.camera()) # 读取示例图像
        self.img = Image.open(config.train.image_path)
        self.channels = -1
        # 判断图像的通道数
        if self.img.mode == 'RGB':
            self.channels = 3
        elif self.img.mode == 'L':
            self.channels = 1
        else:
            self.channels = len(self.img.getbands())
            if self.channels == 4:  # 如果是RGBA，转换为RGB
                self.img = self.img.convert('RGB')
                self.channels = 3
            else:
                raise ValueError(f"Unsupported image mode: {self.img.mode}")
        self.img_tensor = ToTensor()(self.img)  # 转换为 PyTorch 张量，形状为 (3, H, W)
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]
        self.num_blocks = config.net.num_blocks
        self.block_size = (self.h * self.w) // self.num_blocks
        self.coords = get_coords(self.h, self.w)  # 转换为 (h * w, 2)
        self.img = self.img_tensor.permute(1, 2, 0)
        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img.view(-1, self.channels)

        if self.config.net.layers[0].type == 'LearnableEmbedding':
            self.coords = torch.arange(self.h * self.w).long()
        elif self.config.net.num_frequencies is not None:
            self.coords = positional_encoding(self.coords, num_frequencies=self.config.net.num_frequencies)
        elif self.config.net.degree is not None:
            self.coords = poly_fit(self.coords, degree=self.config.net.degree)
        elif self.config.net.ffm_out_features is not None:
            self.coords = FFM(in_features=self.coords.shape[-1], out_features=self.config.net.ffm_out_features).fmap(
                self.coords)
        elif self.config.net.use_polar_coords:
            self.coords = cartesian_to_polar(self.coords)


    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        if idx == self.num_blocks - 1:
            return self.coords[idx*self.block_size:], self.pixels[idx*self.block_size:]
        else:
            return self.coords[idx*self.block_size:(idx+1)*self.block_size], self.pixels[idx*self.block_size:(idx+1)*self.block_size]



class ImgDataset2dBlock(Dataset):
    def __init__(self, config: MyConfig, mode: str = 'test'):
        """
        初始化自定义数据集。

        参数:
        - image_path (str): 图像文件的路径。
        """
        self.config = config
        # 加载图像
        # self.img = Image.fromarray(skimage.data.camera()) # 读取示例图像
        self.img = Image.open(config.train.image_path)
        self.channels = -1
        # 判断图像的通道数
        if self.img.mode == 'RGB':
            self.channels = 3
        elif self.img.mode == 'L':
            self.channels = 1
        else:
            self.channels = len(self.img.getbands())
            if self.channels == 4:  # 如果是RGBA，转换为RGB
                self.img = self.img.convert('RGB')
                self.channels = 3
            else:
                raise ValueError(f"Unsupported image mode: {self.img.mode}")
        self.img_tensor = ToTensor()(self.img)  # 转换为 PyTorch 张量，形状为 (3, H, W)
        self.img = self.img_tensor.permute(1, 2, 0)
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]
        self.h_blocks, self.w_blocks = config.net.h_blocks, config.net.w_blocks
        self.total_blocks = self.h_blocks * self.w_blocks
        # 获取图像的像素值，形状为 (h * w, 3)
        self.pixels = self.img_tensor.permute(1, 2, 0)
        # 分块
        self.data_blocks,self.positions = partition_with_info(self.pixels, (self.h_blocks, self.w_blocks))
        self.coords_blocks = []
        for i in range(len(self.data_blocks)):
            tmp_shape = self.data_blocks[i].shape
            self.coords_blocks.append(get_coords_with_config(tmp_shape[0],tmp_shape[1], config))
            self.data_blocks[i] = self.data_blocks[i].reshape(-1, self.channels)

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, index):
        return self.coords_blocks[index], self.data_blocks[index]
