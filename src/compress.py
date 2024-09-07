import torch
import numpy as np
from PIL import Image
from src.models.inr_model import INRModel

def compress(image_path, model_path, output_path):
    # 加载图像
    img = Image.open(image_path)
    img_np = np.array(img) / 255.0

    # 加载模型
    model = INRModel(in_features=2, hidden_features=256, hidden_layers=3, out_features=3)
    model.load_state_dict(torch.load(model_path))

    # 保存模型参数（这里我们直接保存整个模型，实际应用中可能需要更复杂的压缩方法）
    torch.save(model.state_dict(), output_path)