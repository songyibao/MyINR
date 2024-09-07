# 初始化数据集和数据加载器
# dataset = ImageDataset(config['data_path'])
# dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
# 实现上面的ImageDataset

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.imgs[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img