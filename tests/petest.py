import os

import torch
from torch import nn
from tqdm import tqdm

from src.configs.config import MyConfig
from src.utils.data_loader import FFM
from src.utils.device import get_best_device


# 定义模型
class Net(nn.Module):
    def __init__(self,in_features,out_features=128):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc_out(x)
        return x
config = MyConfig.get_instance()
device = get_best_device()
learned_embedding = torch.load(os.path.join(config.save.base_output_path, 'embedding.pth').__str__(),map_location="cpu",weights_only=True).to(device)
print(learned_embedding.shape)

h,w = 512,768
y_coords, x_coords = torch.meshgrid(
    torch.linspace(0, 1, h, dtype=torch.float32),
    torch.linspace(0, 1, w, dtype=torch.float32), indexing='ij'
)
# coords: (h*w,2)
coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
coords = FFM(in_features=coords.shape[-1], out_features=8).fmap(coords).to(device)
model = Net(in_features=coords.shape[-1],out_features=learned_embedding.shape[-1]).to(device)
num_steps = 10000
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
with tqdm(total=num_steps, desc=f"Training:") as pbar:
    for epoch in range(num_steps):
        optimizer.zero_grad()

        output = model(coords)

        loss = criterion(output, learned_embedding)
        loss.backward()

        optimizer.step()

        val_loss = loss.item()
        pbar.set_postfix({
            "loss": val_loss
        })
        pbar.update()