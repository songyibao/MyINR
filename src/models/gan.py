import torch.nn as nn

from src.configs.config import MyConfig


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        self.h = MyConfig.get_instance().train.h
        self.w = MyConfig.get_instance().train.w
        downsampled_height = self.h // 2 ** 4
        downsampled_width = self.w // 2 ** 4

        # The height and width of downsampled image
        self.adv_layer = nn.Sequential(nn.Linear(128 * downsampled_height * downsampled_width, 1), nn.Sigmoid())

    def forward(self, img):
        img = img.permute(2,0,1).unsqueeze(0)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity