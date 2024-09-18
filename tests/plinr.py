from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.configs.config import MyConfig
from src.decompress import decompress_and_save
from src.models.loss import LossRegistry
from src.models.model1 import ConfigurableINRModel
from src.utils.data_loader import ImageCompressionDataset
from src.utils.evaluate import evaluate_tensor_h_w_3


class PLINR(L.LightningModule):
    def __init__(self, config: MyConfig):
        super().__init__()
        self.config = config
        loss_class = LossRegistry.get(config.train.loss_type)
        self.loss = loss_class()
        self.train_set = ImageCompressionDataset(self.config, mode='train')
        self.test_set = ImageCompressionDataset(self.config, mode='test')
        coords, _, _, _ = self.train_set[0]
        self.INR = ConfigurableINRModel(self.config.net, in_features=coords.shape[-1])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.INR.parameters(), lr=self.config.train.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.config.train.scheduler_step_size,
                           gamma=self.config.train.scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(self.train_set, batch_size=1)
        return train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = DataLoader(self.test_set, batch_size=1)
        return test_loader

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        coords, original_image, h, w = batch
        output_pixels = self.INR(coords)
        loss = self.loss(output_pixels, original_image)
        # evaluate_res = evaluate_tensor_h_w_3(original_image.view(h, w, 3),
        #                                      torch.clamp(output_pixels, 0, 1).view(h, w, 3))

        # self.log("train_loss", loss)
        # self.log("PSNR", evaluate_res["PSNR"])
        # self.log("MS-SSIM", evaluate_res["MS-SSIM"])
        return loss
    def test_step(self, batch, batch_idx):
        res = decompress_and_save(self.INR,self.config,config.save.base_output_path)
        return res
config = MyConfig.get_instance()
print(config.model_dump())
model = PLINR(config)
trainer = L.Trainer(max_epochs=config.train.num_epochs, log_every_n_steps=1)
trainer.fit(model)
trainer.test(model,verbose=True)
