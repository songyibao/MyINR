import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
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


        self.coords, self.original_image, self.h, self.w, self.c = self.train_set[0]
        self.shape = (self.h, self.w, self.c)
        self.INR = ConfigurableINRModel(self.config.net, in_features=self.coords.shape[-1], out_features=self.c)
        self.scheduler = None




    def on_train_start(self):
        self.coords = self.coords.to(self.device)
        self.original_image = self.original_image.to(self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.INR.parameters(), lr=self.config.train.learning_rate)
        self.scheduler = StepLR(optimizer, step_size=self.config.train.scheduler_step_size,
                           gamma=self.config.train.scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler
            },
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(self.train_set, batch_size=1,num_workers=12)
        return train_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = DataLoader(self.test_set, batch_size=1,num_workers=12)
        return test_loader

    def training_step(self,_):
        # training_step defines the train loop.
        # it is independent of forward
        # batch 中包含了 coords, original_image, h, w, c，每个的第一个维度是 batch_size且固定为1，取消掉这个维度
        output_pixels = self.INR(self.coords)
        loss = self.loss(output_pixels, self.original_image)
        evaluate_res = evaluate_tensor_h_w_3(self.original_image.view(self.shape),
                                             torch.clamp(output_pixels, 0, 1).view(self.shape))

        self.log("LOSS", loss,prog_bar=True)
        self.log("PSNR", evaluate_res["PSNR"], logger=False, prog_bar=True)
        self.log("MS-SSIM", evaluate_res["MS-SSIM"], logger=False, prog_bar=True)
        self.log("Learning Rate", self.scheduler.get_last_lr()[0], logger=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        res = decompress_and_save(self.INR, self.config, config.save.base_output_path)
        return res


torch.set_float32_matmul_precision('medium')
config = MyConfig.get_instance()
model = PLINR(config)
mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri=f"file:{config.misc.log_save_path}")
checkpoint_callback = ModelCheckpoint(
    monitor='MS-SSIM',
    save_top_k=1,  # 只保存性能最好的模型
    mode='max',
    dirpath=config.save.net_save_path,  # 保存模型的目录
    filename=config.save.net_name,  # 保存模型文件的名称
    save_weights_only=False,  # 是否只保存权重，默认保存整个模型
    enable_version_counter=False,  # 是否启用版本计数器
)

trainer = L.Trainer(max_epochs=config.train.num_epochs,
                    log_every_n_steps=1,
                    logger=mlf_logger,
                    callbacks=[checkpoint_callback])
trainer.fit(model)
trainer.test(model, verbose=True)