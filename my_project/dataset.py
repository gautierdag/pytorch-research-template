from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class MyProjectDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir_path: str = "path/to/dir",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.dataset = TensorDataset(
            torch.randn((1000, 512)), torch.randint(0, 10, (1000,))
        )
        self.train, self.val = random_split(self.dataset, [800, 200])
        self.test = TensorDataset(torch.randn((100, 512)), torch.randint(0, 10, (100,)))

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
