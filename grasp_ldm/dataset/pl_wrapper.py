from typing import Sequence, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class GraspDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_dataset = train_dataset

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError
