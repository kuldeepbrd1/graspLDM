from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import Sequence

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import DDPStrategy
from torch import optim
from torch.utils.data import DataLoader, Dataset

from grasp_ldm.utils.config import Config, ConfigDict

from .mixins import TrainerEMAMixin


def default(input, default_value):
    return input if input else default_value


class LightningTrainer(LightningModule, TrainerEMAMixin):
    LR_SCHEDULERS = {
        "StepLR": optim.lr_scheduler.StepLR,
        "MultiStepLR": optim.lr_scheduler.MultiStepLR,
        "ConstantLR": optim.lr_scheduler.ConstantLR,
        "LinearLR": optim.lr_scheduler.LinearLR,
        "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    }

    _trainer_required_keys = [
        "max_steps",
        "devices",
        "num_workers",
        "log_every_n_steps",
        "default_root_dir",
    ]

    _trainer_optional_keys = {
        "accelerator": "gpu",
        "strategy": "ddp",
        "num_nodes": 1,
        "cudnn_benchmark": True,
        "log_every_n_steps": 500,
        "gradient_clip_val": 0.5,
        "check_val_every_n_epoch": 10,
    }

    def __init__(
        self,
        model_config: ConfigDict,
        data_config: ConfigDict,
        trainer_config: ConfigDict,
        skip_validation: bool = False,
    ):
        super().__init__()

        # Model
        self.model = self._build_model(model_config)
        self._model_config = model_config

        # EMA model. If specified in trainer_config, it will be set, otherwise no update
        self.ema_model = None
        self.configure_ema(trainer_config)

        # Trainer config
        self.trainer_config = self._validate_trainer_config(trainer_config)
        self.lr = None

        # Optimizer and LR scheduler
        assert hasattr(
            trainer_config, "optimizer"
        ), "Could not find key `optimizer` in trainer_config"

        self.optimizer, self.lr_scheduler = self._get_optimizer_scheduler(
            trainer_config.optimizer
        )

        # Dataloaders
        self._train_dataloader = self._get_dataloaders(
            data_config,
            split="train",
            batch_size=data_config["train"].batch_size,
        )
        if not skip_validation:
            self._val_dataloader = self._get_dataloaders(
                data_config, split="val", batch_size=2  # data_config["val"].batch_size,
            )

        # Lightning Trainer. Settable in run()
        self.pl_trainer = None

        # Resume from checkpoint
        self.resume_from_checkpoint = None

        # Log hyperparams
        self.save_hyperparameters(ignore=["model"])

        # Cache to store across metrics
        self._train_cache = {"global": {}, "epoch": {}, "step": {}}
        self._validation_cache = {"global": {}, "epoch": {}, "step": {}}

    @property
    def use_ema(self):
        return self.ema_model is not None

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader if hasattr(self, "_val_dataloader") else None

    @abstractmethod
    def _build_model(self, model_config):
        """Build model: Abstract method to be implemented by the subclass
        Args:
            model_config (dict): model config
        """
        raise NotImplementedError

    @abstractmethod
    def _build_dataset(self, data_config, split):
        """Build dataset: Abstract method to be implemented by the subclass
        Args:
            data_config (dict): data config
            split (str): split name
        """
        raise NotImplementedError

    @abstractmethod
    def _get_callbacks(self) -> list:
        """Abstract class: Return a list of callbacks to use during training"""
        raise NotImplementedError

    @abstractmethod
    def _get_logger(self) -> Logger:
        """Abstract class: Return a logger to use during training"""
        raise NotImplementedError

    @abstractmethod
    def _compute_metrics(self):
        """Abstract class: Compute metrics during training"""
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step: Abstract method to be implemented by the subclass
        Args:
            batch (dict): batch data returned by __getitem__ in the implemented Dataset
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Validation step: Abstract method to be implemented by the subclass
        Args:
            batch (dict): batch data returned by __getitem__ in the implemented Dataset
        """
        raise NotImplementedError

    def on_validation_epoch_start(self) -> None:
        """Before validation epoch, clear cache""" ""
        self._clear_cache("validation", "epoch")

    def on_validation_epoch_end(self) -> None:
        """After validation epoch, compute metrics""" ""
        self._compute_metrics()

    def on_after_backward(self) -> None:
        """After backward pass, do this"""
        if self.use_ema:
            self.ema_model.update()

    def _get_dataloaders(self, data_config, split, *, batch_size=8):
        dataset = self._build_dataset(data_config, split=split)

        num_workers = self.trainer_config.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if split == "val" else True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=not (num_workers == 0),
        )

    def configure_optimizers(self):
        return [self.optimizer], {"scheduler": self.lr_scheduler, "interval": "step"}

    def run(self):
        self.pl_trainer = self.get_pl_trainer()
        self.pl_trainer.fit(
            model=self,
            train_dataloaders=self.train_dataloader(),
            val_dataloaders=self.val_dataloader() if self.val_dataloader() else None,
        )

    def get_pl_trainer(self):
        """Get training arguments for the trainer"""

        callbacks = self._get_callbacks()

        strategy = (
            DDPStrategy(find_unused_parameters=False)
            if self.trainer_config.strategy == "ddp"
            else self.trainer_config.strategy
        )

        return Trainer(
            max_steps=self.trainer_config.max_steps,
            accelerator=self.trainer_config.accelerator,
            strategy=strategy,
            num_nodes=self.trainer_config.num_nodes,
            devices=self.trainer_config.devices,
            benchmark=self.trainer_config.cudnn_benchmark,
            logger=self._get_logger(),
            log_every_n_steps=self.trainer_config.log_every_n_steps,
            callbacks=callbacks,
            gradient_clip_val=self.trainer_config.gradient_clip_val,
            check_val_every_n_epoch=self.trainer_config.check_val_every_n_epoch,
            resume_from_checkpoint=self.resume_from_checkpoint,
            default_root_dir=self.trainer_config.default_root_dir,
        )

    def _get_optimizer_scheduler(self, optim_config):
        """Get optimizer and scheduler from config

        Args:
            optim_config (dict): Optimizer config

        Returns:
            tuple: optimizer, scheduler
        """

        # Initial LR
        for lr_key in ("learning_rate", "lr", "initial_lr"):
            if hasattr(optim_config, lr_key):
                self.lr = optim_config[lr_key]

        self.lr = self.lr if self.lr is not None else 0.001

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Check is scheduler supported
        if optim_config.scheduler.type not in list(self.LR_SCHEDULERS):
            raise KeyError(
                f"Schedule type: {self.lr_schedule_type} is not supported. \nSupported schedulers are {list(self.LR_SCHEDULERS)}"
            )

        self.lr_schedule_type = optim_config.scheduler.type

        lr_scheduler = self.LR_SCHEDULERS[optim_config.scheduler.type](
            optimizer=optimizer, **optim_config.scheduler.args
        )

        return optimizer, lr_scheduler

    def _validate_trainer_config(self, trainer_config):
        """Validate trainer config and set default values if not present

        Args:
            trainer_config (dict): Trainer config

        Returns:
            dict: Trainer config with default values set for missing keys
        """
        assert trainer_config is not None, "Trainer config cannot be None"

        for key in self._trainer_required_keys:
            assert hasattr(
                trainer_config, key
            ), f"Could not find key {key} in trainer config"

        for key, value in self._trainer_optional_keys.items():
            if not hasattr(trainer_config, key):
                setattr(trainer_config, key, value)

        return trainer_config

    def _update_cache(self, mode, type, key, val):
        """Update cache

        Args:
            mode (str): "train" or "validation"
            type (str): "global, "epoch" or "step"
            key (str): key name
            val (any): value
        """

        assert mode in ("train", "validation"), f"Invalid mode: {mode}"
        assert type in ("global", "epoch", "step"), f"Invalid type: {type}"

        _cache = self._train_cache if mode == "train" else self._validation_cache

        if key in _cache[type]:
            assert isinstance(_cache[type][key], Sequence), f"Invalid cache type: {key}"
            _cache[type][key].append(val)
        else:
            _cache[type][key] = [val]

    def _get_cache(self, mode, type, key=None):
        """Get cache

        Args:
            mode (str): "train" or "validation"
            type (str): "global, "epoch" or "step"
            key (str): key name
        """
        assert mode in ("train", "validation"), f"Invalid mode: {mode}"
        assert type in ("global", "epoch", "step"), f"Invalid type: {type}"

        _cache = self._train_cache if mode == "train" else self._validation_cache

        return _cache[type][key] if key else _cache[type]

    def _clear_cache(self, mode, type):
        """Clear cache

        Args:
            mode (str): "train" or "validation"
            type (str): "global, "epoch" or "step"
        """
        assert mode in ("train", "validation"), f"Invalid mode: {mode}"
        assert type in ("global", "epoch", "step"), f"Invalid type: {type}"

        if mode == "train":
            self._train_cache[type] = {}
        else:
            self._validation_cache[type] = {}

        return
