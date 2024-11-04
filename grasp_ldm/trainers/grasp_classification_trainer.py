import os
import warnings

import einops
import torch
import torch.nn as nn
import torcheval.metrics.functional as Metrics
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from torch.utils.data import Dataset
from utils.rotations import tmrp_to_H

from grasp_ldm.dataset.builder import build_dataset_from_cfg
from grasp_ldm.models.builder import build_model_from_cfg
from grasp_ldm.utils.config import Config, ConfigDict

from .experiment import Experiment
from .trainer import LightningTrainer


class GraspClassificationTrainer(LightningTrainer):
    CLS_PRED_THRESHOLD = 0.5

    def __init__(self, config: Config = None):
        """Grasp Classification Trainer"""

        # Split main sub-configs
        model_config = config.model
        data_config = config.data
        trainer_config = config.trainer

        # Experiment and config
        self._config = config
        self._experiment = Experiment(config.filename)

        # Checkpointing
        self._checkpointing_freq = (
            trainer_config.checkpointing_freq
            if hasattr(trainer_config, "checkpointing_freq")
            else 1000
        )
        trainer_config.default_root_dir = self._experiment.ckpt_dir

        # Initialize parent trainer class
        super().__init__(model_config, data_config, trainer_config)

        self.resume_from_checkpoint = self._experiment.default_resume_checkpoint

    def _build_dataset(self, data_config, split):
        """Custom routine for building dataset"""
        dataset = build_dataset_from_cfg(data_config, split)

        # dataset.pre_load() should define any pre-loading operations before workers are spawned
        dataset.pre_load()

        return dataset

    def _build_model(self, model_config):
        """Custom routine for building model"""
        model = build_model_from_cfg(ConfigDict(model=model_config))

        ## TODO: custom model initialization, if any
        # model.initialize()

        return model

    def training_step(self, batch_data, batch_idx):
        """Training step"""

        # Inputs
        pc = batch_data["pc"]
        grasps = batch_data["grasps"]

        # TODO: verify this reshape consistency
        success_labels = batch_data["success"].view(-1)
        num_grasps = grasps.shape[1]

        # Repeat pc and grasp so there is a 1-1 pairing
        pc = pc.repeat_interleave(num_grasps, dim=0)
        grasps = einops.rearrange(grasps, "b n c d -> (b n) c d")

        # Metas
        metas = batch_data["metas"]

        # Forward
        loss, _ = self.model(pc, grasps, cls_target=success_labels, compute_loss=True)

        # Log Loss
        self.log("loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        """Validation step"""

        # Inputs
        pc = batch_data["pc"]
        grasps = batch_data["grasps"]

        # TODO: verify this reshape consistency
        success_labels = batch_data["success"].view(-1)
        num_grasps = grasps.shape[1]

        # Repeat pc and grasp so there is a 1-1 pairing
        pc = pc.repeat_interleave(num_grasps, dim=0)
        grasps = einops.rearrange(grasps, "b n c d -> (b n) c d")

        # Metas
        metas = batch_data["metas"]

        # Forward
        loss, preds = self.model(
            pc, grasps, cls_target=success_labels, compute_loss=True
        )

        # Convert probs to binary preds
        preds = preds.detach()
        preds[preds > self.CLS_PRED_THRESHOLD] = 1
        preds[preds <= self.CLS_PRED_THRESHOLD] = 0

        # Accumulate preds in cache
        self._update_cache("validation", "epoch", "cls_preds", preds.long())
        self._update_cache("validation", "epoch", "cls_targets", success_labels.long())

        # Log Loss
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return

    def on_validation_epoch_end(self) -> None:
        eval_metrics = self._compute_metrics()
        self.log_dict({"validation_metrics": eval_metrics}, sync_dist=True)
        self.log(
            "val_accuracy", eval_metrics["accuracy"], prog_bar=True, sync_dist=True
        )
        return

    def _get_callbacks(self) -> list:
        """Custom callbacks to be used by the trainer."""

        checkpoint_callback1 = ModelCheckpoint(
            save_top_k=3,
            monitor="loss",
            mode="min",
            dirpath=self._experiment.ckpt_dir,
            filename="epoch_{epoch:02d}-step_{step}-loss_{loss:.2f}",
            save_last=True,
            every_n_train_steps=self._checkpointing_freq,
        )

        checkpoint_callback2 = ModelCheckpoint(
            save_top_k=1,
            monitor="loss",
            mode="min",
            dirpath=self._experiment.ckpt_dir,
            filename="best",
            save_last=True,
            every_n_train_steps=1000,
        )

        lr_monitor_callback = LearningRateMonitor(logging_interval="step")

        callbacks = [checkpoint_callback1, checkpoint_callback2, lr_monitor_callback]

        return callbacks

    def _get_logger(self) -> Logger:
        """Custom logger to be used by the trainer."""
        if hasattr(self.trainer_config, "logger"):
            logger_config = self.trainer_config.logger

            if logger_config.type == "WandbLogger":
                assert hasattr(
                    logger_config, "project"
                ), "WandbLogger requires a project name to be specified in the config."

                logger = WandbLogger(
                    name=self._experiment.name,
                    project=logger_config.project,
                    save_dir=self._experiment.log_dir,
                    config=self._config,
                )
            elif logger_config.type == "TensorBoardLogger":
                logger = TensorBoardLogger(
                    save_dir=self._experiment.log_dir,
                    name=self._experiment.name,
                )
        else:
            logger = CSVLogger(
                save_dir=self._experiment.log_dir,
                name=self._experiment.name,
            )
        return logger

    def _compute_metrics(self):
        """Compute metrics on validation set"""

        # Collect preds and targets from cache
        cls_preds = torch.cat(self._validation_cache["epoch"]["cls_preds"])
        cls_targets = torch.cat(self._validation_cache["epoch"]["cls_targets"])

        # Compute binary classification metrics
        metrics = dict(
            accuracy=Metrics.binary_accuracy(cls_preds, cls_targets),
            precision=Metrics.binary_precision(cls_preds, cls_targets),
            recall=Metrics.binary_recall(cls_preds, cls_targets),
            f1=Metrics.binary_f1_score(cls_preds, cls_targets),
            aP=Metrics.binary_auprc(cls_preds, cls_targets),
            # confusion_matrix=Metrics.binary_confusion_matrix(cls_preds, cls_targets),
        )

        return metrics
