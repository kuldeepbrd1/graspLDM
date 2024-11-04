import os
import random
import warnings
from abc import abstractmethod, abstractproperty

import torch
import torcheval.metrics.functional as Metrics
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger

from grasp_ldm.dataset.builder import build_dataset_from_cfg
from grasp_ldm.models.builder import build_model_from_cfg
from grasp_ldm.utils.torch_utils import fix_state_dict_prefix
from grasp_ldm.utils.config import Config
from utils.rotations import tmrp_to_H

from .experiment import Experiment
from .trainer import LightningTrainer, default


class GraspGenerationTrainer(LightningTrainer):
    def __init__(
        self,
        config: Config = None,
        skip_validation: bool = False,
    ):
        """Grasp Classification Trainer"""

        # Split main sub-configs
        model_config = config.model
        data_config = config.data
        trainer_config = config.trainer

        # Experiment and config
        self._config = config
        self._experiment = Experiment(
            config.filename, model_suffix=self._model_type_str
        )

        # Checkpointing
        self._checkpointing_freq = (
            trainer_config.checkpointing_freq
            if hasattr(trainer_config, "checkpointing_freq")
            else 1000
        )
        trainer_config.default_root_dir = self._experiment.ckpt_dir

        # Initialize parent trainer class
        super().__init__(
            model_config=model_config,
            data_config=data_config,
            trainer_config=trainer_config,
            skip_validation=skip_validation,
        )

        self.resume_from_checkpoint = self._experiment.default_resume_checkpoint

    @abstractproperty
    def _model_type_str(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def _use_qualities(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_grasps(self, pc, metas):
        raise NotImplementedError

    def _build_dataset(self, data_config, split):
        """Custom routine for building dataset"""
        dataset = build_dataset_from_cfg(data_config, split)

        # dataset.pre_load() should define any pre-loading operations before workers are spawned
        dataset.pre_load()

        return dataset

    def _build_model(self, model_config):
        raise NotImplementedError

    def training_step(self, batch_data, batch_idx):
        """Training step"""

        # Point cloud and grasps
        pc = batch_data["pc"]
        grasps_in = batch_data["grasps"]
        metas = batch_data["metas"]

        # self.visualize_inputs(pc, grasps_in, batch_data["metas"])

        # concat qualities if required
        if self._use_qualities:
            grasp_qualities = batch_data["qualities"]
            grasps_in = torch.concatenate((grasps_in, grasp_qualities), -1)

        grasps_in = grasps_in.view(-1, grasps_in.shape[-1])

        # Update any hyperparams that are scheduled
        self._update_scheduled_hyperparams()

        # Optional kwargs passed for post-processing
        kwargs = {"metas": metas}

        # Forward pass
        _, loss_dict = self.model(pc, grasps_in, compute_loss=True, **kwargs)

        self.log_dict({key: val for key, val in loss_dict.items()}, sync_dist=True)

        return loss_dict.loss

    def validation_step(self, batch_data, batch_idx):
        """Validation step"""
        val_cache = self._get_cache(mode="validation", type="batch")

        if "val_batch_idx" not in val_cache:
            self._update_cache(
                mode="validation",
                type="epoch",
                key="val_batch_idx",
                value=random.randint(0, len(self._val_dataloader) - 1),
            )

        if batch_idx == val_cache["val_batch_idx"]:
            pc = batch_data["pc"]
            # grasps_in = batch_data["grasps"]
            metas = batch_data["metas"]

            grasps = self.generate_grasps(pc, metas)

            # TODO: Metrics
        return

    def on_validation_epoch_end(self) -> None:
        return

    def _compute_metrics(self):
        """Compute metrics on validation set"""

        return

    def _update_scheduled_hyperparams(self):
        """Update hyperparams that depend on training steps"""
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
            save_weights_only=True,
            every_n_train_steps=self._checkpointing_freq,
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
                    name=f"{self._model_type_str.upper()}_{self._experiment.name}",
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


class GraspVAETrainer(GraspGenerationTrainer):
    def __init__(self, config: Config = None):
        super().__init__(config, skip_validation=True)

    @property
    def _model_type_str(self):
        return "vae"

    @property
    def _use_qualities(self):
        return self.model.use_grasp_qualities

    def generate_grasps(self, pc, metas):
        """Generate grasps from point cloud"""
        grasps = self.model.generate_grasps(pc, metas)

        return grasps

    def _build_model(self, model_config):
        model = build_model_from_cfg(model_config.vae)

        if model_config.vae.ckpt_path is not None:
            assert os.path.exists(
                model_config.vae.ckpt_path
            ), f"Checkpoint {model_config.vae.ckpt_path} does not exist."
            self.resume_from_checkpoint = model_config.vae.ckpt_path

        return model

    def _update_scheduled_hyperparams(self):
        if not hasattr(self.model, "latent_losses"):
            warnings.warn(
                "Expected model to have latent losses for weight update while annealing is on."
            )
            return

        for loss_instance in self.model.latent_losses:
            if loss_instance.schedule is not None:
                loss_instance.set_weight_from_schedule(self.global_step)
                self.log(f"kl-{loss_instance.name}-weight", loss_instance.weight)
        return


class GraspLDMTrainer(GraspGenerationTrainer):
    def __init__(self, config: Config = None):
        super().__init__(config, skip_validation=True)
        self._use_vae_ema_model_from_ckpt = config.model.ddm.use_vae_ema_model

    @property
    def _model_type_str(self):
        return "ddm"

    @property
    def _use_qualities(self):
        return self.model.use_grasp_qualities

    def on_train_start(self) -> None:
        """Called when the train begins."""
        self._load_and_freeze_vae_model()
        return

    def generate_grasps(self, pc, metas):
        """Generate grasps from point cloud"""
        grasps = self.model.generate_grasps(pc, metas)

        return grasps

    def _build_model(self, model_config):
        """Build model from config

        Args:
            model_config (Config): Model config
        """
        model = build_model_from_cfg(model_config.ddm)
        model.set_vae_model(build_model_from_cfg(model_config.vae))

        cfg_ddm_ckpt_path = model_config.ddm.ckpt_path
        if cfg_ddm_ckpt_path is not None:
            assert os.path.exists(
                cfg_ddm_ckpt_path
            ), f"Checkpoint {cfg_ddm_ckpt_path} does not exist."
            self.resume_from_checkpoint = cfg_ddm_ckpt_path

        # Check if VAE model weights are to be loaded from ema model
        if default(model_config.ddm.use_vae_ema_model, False):
            self._use_vae_ema_model_from_ckpt = True

        # Freeze VAE model
        model.freeze_vae_model()

        return model

    def _load_and_freeze_vae_model(self):
        """Load VAE model from checkpoint and freeze it

        If no resume checkpoint is specified i.e. ddm and vae weights are not loaded.
        Atleast the vae weights need to be loaded for the ddm training to start.
        First check the config for manually specified `vae.ckpt_path` and
        if not specified, then check for the default vae checkpoint path for `last.ckpt`
        """

        if self.resume_from_checkpoint is None:
            cfg_vae_ckpt_path = self._model_config.vae.ckpt_path
            if cfg_vae_ckpt_path is not None:
                assert os.path.exists(
                    cfg_vae_ckpt_path
                ), f"Checkpoint {cfg_vae_ckpt_path} does not exist."
            else:
                cfg_vae_ckpt_path = (
                    f"{self._experiment.exp_dir}/vae/checkpoints/last.ckpt"
                )

            # Load VAE weights
            state_dict = torch.load(cfg_vae_ckpt_path)["state_dict"]
            if self._use_vae_ema_model_from_ckpt:
                state_dict = fix_state_dict_prefix(
                    state_dict, "ema_model.online_model", ignore_all_others=True
                )
            else:
                state_dict = fix_state_dict_prefix(
                    state_dict, "model", ignore_all_others=True
                )
            self.model.load_vae_weights(state_dict=state_dict)

        return
