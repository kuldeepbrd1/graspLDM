from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class TrainerEMAMixin:
    """Mixin for EMA model management in trainer

    The idea is to have all this functionality completely hidden and disconnected
    from the main trainer class. Only activated when specified in trainer config
    """

    def configure_ema(self, trainer_config):
        from ema_pytorch import EMA

        if hasattr(trainer_config, "ema"):
            if trainer_config.ema:
                ema_config = self.get_ema_config(trainer_config)
                self.ema_model = EMA(self.model, **ema_config).to(self.device)
            else:
                self.ema_model = None

    def get_ema_config(self, trainer_config):
        """Get EMA config

        Args:
            trainer_config (dict): trainer config

        Returns:
            dict: EMA config
        """

        def check_key(q_dict, q_key):
            if key in q_dict:
                if q_dict[key] is not None:
                    return True
            return False

        ema_config = dict(
            beta=0.990,
            update_after_step=1000,
            update_every=5,
        )

        for key in list(ema_config):
            if check_key(trainer_config.ema, key):
                ema_config[key] = getattr(trainer_config.ema, key)

        return ema_config

    def get_ema_callback(self):
        # Unused because this requires additional checkpoint to be saved
        # No good way to disconnect from how we implement normal checkpoints in derived class
        return self.EMAModelCheckpoint(
            save_top_k=1,
            monitor="loss",
            mode="min",
            dirpath=self._experiment.ckpt_dir,
            filename="ema-{step}",
            save_weights_only=True,
            every_n_train_steps=1000,
        )
