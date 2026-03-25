class TrainerEMAMixin:
    """Mixin for EMA model management in the trainer.

    Activated when ``ema`` is present in the trainer config.
    """

    def configure_ema(self, trainer_config):
        from ema_pytorch import EMA

        if "ema" in trainer_config and trainer_config.ema:
            ema_config = self.get_ema_config(trainer_config)
            self.ema_model = EMA(self.model, **ema_config).to(self.device)
        else:
            self.ema_model = None

    def get_ema_config(self, trainer_config):
        """Build EMA kwargs from trainer config, falling back to sensible defaults."""
        defaults = dict(beta=0.990, update_after_step=1000, update_every=5)
        for key in list(defaults):
            val = getattr(trainer_config.ema, key, None)
            if val is not None:
                defaults[key] = val
        return defaults
