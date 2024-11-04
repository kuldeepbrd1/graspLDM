import glob
import os
import shutil
import warnings


class Experiment:
    def __init__(
        self,
        config_path,
        resume_from="last",
        out_dir="output/",
        ckpt_format="ckpt",
        model_suffix="",
        configs_dir_name="configs",
    ) -> None:
        """
        NOTE: configs_dir_name is important to split the category and name of the experiment

        Args:
            config_path (str): path to the config file
            resume_from (str, optional): Checkpoint to resume training from. Defaults to "last".
            out_dir (str, optional): Output directory. Defaults to "output/".
            ckpt_format (str, optional): Checkpoint format. Defaults to "ckpt".
            model_suffix (str, optional): Suffix for the model directory. Defaults to None.
            configs_dir_name (str, optional): Name of the directory containing the configs. Defaults to "configs".
        """
        # Checkpoint format
        self._ckpt_format = ckpt_format

        # Experiment naming
        # Split from configs directory
        relative_config_path = config_path.split(configs_dir_name)[-1].strip("/")
        self.name = os.path.basename(relative_config_path).split(".")[0]
        self.category = os.path.dirname(relative_config_path)

        # Experiment directories
        self.out_dir = out_dir
        self.exp_dir = os.path.join(os.path.abspath(out_dir), self.category, self.name)
        self.model_dir = self.exp_dir + (
            f"/{model_suffix}" if model_suffix is not None else ""
        )

        self.ckpt_dir = os.path.join(self.model_dir, "checkpoints")
        self.log_dir = os.path.join(self.model_dir, "logs")
        self._make_dirs()

        # Make a copy of the config file when training
        self.src_config_path = config_path
        self.dst_config_path = os.path.join(self.model_dir, f"{self.name}.py")

        # Maintain a single config in exp dir. Warn if exists and over-write
        if os.path.isfile(self.dst_config_path):
            warnings.warn(
                f"Existing config file will be over-written: {self.dst_config_path}"
            )
        shutil.copy(self.src_config_path, self.dst_config_path)

        # Resume from checkpoint
        self.resume_from = resume_from

    @property
    def all_checkpoints(self):
        return glob.glob(os.path.join(self.ckpt_dir, f"*.{self._ckpt_format}"))

    @property
    def exists(self):
        return os.path.isdir(self.exp_dir)

    @property
    def last_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"last.{self._ckpt_format}")
        return ckpt_path if os.path.exists(ckpt_path) else None

    @property
    def best_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"best.{self._ckpt_format}")
        return ckpt_path if os.path.exists(ckpt_path) else None

    @property
    def default_resume_checkpoint(self):
        _default_checkpoint = self.last_checkpoint

        if self.resume_from in ("best", "last"):
            ckpt_path = (
                self.last_checkpoint
                if self.resume_from == "last"
                else self.best_checkpoint
            )
        else:
            ckpt_path = self.resume_from

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            _default_checkpoint = ckpt_path
        else:
            # Do nothing and start from scratch
            pass

            # warnings.warn(f"Could not find checkpoint: {ckpt_path}")
            # if _default_checkpoint is None:
            #     warnings.warn(
            #         f"Default checkpoint {_default_checkpoint} also not found."
            #     )
        return _default_checkpoint

    def _make_dirs(self):
        # Warn existing checkpoint directory
        if os.path.exists(self.ckpt_dir):
            warnings.warn(
                f"Experiment Checkpoint directory exists: {self.ckpt_dir} \nCheckpoints may be auto-overwritten by the trainer."
            )
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        return
