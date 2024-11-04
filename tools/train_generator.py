import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

from grasp_ldm.trainers import E_Trainers
from grasp_ldm.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for Training Grasp Samplers")
    parser.add_argument("--config", "-c", help="Path to config file", required=True)
    parser.add_argument(
        "--model",
        "-m",
        help="Model type",
        required=True,
        choices=["classifier", "vae", "ddm"],
    )
    parser.add_argument("--root-dir", "-d", help="Root directory")
    parser.add_argument("--num-gpus", "-g", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size per device")
    parser.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Setting this will disable wandb logger and ... TODO",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Make everything deterministic",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Make everything deterministic"
    )

    return parser.parse_args()


def set_deterministic(config, args):
    """Deterministic Run

    Mediate config and CLI args to set deterministic run.
    CLI args take priority and overwrite config.

    In config:

    config.trainer.deterministic =True
    config.seed = 123

    In CLI:
    --deterministic
    --seed 123
    """
    config.trainer.deterministic = (
        False if "deterministic" not in config.trainer else config.trainer.deterministic
    )

    if args.deterministic:
        config.trainer.deterministic = True

    if config.trainer.deterministic:
        if not "seed" in config:
            config.seed = 42
        if args.seed is not None:
            config.seed = args.seed

        from pytorch_lightning import seed_everything

        seed_everything(config.seed, workers=True)
        print(
            "Training will be run in deterministic mode for reproducibility. This might be a bit slower."
        )
    else:
        print(
            "Training is not deterministic. This is a bit faster and alright. If you want deterministic training, set `deterministic=True` in trainer config."
        )

    return config


def main(args):
    ## -- Config --
    config = Config.fromfile(args.config)

    # Overwrite config with args
    ## Overwrite config with args
    # Num gpus
    if args.num_gpus:
        config.trainer.devices = args.num_gpus
        config.trainer.num_workers = args.num_gpus * config.num_workers_per_gpu

    # Batch size
    if args.batch_size:
        config.trainer.batch_size = args.batch_size
        config.data.train.batch_size = args.batch_size

    # Data Root
    if args.root_dir:
        for split in config.data:
            config.data[split].args.data_root_dir = args.root_dir

    # Deterministic
    config = set_deterministic(config=config, args=args)

    ## -- Trainer --
    Trainer = E_Trainers.get(model_type=args.model)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
