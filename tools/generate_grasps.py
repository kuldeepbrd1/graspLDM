import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
sys.path.append((os.getcwd()))

from tools.inference import Conditioning, InferenceLDM, InferenceVAE, ModelType


def parse_args():
    parser = argparse.ArgumentParser(description="Grasp Generation Script")
    parser.add_argument(
        "--exp_path", type=str, required=True, help="Path to experiment checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/ACRONYM", help="Root directory for data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["VAE", "LDM"],
        default="VAE",
        help="Model type to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Data split to use")
    parser.add_argument(
        "--num_grasps", type=int, default=20, help="Number of grasps to generate"
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--no_ema",
        action="store_false",
        dest="use_ema_model",
        help="Disable EMA model usage",
    )
    parser.add_argument(
        "--num_samples", type=int, default=11, help="Number of samples to generate"
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        choices=["unconditional", "class", "region"],
        default="unconditional",
        help="Type of conditioning to use",
    )
    parser.add_argument(
        "--condition_value",
        type=int,
        help="Value for conditioning (class label or region ID)",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="Number of inference steps for LDM",
    )
    return parser.parse_args()


def setup_model(args):
    exp_name = os.path.basename(args.exp_path)
    exp_out_root = os.path.dirname(args.exp_path)

    if args.mode == "LDM":
        model = InferenceLDM(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            use_elucidated=False,
            data_root=args.data_root,
            load_dataset=True,
            num_inference_steps=args.inference_steps,
            use_fast_sampler=False,
            data_split=args.split,
            use_ema_model=args.use_ema_model,
        )
        print(
            f"Trained using noise schedule: beta0 = {model.model.diffusion_model.beta_start} ; betaT = {model.model.diffusion_model.beta_end}"
        )
    elif args.mode == "VAE":
        model = InferenceVAE(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            data_root=args.data_root,
            load_dataset=True,
            data_split=args.split,
            use_ema_model=args.use_ema_model,
        )
    return model


def get_conditioning(args) -> Tuple[Optional[Conditioning], Optional[int]]:
    if args.conditioning == "unconditional":
        return Conditioning.UNCONDITIONAL, None
    elif args.conditioning == "class":
        if args.condition_value is None:
            raise ValueError("Must provide --condition_value for class conditioning")
        return Conditioning.CLASS_CONDITIONED, args.condition_value
    elif args.conditioning == "region":
        if args.condition_value is None:
            raise ValueError("Must provide --condition_value for region conditioning")
        return Conditioning.REGION_CONDITIONED, args.condition_value
    return None, None


def main():
    args = parse_args()
    model = setup_model(args)
    condition_type, conditioning = get_conditioning(args)

    for _ in range(args.num_samples):
        data_idx = np.random.randint(0, len(model.dataset))

        # Skip conditioning for VAE mode
        if args.mode == "VAE":
            condition_type = Conditioning.UNCONDITIONAL
            conditioning = None

        results = model.infer(
            data_idx=data_idx,
            num_grasps=args.num_grasps,
            visualize=args.visualize,
            condition_type=condition_type,
            conditioning=conditioning,
        )

        if args.visualize:
            results.show(line_settings={"point_size": 10})


if __name__ == "__main__":
    main()
