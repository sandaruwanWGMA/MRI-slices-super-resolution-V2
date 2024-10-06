# base_options.py
import argparse
import os
from util import util
import torch


class BaseOptions:
    """This class defines options used during both training and test time."""

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Basic parameters
        parser.add_argument("--dataroot", required=True, help="path to datasets")
        parser.add_argument(
            "--name",
            type=str,
            default="mri_super_resolution",
            help="name of the experiment",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="GPU IDs separated by comma (e.g., 0,1)",
        )

        # Model parameters
        parser.add_argument(
            "--model",
            type=str,
            default="resnet",
            help="model type: [resnet | unet | custom]",
        )
        parser.add_argument(
            "--input_nc",
            type=int,
            default=1,
            help="input image channels: 1 for grayscale",
        )
        parser.add_argument(
            "--output_nc",
            type=int,
            default=1,
            help="output image channels: 1 for grayscale",
        )

        # Training parameters
        parser.add_argument(
            "--batch_size", type=int, default=4, help="input batch size"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )

        # Output directories
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)
        opt = parser.parse_args()
        self.print_options(opt)
        self.opt = opt
        return self.opt

    def print_options(self, opt):
        message = "------------ Options -------------\n"
        for k, v in sorted(vars(opt).items()):
            message += "{:>20}: {:<30}\n".format(str(k), str(v))
        message += "-------------- End ----------------\n"
        print(message)

        # Save to the disk
        file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.txt")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)


# Usage
if __name__ == "__main__":
    opt = BaseOptions().parse()
