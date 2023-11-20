from lightning.pytorch.cli import LightningCLI
from ave3net.datamodule import DataModule
# from data.datamodule import DataModule
from ave3net.model import AVE3Net
import sys
import torch
import os

# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def cli_main():
    torch.autograd.set_detect_anomaly(True)
    cli = LightningCLI(AVE3Net, DataModule, save_config_callback=None)

    # if cli.subcommand == "fit":
    #     cli.trainer.test(cli.model, cli.datamodule, ckpt_path="lightning_logs/version_84/checkpoints/checkpoint.ckpt")


if __name__ == "__main__":
    # sys.tracebacklimit = 0
    cli_main()
