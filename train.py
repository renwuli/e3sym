import os
import os.path as osp

from tqdm import tqdm
import pyvista as pv
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from lib.trainer import Trainer
from lib.logger import Logger
from lib.loss import SymmetryDistanceError
from lib.model import SymmetryNet
from lib.dataset import ShapeNet, ShapeNetEval
from lib.util import DataLoaderX


if __name__ == "__main__":
    import yaml
    import argparse
    from easydict import EasyDict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    f = open(f"{args.config}")
    opt = EasyDict(yaml.safe_load(f))
    print(opt)

    trainer = Trainer(opt)
    trainer.train()
