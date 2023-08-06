import os
import os.path as osp

from tqdm import tqdm
import pyvista as pv
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from .base_trainer import BaseTrainer
from .logger import Logger
from .loss import SymmetryDistanceError
from .model import SymmetryNet
from .dataset import ShapeNet, ShapeNetEval
from .util import DataLoaderX


class Trainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

    def prepare_summary_writer(self):
        self.logger = Logger(self.opts)

    def build_loss(self):
        self.losser = SymmetryDistanceError()

    def build_model(self):
        self.model = SymmetryNet(
            self.opts.mlps,
            self.opts.ks,
            self.opts.radius,
            self.opts.rotations,
            self.opts.thre,
            self.opts.nsample,
            self.opts.min_cluster_size,
        ).cuda()

    def prepare_dataset(self):
        self.train_dataset = ShapeNet(
            self.opts.shapenet_root,
            self.opts.split_root,
            cat=self.opts.cat,
            npoints=self.opts.npoints,
        )
        self.val_dataset = ShapeNetEval(self.opts.eval_root, self.opts.npoints)

        self.train_loader = DataLoaderX(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
        )
        self.val_loader = DataLoaderX(
            self.val_dataset,
            batch_size=self.opts.batch_size,
            num_workers=self.opts.num_workers,
        )

    def create_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opts.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.7
        )

    def train_epoch(self):
        self.model.train()
        with tqdm(total=len(self.train_loader), ncols=80) as bar:
            for i, data in enumerate(self.train_loader):
                bar.set_description(f"Train: [{self.epoch}/{self.global_step}]")
                pos = data
                pos = pos.cuda()

                self.optimizer.zero_grad()
                cluster_plane, cluster_batch = self.model(pos)

                loss = self.losser(pos, cluster_plane, cluster_batch)
                self.loss = loss.item()
                loss.backward()
                self.optimizer.step()

                self.logger.summary(
                    self.global_step, log_dict={"train_loss": self.loss}
                )
                self.global_step += 1

                bar.set_postfix(loss=f"{self.loss:.3}")
                bar.update(1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with tqdm(total=len(self.train_loader), ncols=80) as bar:
            bar.set_description(f"Eval: [{self.epoch}]")

            for i, data in enumerate(self.val_loader):
                pos = data
                pos = pos.cuda()

                cluster_plane, cluster_batch = self.model(pos)
                loss = self.losser(pos, cluster_plane, cluster_batch)
                total_loss += loss.item() * self.opts.batch_size

                self.logger.summary(
                    self.global_step, log_dict={"val_loss": loss.item()}
                )

                bar.set_postfix(loss=f"{loss.item():.3}")
                bar.update(1)

            total_loss = total_loss / len(self.val_loader) / self.opts.batch_size
            self.logger.write(f"eval loss: {total_loss}")
