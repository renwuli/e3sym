import torch
import os
import random
import numpy as np


class BaseTrainer(object):
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0
        self.epoch = 0
        self.logger = None
        self.model = None
        self.losser = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.prepare_summary_writer()
        self.build_model()
        self.build_loss()
        self.prepare_dataset()
        self.create_optimizers()
        self.seed_torch()

        self.loss = 0.0

    def prepare_summary_writer(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def create_optimizers(self):
        raise NotImplementedError

    def prepare_dataset(self):
        raise NotImplementedError

    def tocuda(self, vars):
        for i in range(len(vars)):
            vars[i] = vars[i].cuda()
        return vars

    def train(self):
        self.logger.write(">>> Starting training ...")
        start_epoch = 0
        if self.opts.ckpt_path is not None:
            start_epoch = self.load(self.opts.ckpt_path) + 1

        for epoch in range(start_epoch, self.opts.train_epochs):
            self.logger.write(f">>> Training epoch: {epoch} ...")
            self.epoch = epoch
            self.train_epoch()

            self.lr_scheduler.step()

            if epoch % self.opts.save_epochs == 0:
                self.save(epoch)
            if epoch % self.opts.eval_epochs == 0:
                self.logger.write(f">>> Evaulating epoch: {epoch} ...")
                self.evaluate()

            if self.opts.reset_dataset_each_epoch:
                self.reset_dataset()

        self.logger.close()

    def reset_dataset(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train_epoch(self):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def save(self, epoch):
        ckpt_dir = os.path.join(self.logger.save_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"{epoch}.ckpt")

        self.logger.write(f">>> Saving checkpoint into {ckpt_path} ...")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            ckpt_path,
        )
        self.logger.write(">>> Save Done")

    def load(self, ckpt_path):
        self.logger.write(f">>> Loading checkpoint from {ckpt_path} ...")

        checkpoints = torch.load(ckpt_path)
        self.logger.write(">>> Done")

        self.model.load_state_dict(checkpoints["model_state_dict"])
        self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

        self.logger.write(">>> Load Done")
        return checkpoints["epoch"]

    def seed_torch(self):
        random.seed(self.opts.seed)
        os.environ["PYTHONHASHSEED"] = str(self.opts.seed)
        np.random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)
        torch.cuda.manual_seed(self.opts.seed)
        torch.cuda.manual_seed_all(self.opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
