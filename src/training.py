import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src import utils


class Trainer:
    def __init__(self, config) -> None:
        utils.set_seed(config.experience.seed)

        # dataloader
        self.dataset_dir = config.experience.data_dir
        self.dataset = config.dataset

        # transforms
        self.train_transform = utils.get_transform(config.transform, is_train=True)
        self.test_transform = utils.get_transform(config.transform, is_train=False)

        # exp paths
        self.exp_dir = Path(config.experience.exp_dir)

        # training params
        self.net = utils.get_model(
            config.encoder,
            config.decoder,
            config.experience.quantize_level,
            config.experience.pretrained_weights,
        )
        self.criterion = utils.get_criterion(config.loss)
        self.optimizer = utils.load_obj(config.optimizer.name)(
            self.net.parameters(), **config.optimizer.args
        )
        self.scheduler = utils.load_obj(config.scheduler.name)(
            self.optimizer, **config.scheduler.args
        )
        self.clip_value = config.experience.clip_value

        self.device = torch.device(config.experience.device)
        self.n_epochs = config.experience.n_epochs
        self.batch_size = config.experience.batch_size
        self.n_workers = config.experience.n_workers
        self.use_amp = config.experience.use_amp

        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.use_amp and torch.cuda.is_available()
            else None
        )

        self.net.to(self.device)

    def _get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = utils.load_obj(self.dataset.name)(
            Path(self.dataset.args.root), stage="train", transform=self.train_transform
        )
        eval_dataset = utils.load_obj(self.dataset.name)(
            Path(self.dataset.args.root), stage="val", transform=self.test_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )
        self.train_steps = len(train_loader)
        self.eval_steps = len(eval_loader)

        return train_loader, eval_loader

    def train_step(
        self,
        epoch: int,
        data_loader: DataLoader,
    ):
        self.net.train()

        # train step
        result_loss = 0

        pbar = tqdm(data_loader)
        for bid, (images, targets) in enumerate(pbar):
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                images = images.to(self.device)
                targets = targets.to(self.device)
                output = self.net(images)

                loss = sum(c[0](output, images) * c[1] for c in self.criterion)

            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.clip_value:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clip_value)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            result_loss += loss.item()

            pbar.set_description(f"Epoch: {epoch}")
            pbar.set_postfix_str(f"Train Loss: {(result_loss / (bid + 1)):.6f}")

        result_loss /= self.train_steps

        return result_loss

    def eval_step(
        self,
        epoch: int,
        data_loader: DataLoader,
    ):
        self.net.eval()

        with torch.no_grad():
            result_loss = 0
            pbar = tqdm(data_loader)
            for bid, (images, targets) in enumerate(pbar):
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    output = self.net(images)
                    loss = sum(c[0](output, images) * c[1] for c in self.criterion)

                result_loss += loss.item()

                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix_str(f"Test Loss: {(result_loss / (bid + 1)):.6f}")

            result_loss /= self.eval_steps

        return result_loss

    def train(self):
        epoch = 0
        best_loss = np.inf
        train_loader, eval_loader = self._get_dataloaders()

        tb_writer = SummaryWriter(self.exp_dir / "tensorboard_logs")

        logging.info(f"Training for {self.n_epochs - epoch} epochs.")

        while epoch < self.n_epochs:
            # train
            train_loss = self.train_step(epoch, train_loader)

            # Logging train
            tb_writer.add_scalar("Train/Loss", train_loss, epoch)

            # evaluate
            eval_loss = self.eval_step(epoch, eval_loader)

            # Logging valid
            tb_writer.add_scalar("Eval/Loss", eval_loss, epoch)
            tb_writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # scheduler step
            self.scheduler.step()

            # Best net save
            # if metrics["best_score"] < eval_score:
            if eval_loss < best_loss:
                best_loss = eval_loss
                checkpoint = {
                    "epoch": epoch,
                    "loss": best_loss,
                    "encoder": self.net.encoder.state_dict(),
                    "decoder": self.net.decoder.state_dict(),
                    "criterion_state": self.criterion.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None,
                }
                scripted = torch.jit.trace(
                    self.net, train_loader.dataset[0][0].unsqueeze(0).to(self.device)
                )
                torch.jit.save(scripted, self.exp_dir / "best_model.torchscript")
                torch.save(checkpoint, self.exp_dir / "best_checkpoint.pth")

            # save last
            checkpoint = {
                "epoch": epoch,
                "loss": eval_loss,
                "encoder": self.net.encoder.state_dict(),
                "decoder": self.net.decoder.state_dict(),
                "criterion_state": self.criterion.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
            }

            scripted = torch.jit.trace(
                self.net, train_loader.dataset[0][0].unsqueeze(0).to(self.device)
            )
            torch.jit.save(scripted, self.exp_dir / "last_model.torchscript")
            torch.save(checkpoint, self.exp_dir / "last_checkpoint.pth")

            epoch += 1

        tb_writer.close()
        logging.info(f"Training was finished")
