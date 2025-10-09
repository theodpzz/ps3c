import os
import time
import torch
import warnings
import argparse

from step1.src.modules.dataset import getTrainDataset
from step1.src.modules.dataloader import prepare_dataloader

from step1.src.models.model import Model
from step1.src.modules.optimizer import get_optimizer_scheduler
from step1.src.modules.utils import make_output_dir, parse_yaml
from step1.src.modules.figures import plot_losses

from tqdm import tqdm
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        args,
    ) -> None:
        self.logs_, self.metrics_train_, self.metrics_val_ = {}, {}, {}
        self.args              = args
        self.device            = self.args.device
        self.model             = model.to(self.device)
        self.dataloader        = dataloader
        self.optimizer         = optimizer
        self.lr_scheduler      = lr_scheduler
        self.num_batches       = len(dataloader)
        self.save_after        = args.save_after
        self.accumulation_step = self.args.batch_size_sim / self.args.batch_size_train


    def accum_log_(self, epoch, logs):
        self.logs_[epoch] = logs

    def plot_losses_(self):
        plot_losses(self.args, self.logs_)

    def _run_batch(self, i, images, labels):

        # forward pass
        _, loss_forward = self.model(images, labels)

        # accumulate loss
        loss = loss_forward / self.accumulation_step

        # backward pass
        loss.mean().backward()

        # optimizer step
        if((i+1) % self.accumulation_step == 0):
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        loss_value = loss_forward.mean().detach().item()

        return {'loss_train': loss_value}
    
    def step_train(self):
        logs = {}
        with tqdm(total=len(self.dataloader)) as pbar:
            for i, data in enumerate(self.dataloader):
                _, images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                loss   = self._run_batch(i, images, labels)
                del images, labels
                logs = accum_log(logs, loss)
                pbar.update(1)
        return logs

    def _run_epoch(self, epoch):

        # model training model
        self.model.train()

        # log
        print(f"Train | Epoch {epoch} | Steps: {len(self.dataloader)}")

        # train
        logs = self.step_train()

        # accum log
        self.accum_log_(epoch, logs)

        # plot losses
        self.plot_losses_()

        return logs

    def _save_checkpoint(self, epoch):
        
        if epoch >= self.save_after:

            # checkpoint with model weights, optimizer, scheduler states and epoch
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'epoch': epoch,
            }

            # save it
            PATH = os.path.join(self.args.save_dir, f"checkpoints/checkpoint_{epoch}.pt")
            torch.save(checkpoint, PATH)

    def adjust_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr

    def train(self, max_epochs: int):
        print(f'\n>>> Starting to train...\n')
        self.start_epoch = time.time()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self._save_checkpoint(epoch)

def train_step_1(yaml_file):

    # read the config file from the yaml path
    params         = parse_yaml(yaml_file)
    args           = argparse.Namespace(**params)
    args.yaml_file = yaml_file

    # directory where results are saved
    args = make_output_dir(args)

    # device
    args.device = torch.device(f'cuda:{args.gpu_index}')

    # dataloader
    dataset          = getTrainDataset(args)
    dataloader       = prepare_dataloader(dataset, args, "train")
    args.num_batches = len(dataloader)

    # model
    model = Model(args)

    # optimizer, scheduler
    optimizer, lr_scheduler = get_optimizer_scheduler(args, model, num_batches=args.num_batches)

    ### TRAINER
    total_epochs = args.epochs
    trainer = Trainer(model, dataloader, optimizer, lr_scheduler, args)
    trainer.train(total_epochs)