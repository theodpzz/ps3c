import os
import time
import torch
import warnings
import argparse
import numpy as np

from src.modules.dataset import getDatasets
from src.modules.dataloader import prepare_dataloader

from src.models.model import Model
from src.modules.optimizer import get_optimizer_scheduler
from src.modules.utils import make_output_dir, parse_yaml

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
        train_data: DataLoader,
        valid_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        args,
    ) -> None:
        self.logs_, self.metrics_train_, self.metrics_val_ = {}, {}, {}
        self.args              = args
        self.device            = self.args.device
        self.model             = model.to(self.device)
        self.train_data        = train_data
        self.valid_data        = valid_data
        self.test_data         = test_data
        self.optimizer         = optimizer
        self.lr_scheduler      = lr_scheduler
        self.sizes             = (len(self.train_data), len(self.test_data))
        self.num_batches       = len(train_data)
        self.learning_rates    = []
        self.args.size_train   = len(train_data)
        self.args.size_valid   = len(valid_data)
        self.args.size_test    = len(test_data)
        self.accumulation_step = self.args.batch_size_sim / self.args.batch_size_train


    def accum_log_(self, epoch, logs):
        self.logs_[epoch] = logs

    def accum_log_metric(self, epoch, metrics):
        self.metrics_train_[epoch] = metrics['train']
        self.metrics_val_[epoch]   = metrics['val']

    def _run_batch(self, i, epoch, volumes, labels, save_pred):
        step = i + epoch * self.num_batches

        # forward pass
        self.optimizer.zero_grad()
        predictions, loss_forward = self.model(volumes, labels) # (batch_size, num_pathologies)

        # accumulate loss
        loss = loss_forward / self.accumulation_step

        # backward pass
        loss.mean().backward()

        # optimizer step
        if((i+1) % self.accumulation_step == 0):
            self.optimizer.step()
            self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # scheduler step
            if(self.args.use_scheduler): 
              self.lr_scheduler.step()

        loss_value = loss_forward.mean().detach().item()
        del volumes, labels, predictions, loss
        return {'loss_train': loss_value}

    def _run_batch_val(self, volumes, labels):
        with torch.no_grad():
            predictions, loss = self.model(volumes, labels)
        loss_value = loss.mean().detach().item()
        del volumes, labels, predictions, loss
        return {'loss_val': loss_value}

    def _run_batch_test(self, volumes, labels):
        with torch.no_grad():
            predictions, loss = self.model(volumes, labels)
        loss_value = loss.mean().detach().item()
        del volumes, labels, predictions, loss
        return {'loss_test': loss_value}
    

    def _run_epoch(self, epoch):
        logs, metrics = {}, {}

        # training step
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"Train | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        self.start_first_batch = time.time()
        with tqdm(total=len(self.train_data)) as pbar:
          for i, data in enumerate(self.train_data):
            volumes_id, volumes, labels = data
            volumes   = volumes.to(self.device)
            labels    = labels.to(self.device)

            loss   = self._run_batch(i, epoch, volumes, labels, (epoch % self.validate_every == 0))
            del volumes, labels
            logs = accum_log(logs, loss)
            pbar.update(1)

        # Validation step
        for i, (sample_names, volumes, labels) in enumerate(self.valid_data):
            volumes = volumes.to(self.device)
            labels  = labels.to(self.device)

            loss, probabilities = self._run_batch_val(volumes, labels)
            samples_names_valid.append(sample_names)
            probabilities_valid.append(probabilities)
            logs   = accum_log(logs, loss)
            del volumes, labels

        # Test step
        for i, (sample_names, volumes, labels) in enumerate(self.test_data):
          volumes = volumes.to(self.device)
          labels  = labels.to(self.device)

          loss, probabilities = self._run_batch_test(volumes, labels)
          samples_names_test.append(sample_names)
          probabilities_test.append(probabilities)
          logs   = accum_log(logs, loss)
          del volumes, labels

        self.accum_log_(epoch, logs)

        return logs, metrics

    def _save_checkpoint(self, epoch):
        
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
            logs, metrics = self._run_epoch(epoch)
            self._save_checkpoint(epoch)

def main(args):

    # directory where results are saved
    args.save_dir = make_output_dir(args)

    # device
    args.device = torch.device(f'cuda:{args.gpu_index}')

    # dataloader
    trainset, validset, testset             = getDatasets(args)
    train_loader, valid_loader, test_loader = prepare_dataloader(trainset, validset, testset, args)
    args.num_batches                        = len(train_loader)

    # model
    model = Model(args)

    # optimizer, scheduler
    optimizer, lr_scheduler = get_optimizer_scheduler(args, model, num_batches=args.num_batches)

    ### TRAINER
    total_epochs = args.epochs
    trainer = Trainer(model, train_loader, valid_loader, test_loader, optimizer, lr_scheduler, args)
    trainer.train(total_epochs)
    

if __name__ == "__main__":
    
    # read config file
    parser = argparse.ArgumentParser(description='Script with parameters from JSON file')
    parser.add_argument('--yaml_file', type=str, default='configs/default.yaml', help='YAML file containing parameters')
    args = parser.parse_args()

    # from the argparse, extract yaml path and fold
    yaml_file = args.yaml_file

    # read the config file from the yaml path
    params = parse_yaml(args.yaml_file)
    args = argparse.Namespace(**params)
    args.yaml_file = yaml_file

    # multi processing config
    main(args)
