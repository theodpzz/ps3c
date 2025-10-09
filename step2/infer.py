import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd

from step1.src.modules.dataset import getEvaluationDatasets
from step1.src.modules.dataloader import prepare_dataloader

from step1.src.models.model import Model
from step1.src.modules.utils import parse_yaml

from tqdm import tqdm
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

class Inferer:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        eval_loader: torch.optim.Optimizer,
        args,
    ) -> None:
        self.args              = args
        self.device            = self.args.device
        self.model             = model.to(self.device)
        self.test_loader       = test_loader
        self.eval_loader       = eval_loader

        self.test_names, self.eval_names = [], []
        self.test_predictions, self.eval_predictions = [], []

    def make_csv(self, split_name):
        if split_name == "test":
            names_list = self.test_names
            arr_list   = self.test_predictions
        elif split_name == "eval":
            names_list = self.eval_names
            arr_list   = self.eval_predictions

        all_names = [name for tup in names_list for name in tup]
        all_probs = np.vstack(arr_list)
        df        = pd.DataFrame({"names": all_names, "probability_healthy": all_probs[:, 0], "probability_unhealthy": all_probs[:, 1]})
        return df

    def save_predictions(self):

        test_csv = self.make_csv("test")
        eval_csv = self.make_csv("eval")

        path_config      = self.args.path_config
        path_predictions = os.path.join(path_config, f"predictions")
        os.makedirs(path_predictions, exist_ok=True)
        
        path_test        = os.path.join(path_predictions, f"test.csv")
        path_eval        = os.path.join(path_predictions, f"eval.csv")

        test_csv.to_csv(path_test)
        eval_csv.to_csv(path_eval)

    def _run_batch_test(self, images_id, images):

        # forward pass
        probabilities = self.model.infer(images)

        self.test_predictions.append(probabilities.detach().cpu().numpy())
        self.test_names.append(images_id)
    
    def _run_batch_eval(self, images_id, images):

        # forward pass
        probabilities = self.model.infer(images)

        self.eval_predictions.append(probabilities.detach().cpu().numpy())
        self.eval_names.append(images_id)

    def step_test(self):
        with tqdm(total=len(self.test_loader)) as pbar:
            for i, data in enumerate(self.test_loader):
                images_id, images, _ = data
                images   = images.to(self.device)
                self._run_batch_test(images_id, images)
                pbar.update(1)

    def step_eval(self):
        with tqdm(total=len(self.eval_loader)) as pbar:
            for i, data in enumerate(self.eval_loader):
                images_id, images, _ = data
                images   = images.to(self.device)
                self._run_batch_eval(images_id, images)
                pbar.update(1)

    def infer(self):

        # evaluation mode
        self.model.eval()

        with torch.no_grad():
            self.step_test()
            self.step_eval()

        self.save_predictions()

def infer_step_2(yaml_file):

    # read the config file from the yaml path
    params         = parse_yaml(yaml_file)
    args           = argparse.Namespace(**params)
    args.yaml_file = yaml_file

    # directory where results are saved
    args.save_dir = args.path_config

    # device
    args.device = torch.device(f'cuda:{args.gpu_index}')

    # dataloader
    test, eval  = getEvaluationDatasets(args)
    test_loader = prepare_dataloader(test, args, "test")
    eval_loader = prepare_dataloader(eval, args, "eval")

    # model
    model = Model(args)
    model.load()

    ### TRAINER
    inferer = Inferer(model, test_loader, eval_loader, args)
    inferer.infer()