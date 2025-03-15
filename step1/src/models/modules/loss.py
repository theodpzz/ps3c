import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        # if weighted cross entropy
        if(args.use_weight_ce):
            # read labels
            path_labels        = os.path.join(args.path_labels, f"folder_{args.fold}")
            path_train_labels  = os.path.join(path_labels, "train.csv")
            train_labels       = pd.read_csv(path_train_labels)
            train_labels       = train_labels.loc[train_labels['label'].isin(['healthy', 'unhealthy'])]
                                                    
            # compute class weights
            class_weights      = compute_class_weight(
                                                    class_weight = "balanced",
                                                    classes      = np.array(['healthy', 'unhealthy']),
                                                    y            = train_labels.label.tolist()                                                    
                                                    )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

        # CE loss
        self.loss = nn.CrossEntropyLoss(reduction = 'none', weight=self.class_weights)

    def forward(self, prediction, target):
        loss = self.loss(prediction, target)
        return loss
