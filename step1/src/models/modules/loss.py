import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, predictions, labels):
        # Ensure predictions are in the range (0, 1)
        predictions = torch.clamp(predictions, min=1e-7, max=1 - 1e-7)

        # Compute the binary cross-entropy loss for each sample
        bce_loss = -(labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))

        # Compute the weights using the focal loss formula
        pt = torch.where(labels == 1, predictions, 1 - predictions)  # Probability of the true class
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha to balance positive and negative samples
        balanced_weight = torch.where(labels == 1, self.alpha, 1 - self.alpha)

        # Combine the weights and compute the final loss
        loss = focal_weight * balanced_weight * bce_loss

        # Return the mean loss
        return loss.mean()

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(BinaryCrossEntropyLoss, self).__init__()

        # if weighted cross entropy
        if(args.use_weight_ce):
            # read labels
            path_labels        = os.path.join(args.path_labels, f"folder_{args.fold}")
            path_train_labels  = os.path.join(path_labels, "train.csv")
            train_labels       = pd.read_csv(path_train_labels)

            # group not_rubbish vs rubbish
            train_labels.loc[train_labels['label'].isin(['healthy', 'unhealthy', 'bothcells']), 'label'] = 'not_rubbish'

            # compute class weights
            class_weights      = compute_class_weight(
                                                    class_weight = "balanced",
                                                    classes      = np.array(['not_rubbish', 'rubbish']),
                                                    y            = train_labels.label.tolist()                                                    
                                                    )
            class_weights   = torch.tensor(class_weights, dtype=torch.float32)
            self.pos_weight = class_weights[1]
        else:
            self.pos_weight = None

        # Define the loss function with pos_weight
        self.loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight)

    def forward(self, prediction, target):
        loss = self.loss(prediction, target.float())
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, args):
        super(CombinedLoss, self).__init__()

        self.args = args

        # Cross Entropy Loss
        if(self.args.use_loss_ce):
            self.loss_ce = BinaryCrossEntropyLoss(args)

        # Focal Loss
        if(self.args.use_loss_focal):
            self.loss_focal = FocalLoss()

    def forward(self, prediction, target):

        # Combined loss
        if(self.args.use_loss_ce and self.args.use_loss_focal):
            return self.loss_ce(prediction, target).mean() + self.loss_focal(prediction, target).mean()
        
        # Only Cross-Entropy
        elif(self.args.use_loss_ce):
            return self.loss_ce(prediction, target).mean()
        
        # Only Focal Loss
        else:
            return self.loss_focal(prediction, target).mean()
