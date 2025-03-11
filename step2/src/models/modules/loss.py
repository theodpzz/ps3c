import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight

class MulticlassFocalLoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super().__init__()
        self.gamma     = args.gamma
        self.reduction = reduction
        
        # read labels
        path_labels        = args.path_labels
        path_train_labels  = os.path.join(path_labels, "train.csv")
        train_labels       = pd.read_csv(path_train_labels)

        # only conserve healthy and unhealthy proportion to compute weights
        train_labels = train_labels.loc[train_labels['label'].isin(['healthy', 'unhealthy'])]
        
        # compute class weights
        class_weights      = compute_class_weight(
                                                class_weight = "balanced",
                                                classes      = np.array(['healthy', 'unhealthy']),
                                                y            = train_labels.label.tolist()                                                    
                                                )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, logits, labels):
        # Convert class indices to one-hot if necessary
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes=logits.size(-1))
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Calculate focal loss
        ce_loss = -labels * torch.log(probs + 1e-8)  # Add small epsilon for numerical stability
        
        # Apply alpha weighting
        alpha_weight = self.class_weights.view(1, -1) * labels
        ce_loss = alpha_weight * ce_loss
        
        # Apply focal term
        pt = (labels * probs).sum(dim=-1)  # Get the probability of the true class
        focal_term = (1 - pt) ** self.gamma
        
        # Apply focal term to each sample
        focal_loss = focal_term.unsqueeze(-1) * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(CrossEntropyLoss, self).__init__()

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

class CombinedLoss(nn.Module):
    def __init__(self, args):
        super(CombinedLoss, self).__init__()

        self.args = args

        # Cross Entropy Loss
        if(self.args.use_loss_ce):
            self.loss_ce = CrossEntropyLoss(args)

        # Focal Loss
        if(self.args.use_loss_focal):
            self.loss_focal = MulticlassFocalLoss(args)

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
