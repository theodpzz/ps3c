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

        # read labels
        path_data         = args.path_data
        path_train_labels = os.path.join(path_data, "isbi2025-ps3c-train-dataset.csv")
        train_labels      = pd.read_csv(path_train_labels)
        
        # rubbish vs non-rubbish
        train_labels['binary_label'] = train_labels['label'].apply(lambda x: 'rubbish' if x == 'rubbish' else 'non-rubbish')
                                                
        # compute class weights
        class_weights = compute_class_weight(
            class_weight = 'balanced',
            classes      = np.unique(train_labels['binary_label']),
            y            = train_labels['binary_label']
        )

        # class weight
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # positive class weight
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]])

        # loss function
        self.loss = nn.BCEWithLogitsLoss(reduction = 'none', pos_weight=pos_weight)

    def forward(self, prediction, target):
        loss = self.loss(prediction, target.float())
        return loss
