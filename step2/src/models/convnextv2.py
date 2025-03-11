import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ConvNextV2ForImageClassification

class ConvNeXtV2Backbone(nn.Module):
    def __init__(self, args):
        super(ConvNeXtV2Backbone, self).__init__()
        
        self.args = args

        # Load ConvNeXt pretrained features
        self.features = ConvNextV2ForImageClassification.from_pretrained(args.path_convnext).convnextv2

    def forward(self, images):

        # extract features with resnet
        x = self.features(images)

        # global average pooling
        x = x.last_hidden_state.mean(dim=(2, 3))

        return x
