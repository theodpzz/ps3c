import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from .modules.loss import CombinedLoss

from .resnet import ResNetBackbone
from .convnextv2 import ConvNeXtV2Backbone
from .swin import SwinTransformerV2Backbone
from .seresnext import SEResNeXtBackbone

def get_backbone(args):
    elif(args.model_name == "convnextv2"):
        return ConvNeXtV2Backbone(args)
    elif(args.model_name == "swin"):
        return SwinTransformerV2Backbone(args)
    elif(args.model_name == "seresnext"):
        return SEResNeXtBackbone(args)
    elif(args.model_name == "resnet"):
        return ResNetBackbone(args)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # Argument
        self.args = args

        # Initialize backbone
        self.backbone = get_backbone(args)

        # Projection head
        self.projection = nn.Sequential(nn.Linear(self.args.dim_embed, 512), nn.ReLU(True), nn.Dropout(0.5))

        # Classification head
        self.classifier = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(128, args.num_classes)
                    )

        # Softmax
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.loss = CombinedLoss(args)

    def getloss(self, prediction, target):
        # compute BCE Loss
        loss = self.loss(prediction, target)
        return loss  

    def forward(self, images, labels):

        # extract features with resnet
        x = self.backbone(images)

        # projection head
        x = self.projection(x)

        # classification head
        logits = self.classifier(x).squeeze(1)
        
        # Softmax is usually applied in the loss function, but for clarity:
        probabilities = self.sigmoid(logits)
        
        # Compute loss
        loss = self.getloss(logits, labels)
        
        return probabilities, loss
