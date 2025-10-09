import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class ResNetBackbone(nn.Module):
    def __init__(self, args):
        super(ResNetBackbone, self).__init__()
        
        # Load ResNet features
        self.args = args
        if(self.args.size == "18"):
            resnet = models.resnet18(weights=None)
        elif(self.args.size == "50"):
            resnet = models.resnet50(weights=None)
        resnet.load_state_dict(torch.load(args.path_resnet))
        self.features = nn.Sequential(*(list(resnet.children())[:-args.layer]))

    def forward(self, images, labels):

        # extract features with resnet
        x = self.features(images)

        # global average pooling
        x = x.mean(dim=(2, 3))
        
        return x
