import torch.nn as nn

from step1.src.models.modules.loss import Loss

from step2.src.models.modules.convnextv2 import ConvNeXtV2Backbone
from step2.src.models.modules.swinv2 import SwinTransformerV2Backbone
from step2.src.models.modules.seresnext import SEResNeXtBackbone

def get_backbone(args):
    """Returns backbone of interest.
    """
    if args.model_name == "convnextv2":
        return ConvNeXtV2Backbone(args)
        
    elif args.model_name == "swin":
        return SwinTransformerV2Backbone(args)
        
    elif args.model_name == "seresnext":
        return SEResNeXtBackbone(args)

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
        self.activation = nn.Sigmoid()

        # loss function
        self.loss = Loss(args)

    def getloss(self, prediction, target):
        loss = self.loss(prediction, target)
        return loss  

    def forward(self, images, labels):

        # extract features with resnet
        x = self.backbone(images)

        # projection head
        x = self.projection(x)

        # classification head
        logits = self.classifier(x).squeeze(1)
        
        # Activation function for probabilities
        probabilities = self.activation(logits)
        
        # Compute loss
        loss = self.getloss(logits, labels)
        
        return probabilities, loss
