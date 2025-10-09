import timm
import torch
import torch.nn as nn

class SEResNeXtBackbone(nn.Module):
    def __init__(self, args):
        super(SEResNeXtBackbone, self).__init__()
        
        self.args = args

        # Load pretrain backbone
        features      = timm.create_model('seresnextaa101d_32x8d.sw_in12k_ft_in1k_288', pretrained=False)
        features.load_state_dict(torch.load(args.path_backbone_pretrain, weights_only=True))
        self.features = nn.Sequential(*(list(features.children())[:-2]))

    def forward(self, images):

        # extract features with resnet
        x = self.features(images)

        # global average pooling
        x = x.mean((2, 3))

        return x
