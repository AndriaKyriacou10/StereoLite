import torch
import torchvision.models as models
import torch.nn as nn
import timm
from .submodule import FPNLayer
class ContextNet(nn.Module):
    """
    Outputs Initial Disparity and Context Features at 1/4 scale using a modified ResNet-34 architecture
    """
    
    def __init__(self, scale_right = 0.3, layer2=False):
        super().__init__()
        self.model = models.resnet34(weights='DEFAULT', progress=True)
        
        # Get old conv1 weights
        old_weights = self.model.conv1.weight.data
        
        # Left, Right Image Frames are concatenated -> In Channels = 6 (RGB + RGB)
        self.model.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        nn.init.zeros_(self.model.conv1.weight)
        self.model.conv1.weight.data[:, :3, :, :] = old_weights  # Copy weights for left image
        
        # Initialize right image weights with a subset of the left image weights
        
        self.model.conv1.weight.data[:, 3:, :, :] = old_weights * scale_right
        
        # Output Channels = 1 (Initial Disparity) + 128 (Context Features / Hidden State) = 129
        init_channels = 64
        
        if layer2:
            self.fpn = FPNLayer(chan_low=128, chan_high=64)

        self.out_conv = nn.Conv2d(init_channels, 129, (1,1), stride=(1,1), padding=(0,0))
        
        self.layer2 = layer2
        
        if layer2:
            del self.model.layer3
            del self.model.layer4
            del self.model.avgpool
            del self.model.fc
        else:
            del self.model.layer2
            del self.model.layer3
            del self.model.layer4
            del self.model.avgpool
            del self.model.fc
    
    def forward(self, left, right):
        concat_imgs = torch.cat([left, right], dim=1)
        x = self.model.conv1(concat_imgs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        c1 = self.model.layer1(x) # 64 channels @ 1/4 
        
        if self.layer2:
            c2 = self.model.layer2(c1) # 128 channels @ 1/8 
            p = self.fpn(c2, c1) # 64 channels @ 1/4
            out = self.out_conv(p) # 129 channels @ 1/4
        else:
            out = self.out_conv(c1) # 129 channels @ 1/4

        
        init_disparity = out[:, 0:1, :, :]
        context_features = out[:, 1:, :, :]
        
        return init_disparity, context_features
