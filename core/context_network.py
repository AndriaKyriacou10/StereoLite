import torch
import torchvision.models as models
import torch.nn as nn
import timm

class ContextNet(nn.Module):
    """
    Outputs Initial Disparity and Context Features at 1/4 scale using a modified ResNet-34 architecture
    """
    
    def __init__(self, scale_right = 0.3):
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
        self.out_conv = nn.Conv2d(64, 129, (1,1), stride=(1,1), padding=(0,0))
    
    def forward(self, left, right):
        concat_imgs = torch.cat([left, right], dim=1)
        x = self.model.conv1(concat_imgs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        out = self.out_conv(x)
        
        init_disparity = out[:, 0:1, :, :]
        context_features = out[:, 1:, :, :]
        
        return init_disparity, context_features
        