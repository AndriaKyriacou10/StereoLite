import torch
import torchvision.models as models
import torch.nn as nn
import timm

class ConvGRU(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """ Input: Initial Disparity from Context Network, Cost Volume/Zero Tensor (refined during the iterations)
            Hidden State: Context Features
        """
        super().__init__()
        self.input_channels = input_channels + 48
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        self.padding = (kernel_size - 1) // 2
        
        # Instead of having three separate convolutions for reset, update, and candidate gate, concatinate reset and update to maximize efficiency
        self.gated_conv = nn.Conv2d(self.input_channels + self.hidden_channels, 2*self.hidden_channels, kernel_size, 1, self.padding)
        
        # Candidate requires new reset gate -> Needs to be computed separately 
        self.candidate_conv = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels, kernel_size, 1, self.padding)
    
    def forward(self, hidden_state, input):
        """ Input: Initial Disparity from Context Network, Cost Volume/Zero Tensor (refined during the iterations)
            Hidden State: Context Features
        """
        combined = torch.cat((input, hidden_state), dim=1)
        
        # Update and Rest Gate
        gated_conv_out = self.gated_conv(combined)
        z_t, r_t = torch.chunk(gated_conv_out, 2, 1)
        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)
        
        # Candidate Hidden State
        hidden_comb = torch.cat((input, r_t * hidden_state), dim=1)
        
        candidate_hidden = self.candidate_conv(hidden_comb)
        candidate_hidden = torch.tanh(candidate_hidden)
        
        # Updated Hidden State
        hidden_state = (1 - z_t) * candidate_hidden + (z_t * hidden_state)
        
        return hidden_state
        
        