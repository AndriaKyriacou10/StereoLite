import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
from .fnet import FeatureNet
from .aggregation import Aggregation2D
from .context_network import ContextNet
from .conv_gru import ConvGRU
from core.utils.utils import InputPadder
from collections import defaultdict

class CustomLiteAnyStereo(nn.Module):
    # StereoLite
    def __init__(self,  scale_right=0.3, layer2=False):
        super(CustomLiteAnyStereo, self).__init__()
        self.fnet = FeatureNet()
       
        self.context_net = ContextNet(scale_right, layer2)
        
        disp_channels = 1
        cv_channels = 192 // 4  # max_disp // 4 = 192 // 4 = 48
        context_channels = 128
        input_channels = disp_channels + cv_channels + context_channels
                
        self.conv_gru = ConvGRU(input_channels=input_channels, hidden_channels=128, kernel_size=3)
        self.disp_head = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
                                       )

        self.cost_agg_2d = Aggregation2D(in_channels=48,
                                    left_att=True,
                                    blocks=[4, 8, 16],
                                    expanse_ratio=4,
                                    backbone_channels=[24, 32, 96, 160])

        self.cost_stem_3d = nn.Sequential(BasicConv(1, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 1, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          )
                
        self.mask_head = nn.Conv2d(in_channels=128, out_channels=144, kernel_size = 3, stride=1, padding=1)

    def upsample_disp(self, disp, mask, scale=4):
        """ Upsample disp field [H//4, W//4] -> [H, W] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(scale * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, scale * H, scale * W)

    def forward(self, left, right, max_disp=192, test_mode=False, kd_mode=False, compute_cost_volume=True, iterations=8):
        left = (2 * (left / 255.0) - 1.0).contiguous()
        right = (2 * (right / 255.0) - 1.0).contiguous()
        
        init_disp, context_features = self.context_net(left, right)
        disparity_predictions = []
        
        if compute_cost_volume:
            features_left = self.fnet(left)
            features_right = self.fnet(right)
            cost_volume = build_correlation_volume(features_left[0], features_right[0], max_disp // 4)


            cv_3d = self.cost_stem_3d(cost_volume[:,None]).squeeze(1)

            cv = self.cost_agg_2d(cv_3d, features_left)
        else:
            cv = torch.zeros((left.shape[0], max_disp // 4, left.shape[2] // 4, left.shape[3] // 4), device=left.device, dtype=left.dtype)
        
        # === REFINEMENT NETWORK === #
        disp = init_disp
        hidden_state = context_features
        for iters in range(iterations):
            input = torch.cat((disp, cv, context_features), dim=1) # keep the context features present in EVERY iteration, not only init
            hidden_state = self.conv_gru(hidden_state, input)
            delta_disp = self.disp_head(hidden_state)
            
            disp = disp + delta_disp # Resolution: 1/4
            
            mask = self.mask_head(hidden_state)
            disp_up = self.upsample_disp(disp, mask)
            
            if not test_mode or iters == iterations - 1:
                disparity_predictions.append(disp_up)

        if test_mode:
            return torch.clamp(disparity_predictions[-1], min=0)
        else:
            return disparity_predictions

class original_LAS(nn.Module):
    def __init__(self, fnet_pretrained=True):
        super(original_LAS, self).__init__()
        self.fnet = FeatureNet(pretrained=fnet_pretrained)

        self.cost_agg_2d = Aggregation2D(in_channels=48,
                                    left_att=True,
                                    blocks=[4, 8, 16],
                                    expanse_ratio=4,
                                    backbone_channels=[24, 32, 96, 160])

        self.cost_stem_3d = nn.Sequential(BasicConv(1, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 1, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          )

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        # Input: Raw-unormalized left RGB image. 
        # Inject raw pixel detail (edges, colour boundaries) that got lost in the 1/4 resolution
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def upsample_disp(self, disp, mask, scale=4):
        """ Upsample disp field [H//4, W//4] -> [H, W] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(scale * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, scale * H, scale * W)

    def forward(self, left, right, max_disp=192, test_mode=False, kd_mode=False):
        left = (2 * (left / 255.0) - 1.0).contiguous()
        right = (2 * (right / 255.0) - 1.0).contiguous()

        features_left = self.fnet(left)
        features_right = self.fnet(right)
        cost_volume = build_correlation_volume(features_left[0], features_right[0], max_disp // 4) # [B, D, H, W] where D = max_disp // 4

        cv_3d = self.cost_stem_3d(cost_volume[:,None]).squeeze(1)

        cv = self.cost_agg_2d(cv_3d, features_left) # Cost aggregated Cost Volume

        prob = F.softmax(cv, dim=1) # dim=1 is disparity dimension
        disp = disparity_regression(prob, max_disp // 4)

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_up = context_upsample(disp * 4., spx_pred.float())

        if test_mode:
            return disp_up
        elif kd_mode:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up, disp_linear * 4.], features_left, features_right
        else:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up, disp_linear * 4.]

    def forward_stabilizer(self, batch_dict, model_stabilizer, kernel_size=50):

        predictions = defaultdict(list)
        for stereo_pair in batch_dict["stereo_video"]:
            left_image_rgb = stereo_pair[None, 0].cuda()  # stereo_pair[None, 0] — same indexing as repo
            right_image_rgb = stereo_pair[None, 1].cuda()  # stereo_pair[None, 1]

            padder = InputPadder(left_image_rgb.shape, divis_by=32)           # reuse core.utils.utils.InputPadder — you already
                                    # import this elsewhere, no need for raft_stereo_utils
            left_image_rgb, right_image_rgb = padder.pad(left_image_rgb, right_image_rgb)

            disp_up = self.forward(left_image_rgb, right_image_rgb, max_disp=192, test_mode=True)
            # ^ no `iters`/autocast — that's RAFT-Stereo-specific, LAS's forward()
            #   doesn't take either

            disp_up = padder.unpad(disp_up)
            predictions["disparity"].append(disp_up)

        predictions["disparity"] = torch.stack(predictions["disparity"], dim=1)

        disparities = model_stabilizer.forward_batch(
            batch_dict["stereo_video"][:, 0].cuda(),
            -predictions["disparity"].squeeze(0),                     # <-- the one line that must NOT match the repo
            kernel_size=kernel_size,
        )
        predictions["disparity"] = disparities.squeeze(1).abs()
        return predictions