import sys
sys.path.append('core')
import argparse

import torch
from thop import profile, clever_format
from core.liteanystereo import CustomLiteAnyStereo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    parser.add_argument('--cost_volume_off', action='store_true', help='Bypass the Feature Network (ReCoVEr Mode)')

    args = parser.parse_args()

    # The standard high-res input shape
    img = torch.randn(1, 3, 384, 1248).cuda()
    
    model = CustomLiteAnyStereo().cuda()
    
    # We must explicitly tell the model whether to compute the cost volume
    # Because args.cost_volume_off is True when the flag is passed, we invert it.
    compute_cv = not args.cost_volume_off
    
    print(f"Profiling with Cost Volume ON: {compute_cv}")

    # Pass the kwarg explicitly in the inputs tuple
    # Note: thop requires inputs to be a tuple.
    macs, params = profile(model, inputs=(img, img, args.max_disp, False, False, compute_cv))
    
    macs, params = clever_format([macs, params], "%.3f")  # Format for readability

    print("Input size:", img.size())
    print("MACs:", macs)
    print("Params:", params)