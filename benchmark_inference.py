import torch
import argparse, csv
import torch
import torch.nn as nn
from core.liteanystereo import CustomLiteAnyStereo, original_LAS
import time
import numpy as np
from core.utils.utils import InputPadder
import flops_count
from ptflops import get_model_complexity_info
import re

@torch.no_grad()
def benchmark_forward(model, device, shape=(1, 3, 375, 1242),
                       compute_cost_volume=True, warmup=15, iters=50, is_original=False, compute_flops=False):
    model.eval()
    torch.backends.cudnn.benchmark = True  # let it pick fast kernels for fixed shape

    left = torch.randn(shape, device=device)
    right = torch.randn(shape, device=device)
    
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    
    if compute_flops:
        _, C, H, W = left.shape          # padded dims

        def input_constructor(input_res):
            c, h, w = input_res
            kwargs = {
                "left":  torch.randn(1, c, h, w, device=device),
                "right": torch.randn(1, c, h, w, device=device),
                "test_mode": True,
            }
            if not is_original:
                kwargs["compute_cost_volume"] = compute_cost_volume
                kwargs["iterations"] = 8
            return kwargs

        macs, params = get_model_complexity_info(
            model, (C, H, W),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=True,
        )
        print(f"{macs/1e9:.3f} GMACs | {2*macs/1e9:.3f} GFLOPs | {params/1e6:.3f}M params")
        return
        
    def run():
        if is_original:
            return model(left, right, test_mode=True)
        else:
            return model(left, right, test_mode=True, compute_cost_volume=compute_cost_volume)

    
    # warm-up: not timed
    for _ in range(warmup):
        _ = run()
    
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    time_ms = []
    
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = run()
        torch.cuda.synchronize()
        elapsed_time_ms = (time.perf_counter() - start) * 1000
        time_ms.append(elapsed_time_ms)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    avg_runtime = float(np.mean(time_ms))
    parameter_count = sum(p.numel() for p in model.parameters())/1e6
    # flops = FlopCountAnalysis(model, (left, right))
    return avg_runtime, float(np.std(time_ms)), peak_mb, parameter_count


def parse_args():
    p = argparse.ArgumentParser(description='Time/memory benchmark for CustomLiteAnyStereo')
    p.add_argument('--ckpt_cv', type=str, default=None, help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--ckpt_no_cv', type=str, default=None, help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--ckpt_original', type=str, default='./checkpoints/LiteAnyStereo.pth', help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--shapes', type=int, nargs='+', default=[375, 1242], help='H W pairs, e.g. --shapes 384 768 544 960')
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--out_csv', type=str, default='./benchmark_results.csv')
    p.add_argument('--layer2', action='store_true', help='Use ContextNet with a second layer')
    p.add_argument('--compute_flops', action='store_true', help='Compute FLOPs of the model')
    return p.parse_args()

def build(ckpt_path, device, layer2):
    model = CustomLiteAnyStereo(layer2=layer2).to(device)
    if ckpt_path:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)['model_state'], strict=True)
    
    model.eval()
    
    # before = torch.cuda.memory_allocated()
    # # Prune unused ResNet-34 layers for speed/memory benchmark
    # for name in ['layer2', 'layer3', 'layer4']:
    #     setattr(model.context_net.model, name, nn.Identity())
    # model.context_net.model.avgpool = nn.Identity()
    # model.context_net.model.fc = nn.Identity()
    # torch.cuda.empty_cache()
    
    # after = torch.cuda.memory_allocated()
    # print(f"Memory freed by pruning ResNet-34 layers: {(before - after) / 1e6:.2f} MB")
    return model


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda')

    peak_mem = None

    rows = []
    
    for cv_flag in [True, False]:
        ckpt = args.ckpt_cv if cv_flag else args.ckpt_no_cv
        model = build(ckpt, device, layer2=args.layer2)
        if not cv_flag:
            for name in ['fnet', 'cost_stem_3d', 'cost_agg_2d']:
                setattr(model, name, nn.Identity())
            torch.cuda.empty_cache()
            
        if args.compute_flops:
            benchmark_forward(model, device, shape=(1, 3, *args.shapes), compute_cost_volume=cv_flag, 
                              warmup=args.warmup, iters=args.iters, is_original=False, compute_flops=args.compute_flops)
        else:
            mean_ms, std_ms, peak_mem, parameter_count = benchmark_forward(
                model, device, shape=(1, 3, *args.shapes), compute_cost_volume=cv_flag,
                warmup=args.warmup, iters=args.iters, is_original=False)
            rows.append({'model': 'CV' if cv_flag else 'no-CV', 'mean_ms': mean_ms,
                        'std_ms': std_ms, 'peak_mem_MB': peak_mem, 'parameter_count_M': parameter_count})
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    orig_model = original_LAS().to(device)
    orig_model.load_state_dict(torch.load(args.ckpt_original, map_location=device), strict=True)
    orig_model.eval()
    
    if args.compute_flops:
        benchmark_forward( orig_model, device, shape=(1, 3, *args.shapes), compute_cost_volume=True, 
                          warmup=args.warmup, iters=args.iters, is_original=True, compute_flops=args.compute_flops)
    else:
        mean_ms, std_ms, peak_mem, parameter_count = benchmark_forward(
            orig_model, device, shape=(1, 3, *args.shapes), is_original=True,
            warmup=args.warmup, iters=args.iters)
        rows.append({'model': 'original_LAS', 'mean_ms': mean_ms, 'std_ms': std_ms, 'peak_mem_MB': peak_mem, 'parameter_count_M': parameter_count})
        
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)