import torch
import argparse, csv
import torch
from core.liteanystereo import CustomLiteAnyStereo, original_LAS

@torch.no_grad()
def benchmark_forward(model, device, shape=(1, 3, 384, 768),
                       compute_cost_volume=True, warmup=15, iters=50):
    model.eval()
    torch.backends.cudnn.benchmark = True  # let it pick fast kernels for fixed shape

    left = torch.randn(shape, device=device)
    right = torch.randn(shape, device=device)

    # warm-up: not timed
    for _ in range(warmup):
        if isinstance(model, original_LAS):
            _ = model(left, right, test_mode=True)
        else:
            _ = model(left, right, test_mode=True, compute_cost_volume=compute_cost_volume)
    
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for _ in range(iters):
        starter.record()
        if isinstance(model, original_LAS):
            _ = model(left, right, test_mode=True)
        else:
            _ = model(left, right, test_mode=True, compute_cost_volume=compute_cost_volume)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))  # ms

    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()


def parse_args():
    p = argparse.ArgumentParser(description='Time/memory benchmark for CustomLiteAnyStereo')
    p.add_argument('--ckpt_cv', type=str, default='./continuous_training/checkpoints_continuous/train_continuous_best_cv_Aug11_11-41-06.pth', help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--ckpt_no_cv', type=str, default='./continuous_training/checkpoints_continuous/train_continuous_best_no_cv_Aug11_11-41-06.pth', help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--ckpt_original', type=str, default='./checkpoints/LiteAnyStereo.pth', help='Optional; speed is arch-only but loading real weights avoids drift')
    p.add_argument('--shapes', type=int, nargs='+', default=[384, 768], help='H W pairs, e.g. --shapes 384 768 544 960')
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--out_csv', type=str, default='./benchmark_results.csv')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda')
    model = CustomLiteAnyStereo().to(device)
    
    # if args.ckpt:
    #     model.load_state_dict(torch.load(args.ckpt, map_location=device)['model_state'])
    model.eval()
    
    peak_mem = None

    rows = []
    for cv_flag in [True, False]:
        mean_ms, std_ms = benchmark_forward(model, device, shape=(1,3,*args.shapes),compute_cost_volume=cv_flag, warmup=args.warmup, iters=args.iters)
        rows.append({'compute_cost_volume': cv_flag, 'mean_ms': mean_ms, 'std_ms': std_ms, 'peak_mem_MB': peak_mem})
    
    # Original LAS benchmark (for comparison)
    orig_model = original_LAS().to(device)
    mean_ms, std_ms = benchmark_forward(orig_model, device, shape=(1,3,*args.shapes))
    rows.append({'compute_cost_volume': None, 'mean_ms': mean_ms, 'std_ms': std_ms, 'peak_mem_MB': peak_mem})
    
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)