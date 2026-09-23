#!/usr/bin/env python3
"""
Computational-efficiency benchmark for the LAS + BiDAStabilizer pipeline.

Reports trainable parameters, peak GPU memory and inference latency for four
stages measured independently:

  1. las_single_frame  - one stereo pair, nothing accumulated. This is the
                         number comparable to BiDA Tab. 11's image-based rows
                         (RAFTStereo 5.4G, IGEVStereo 4.6G), which are quoted
                         at a single-frame footprint because those methods
                         process frames independently.
  2. las_clip          - the full T-frame loop including the accumulated
                         disparity stack, i.e. what LAS actually costs inside
                         the pipeline.
  3. stabilizer        - the clip-level forward_batch() call alone, given
                         precomputed disparities. Comparable to the
                         BiDAStabilizer row (0.7M / 13.8G).
  4. end_to_end        - both stages back to back. This is the headline
                         "LAS + stabilizer" figure; 1-3 explain where it
                         comes from.

Timing uses paired CUDA events with explicit synchronisation. Peak memory uses
max_memory_allocated() with the counter reset immediately before each measured
call, after warm-up, so one-off cuDNN workspace allocations are excluded.

Example
-------
python benchmark_las_stabilizer.py \
    --ckpt_las ./checkpoints/LiteAnyStereo.pth \
    --ckpt_stb ./checkpoints/model_LAS_stabilizer_035000.pth \
    --frames 20 --height 720 --width 1280 \
    --repeats 10 --out ./eval_results_video/efficiency.json
"""

import argparse
import json
import logging
import platform
from datetime import datetime

import numpy as np
import torch

from core.liteanystereo import original_LAS
from core.utils.utils import InputPadder
from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #

def _strip_state_dict(state_dict):
    """Handle both the {'model': ...} wrapper written by train_bidastabilizer.py
    and the bare state_dict written at the end of training, plus DDP prefixes."""
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_models(ckpt_las, ckpt_stb, device):
    """Returns (las, stabilizer, param_counts).

    Parameter counts are taken BEFORE requires_grad_(False) is applied, because
    the paper's 'trainable parameters' column describes model capacity, not the
    frozen-for-inference state the modules end up in here.
    """
    las = original_LAS(fnet_pretrained=True)
    stb = BiDAStabilizer()

    params = {
        "las_total": sum(p.numel() for p in las.parameters()),
        "las_trainable": sum(p.numel() for p in las.parameters() if p.requires_grad),
        "stb_total": sum(p.numel() for p in stb.parameters()),
        "stb_trainable": sum(p.numel() for p in stb.parameters() if p.requires_grad),
    }
    # The stabilizer's internal RAFT is frozen during training (see
    # fetch_optimizer in train_bidastabilizer.py), so the genuinely-optimised
    # subset is smaller than stb_total. Report it explicitly.
    params["stb_excluding_raft"] = sum(
        p.numel() for n, p in stb.named_parameters() if "raft" not in n
    )
    params["pipeline_total"] = params["las_total"] + params["stb_total"]

    if ckpt_las is not None:
        logging.info("Loading LAS checkpoint %s", ckpt_las)
        las.load_state_dict(_strip_state_dict(torch.load(ckpt_las, map_location=device)),
                            strict=True)
    if ckpt_stb is not None:
        logging.info("Loading stabilizer checkpoint %s", ckpt_stb)
        stb.load_state_dict(_strip_state_dict(torch.load(ckpt_stb, map_location=device)),
                            strict=True)

    las.to(device).eval()
    stb.to(device).eval()
    for p in las.parameters():
        p.requires_grad_(False)
    for p in stb.parameters():
        p.requires_grad_(False)

    return las, stb, params


# --------------------------------------------------------------------------- #
# pipeline stages -- these mirror run_stereo_model / run_stabilizer_on_video
# in evaluation_og_stabilizer.py exactly. Any divergence here makes the
# measured cost not the cost of the evaluated pipeline.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def las_forward(las, left, right, amp):
    """Per-frame LAS loop -> disparity stack (T, 1, H, W), positive, unpadded."""
    disps = []
    for t in range(left.shape[0]):
        img1, img2 = left[t:t + 1], right[t:t + 1]
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            disp = las(img1, img2, max_disp=192, test_mode=True)
        disps.append(padder.unpad(disp.float()).squeeze(0))
    return torch.stack(disps, dim=0)


@torch.no_grad()
def stabilizer_forward(stb, video, disps, kernel_size, amp):
    """Clip-level stabilisation. Sign convention: LAS emits positive disparity,
    BiDAStabilizer expects negative, output is taken back to positive via abs()."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = stb.forward_batch(video, -disps, kernel_size)
    return out.squeeze(1).abs()


# --------------------------------------------------------------------------- #
# flow isolation
# --------------------------------------------------------------------------- #

class FlowInterceptor:
    """Wraps the frozen SEA-RAFT module to measure or elide its cost.

    Three modes:

      time    call the real module, but bracket each call with paired CUDA
              events. Events are recorded on the stream and only read after a
              single synchronise at end of pass, so per-call timing costs one
              sync per pass rather than one per call -- the pipeline is barely
              perturbed. This is the direct measurement of flow cost.

      record  call the real module and stash outputs, to seed replay.

      replay  return stashed outputs instead of computing. What remains in the
              measured region is the trained trunk (feat_extract, forward and
              backward resblocks, fusion, conv_hr, conv_last) alone.

    Timing directly beats subtracting trunk from total: subtraction takes a
    difference of two large noisy numbers, whereas the events measure flow
    positively. Running both gives a consistency check -- flow + trunk should
    reconstruct the full stabilizer time, and a large residual means the
    interception missed a call path.

    Caveats on replay:
      * Peak memory during replay is NOT the trunk's footprint -- the stash
        holds every flow tensor resident for the whole pass, which the real
        pipeline does not. Take time from that stage, not memory.
      * If the trunk mutates a flow tensor in place, replay feeds it the
        already-mutated version. That changes values, not shapes or control
        flow, so timing stays valid; do not use replay to produce disparity.
    """

    TIME, RECORD, REPLAY = "time", "record", "replay"

    def __init__(self, module, method="forward"):
        self.module = module
        self.method = method
        self.orig = getattr(module, method)
        self.mode = self.TIME
        self.cache = []
        self.idx = 0
        self.events = []

    def __call__(self, *args, **kwargs):
        if self.mode == self.REPLAY:
            # A replay pass asking for more calls than were recorded means the
            # flow call count is data-dependent, which would invalidate both
            # the stash and the subtraction.
            if self.idx >= len(self.cache):
                raise RuntimeError(
                    f"replay wanted call {self.idx + 1} but only "
                    f"{len(self.cache)} were recorded -- flow call count is "
                    "not deterministic")
            out = self.cache[self.idx]
            self.idx += 1
            return out

        if self.mode == self.TIME:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self.orig(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return out

        out = self.orig(*args, **kwargs)
        self.cache.append(out)
        return out

    def install(self):
        setattr(self.module, self.method, self)
        return self

    def restore(self):
        setattr(self.module, self.method, self.orig)

    def clear_events(self):
        self.events = []

    def elapsed_ms(self):
        """Per-call durations for the pass just run. One sync, then read."""
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in self.events]

    def rewind(self):
        self.idx = 0


def resolve_flow_module(stb, path):
    """Walk a dotted attribute path, e.g. 'raft' or 'raft.model'."""
    mod = stb
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def measure_flow(interceptor, fn, warmup, repeats):
    """Direct per-call timing of the intercepted flow module."""
    interceptor.mode = FlowInterceptor.TIME
    for _ in range(warmup):
        out = fn()
        del out
        interceptor.clear_events()
    torch.cuda.synchronize()

    pass_totals, all_calls, calls_per_pass = [], [], []
    for i in range(repeats):
        interceptor.clear_events()
        out = fn()
        per_call = interceptor.elapsed_ms()
        del out
        pass_totals.append(sum(per_call))
        all_calls.extend(per_call)
        calls_per_pass.append(len(per_call))
        logging.info("[flow] rep %d/%d: %d calls, %.2f ms total",
                     i + 1, repeats, len(per_call), pass_totals[-1])

    if not all_calls:
        raise RuntimeError(
            "flow module was never called -- the trunk reaches it by another "
            "path. Try --flow_module raft.model, or grep forward_batch for "
            "the call site.")
    if len(set(calls_per_pass)) != 1:
        logging.warning("flow call count varied across passes: %s", calls_per_pass)

    t, c = np.asarray(pass_totals), np.asarray(all_calls)
    return {
        "time_ms_mean": float(t.mean()),
        "time_ms_std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        "per_call_ms_mean": float(c.mean()),
        "per_call_ms_std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
        "flow_calls": int(calls_per_pass[0]),
        "peak_mem_gb": None,
        "repeats": repeats,
    }


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def measure(fn, warmup, repeats, label):
    """Time and peak-memory a callable. Warm-up passes are discarded so cuDNN
    autotuning and first-call kernel setup don't contaminate the mean."""
    logging.info("[%s] warm-up (%d passes)", label, warmup)
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    times_ms, peak_bytes = [], 0
    for i in range(repeats):
        # Free cached blocks and reset the counter BEFORE the measured region.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()

        times_ms.append(start.elapsed_time(end))
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
        del out
        logging.info("[%s] rep %d/%d: %.2f ms", label, i + 1, repeats, times_ms[-1])

    t = np.asarray(times_ms)
    return {
        "time_ms_mean": float(t.mean()),
        "time_ms_std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        "time_ms_min": float(t.min()),
        "peak_mem_gb": peak_bytes / 1024 ** 3,
        "repeats": repeats,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    assert torch.cuda.is_available(), "CUDA required -- these numbers are meaningless on CPU"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    las, stb, params = load_models(args.ckpt_las, args.ckpt_stb, device)

    T, H, W = args.frames, args.height, args.width
    # Random input: both LAS and the stabilizer have data-independent control
    # flow (fixed layer count, fixed RAFT iterations), so content does not
    # affect latency or footprint. Values are in [0, 255] because LAS
    # normalises internally with 2*(x/255)-1.
    left = (torch.rand(T, 3, H, W, device=device) * 255.0)
    right = (torch.rand(T, 3, H, W, device=device) * 255.0)

    pad_probe = InputPadder(left[0:1].shape, divis_by=32)
    padded_h, padded_w = pad_probe.pad(left[0:1], right[0:1])[0].shape[-2:]

    results = {}

    # 1. single frame -- comparable to Tab. 11's image-based rows
    results["las_single_frame"] = measure(
        lambda: las_forward(las, left[:1], right[:1], args.amp),
        args.warmup, args.repeats, "las_single_frame")

    # 2. full clip loop, including the accumulated disparity stack
    results["las_clip"] = measure(
        lambda: las_forward(las, left, right, args.amp),
        args.warmup, args.repeats, "las_clip")
    results["las_clip"]["time_ms_per_frame"] = results["las_clip"]["time_ms_mean"] / T

    # 3. stabilizer alone. Disparities are precomputed and held outside the
    #    measured region -- they are an input, and their footprint is counted
    #    in the peak, which is correct: the stabilizer cannot run without them.
    disps = las_forward(las, left, right, args.amp)
    torch.cuda.synchronize()
    results["stabilizer"] = measure(
        lambda: stabilizer_forward(stb, left, disps, args.kernel_size, args.amp),
        args.warmup, args.repeats, "stabilizer")
    results["stabilizer"]["time_ms_per_frame"] = results["stabilizer"]["time_ms_mean"] / T

    # 3b. Split the stabilizer into the frozen SEA-RAFT it depends on and the
    #     trained 0.74M trunk. Flow is timed directly with per-call events;
    #     the trunk is timed with flow replayed from a stash. Measuring both
    #     lets them be checked against the full stabilizer time.
    if not args.skip_flow_ablation:
        flow_mod = resolve_flow_module(stb, args.flow_module)
        interceptor = FlowInterceptor(flow_mod, args.flow_method).install()
        stabilize = lambda: stabilizer_forward(stb, left, disps,
                                               args.kernel_size, args.amp)
        try:
            results["flow_only"] = measure_flow(
                interceptor, stabilize, args.warmup, args.repeats)
            n_calls = results["flow_only"]["flow_calls"]
            results["flow_only"]["time_ms_per_frame"] = \
                results["flow_only"]["time_ms_mean"] / T
            logging.info("[flow] %d calls for T=%d (%.1f per frame)",
                         n_calls, T, n_calls / T)

            # Seed the stash, then replay it so only the trunk is timed.
            interceptor.mode = FlowInterceptor.RECORD
            out = stabilize()
            torch.cuda.synchronize()
            del out
            interceptor.mode = FlowInterceptor.REPLAY

            results["stabilizer_no_flow"] = measure(
                lambda: (interceptor.rewind(), stabilize())[1],
                args.warmup, args.repeats, "stabilizer_no_flow")
            results["stabilizer_no_flow"]["time_ms_per_frame"] = \
                results["stabilizer_no_flow"]["time_ms_mean"] / T
            # Memory here includes the resident stash; see FlowInterceptor.
            results["stabilizer_no_flow"]["peak_mem_gb"] = None
        finally:
            interceptor.restore()
            del interceptor
            torch.cuda.empty_cache()

    del disps
    torch.cuda.empty_cache()

    # 4. end to end
    def _e2e():
        d = las_forward(las, left, right, args.amp)
        return stabilizer_forward(stb, left, d, args.kernel_size, args.amp)

    results["end_to_end"] = measure(_e2e, args.warmup, args.repeats, "end_to_end")
    results["end_to_end"]["time_ms_per_frame"] = results["end_to_end"]["time_ms_mean"] / T

    report = {
        "params": params,
        "stages": results,
        "config": {
            "frames": T,
            "resolution": [H, W],
            "padded_resolution": [int(padded_h), int(padded_w)],
            "kernel_size": args.kernel_size,
            "precision": "fp16 autocast" if args.amp else "fp32",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }

    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


def print_report(report):
    p, s, c = report["params"], report["stages"], report["config"]

    print("\n" + "=" * 78)
    print(f"LAS + BiDAStabilizer efficiency  |  {c['frames']} frames x "
          f"{c['resolution'][0]}x{c['resolution'][1]}  |  {c['precision']}")
    print(f"LAS runs padded at {c['padded_resolution'][0]}x{c['padded_resolution'][1]} "
          f"(InputPadder divis_by=32); stabilizer runs at native resolution")
    print(f"{c['gpu']}  |  torch {c['torch']}  |  kernel_size={c['kernel_size']}")
    print("=" * 78)

    print("\nParameters (M)")
    print(f"  {'LAS (original_LAS)':<34}{p['las_total'] / 1e6:>10.3f}")
    print(f"  {'BiDAStabilizer':<34}{p['stb_total'] / 1e6:>10.3f}")
    print(f"  {'  of which optimised (non-RAFT)':<34}{p['stb_excluding_raft'] / 1e6:>10.3f}")
    print(f"  {'Pipeline total':<34}{p['pipeline_total'] / 1e6:>10.3f}")

    print(f"\n{'Stage':<20}{'Time (ms)':>20}{'ms/frame':>12}{'Peak mem (G)':>15}")
    print("-" * 78)
    for key, label in [("las_single_frame", "LAS (1 frame)"),
                       ("las_clip", f"LAS ({c['frames']} frames)"),
                       ("stabilizer", "Stabilizer"),
                       ("stabilizer_no_flow", "  trunk only"),
                       ("flow_only", "  SEA-RAFT only"),
                       ("end_to_end", "End to end")]:
        if key not in s:
            continue
        r = s[key]
        per_frame = f"{r['time_ms_per_frame']:.2f}" if "time_ms_per_frame" in r else "-"
        mem = f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] is not None else "n/a"
        print(f"{label:<20}{r['time_ms_mean']:>13.2f} ± {r['time_ms_std']:<5.2f}"
              f"{per_frame:>12}{mem:>15}")
    print("-" * 78)

    if "flow_only" in s:
        total = s["stabilizer"]["time_ms_mean"]
        flow = s["flow_only"]["time_ms_mean"]
        n = s["flow_only"]["flow_calls"]
        print(f"SEA-RAFT (measured directly): {flow:.0f} ms = "
              f"{100.0 * flow / total:.0f}% of the stabilizer's {total:.0f} ms, "
              f"over {n} calls ({n / c['frames']:.1f} per frame) at "
              f"{s['flow_only']['per_call_ms_mean']:.1f} ± "
              f"{s['flow_only']['per_call_ms_std']:.1f} ms each.")

        if "stabilizer_no_flow" in s:
            trunk = s["stabilizer_no_flow"]["time_ms_mean"]
            residual = total - flow - trunk
            print(f"Trained 0.74M trunk: {trunk:.0f} ms "
                  f"({100.0 * trunk / total:.0f}%).")
            print(f"Reconciliation: flow + trunk = {flow + trunk:.0f} ms vs "
                  f"{total:.0f} ms measured, residual {residual:+.0f} ms "
                  f"({100.0 * abs(residual) / total:.1f}%).")
            if abs(residual) > 0.10 * total:
                print("  WARNING: residual >10%. Either the interception is "
                      "missing a flow call path, or the trunk does substantial "
                      "work that neither stage attributes. Check before "
                      "quoting the split.")
        print("  ('trunk only' memory is n/a -- the replay stash holds every "
              "flow tensor resident, which the real pipeline does not.)")
        print("-" * 78)

    overhead_ms = s["end_to_end"]["time_ms_mean"] - s["las_clip"]["time_ms_mean"]
    overhead_pct = 100.0 * overhead_ms / s["las_clip"]["time_ms_mean"]
    print(f"Stabilizer adds {overhead_ms:.1f} ms ({overhead_pct:.0f}%) on top of the "
          f"LAS backbone")
    print(f"Peak memory is set by the {'stabilizer' if s['stabilizer']['peak_mem_gb'] > s['las_clip']['peak_mem_gb'] else 'LAS loop'} "
          f"stage, as expected: LAS frees per-frame activations each iteration "
          f"while the stabilizer holds all {c['frames']} frames at once.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_las", default="./checkpoints/LiteAnyStereo.pth")
    ap.add_argument("--ckpt_stb", default=None,
                    help="stabilizer checkpoint; weights do not affect timing "
                         "or memory, but load them anyway so the reported "
                         "configuration matches the evaluated one")
    ap.add_argument("--frames", type=int, default=20,
                    help="clip length T (BiDA Tab. 11 uses 20)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--kernel_size", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast. Off by default -- match whatever "
                         "evaluate() used, and state it in the thesis table")
    ap.add_argument("--flow_module", default="raft",
                    help="dotted path to the flow module on the stabilizer, "
                         "e.g. 'raft' or 'raft.model'")
    ap.add_argument("--flow_method", default="forward_fullres",
                    help="method on --flow_module that the trunk calls")
    ap.add_argument("--skip_flow_ablation", action="store_true",
                    help="omit the trunk-only stage")
    ap.add_argument("--out", default=None, help="path for JSON output")
    return ap.parse_args()


if __name__ == "__main__":
    main()