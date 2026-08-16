import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import sys
import torch
import torch.optim as optim
import time
from collections import defaultdict
from bidastabilizer_integration.train_utils.losses import sequence_loss, consistency_loss
from .core.liteanystereo import original_LAS
import json
@torch.no_grad()
def diagnose_batch(batch, model, model_stabilizer, Flow_Model, args, total_steps=None, skipped_frame_log=None):
    disparities_list = []
    b, T, *_ = batch["img"][:, :, 0].shape
    for i in range(T):        
        # ==== LAS MODEL INFERENCE ====
        if args.name == "raftstereo_stabilizer":
            _, flow_predictions = model(batch["img"][:, :, 0][:, i], batch["img"][:, :, 1][:, i],
                                        iters=args.train_iters, test_mode=True)
        elif args.name == "LAS_stabilizer":
            flow_predictions,_ = model(batch["img"][:, :, 0][:, i], batch["img"][:, :, 1][:, i], max_disp=192, test_mode=False)
        disparities_list.append(flow_predictions)
        
    disparities = torch.stack(disparities_list, dim=0) # T B C H W

    if args.name == "LAS_stabilizer":
        disparities = -disparities # # Disparity from LAS model is positive, but BiDAStabilizer expects nagative disparity value and also GT is negative

    print(f"{args.name.split("_")[0]} output: min={disparities.min().item()}, max={disparities.max().item()}")
    print(f"GT disp:     min={batch['disp'].min().item()}, max={batch['disp'].max().item()}")

    num_traj = len(batch["disp"][0]) # number of frames
    # logging.info(f"Number of frames in the video: {num_traj}")
    # Input: B T C H W    Output: T B C H W
    
    # Disparity from LAS model is positive, but BiDAStabilizer expects nagative disparity value and also GT is negative
    disparities_stb = model_stabilizer(batch["img"][:, :, 0], disparities.permute(1,0,2,3,4)) # B T C H W 
    out = defaultdict(float)
    for i in range(num_traj):
        #eq.14 from paper 
        valid_i = batch['valid_disp'][:, i, 0]
        skipped_frames = []
        if valid_i.sum() == 0:
            continue
        gt_i = batch["disp"][:, i, 0]
        
        print(gt_i.shape, valid_i.shape, disparities[i].shape, disparities_stb[i].shape)
        
        assert gt_i.shape == disparities[i].shape, f"GT shape {gt_i.shape} and disparity shape {disparities[i].shape} do not match"
        
        loss_base, metrics_base = sequence_loss(disparities[None][:, i], gt_i, valid_i)
        loss_stb, metrics_stb = sequence_loss(disparities_stb[None][:, i], gt_i, valid_i)
        
        out["supervised_loss_base"] += loss_base / num_traj
        out["supervised_loss_stb"] += loss_stb / num_traj
        out["epe_base"] += metrics_base['epe'] / num_traj
        out["epe_stb"] += metrics_stb['epe'] / num_traj
        
        # --- fve: did the correction match the correction that was needed? ---
        m = valid_i.bool()
        base_i, stb_i = disparities[i], disparities_stb[i]
        residual = (stb_i - base_i)[m]                          # what it did
        target   = (gt_i  - base_i)[m]                          # what it should have done
        out["fve_num"] += ((target - residual) ** 2).sum().item()
        out["fve_den"] += (target ** 2).sum().item()

    out["tmp_stb"]  = (0.2 * consistency_loss(batch["img"][:, :, 0],
                        disparities_stb.permute(1,0,2,3,4), Flow_Model, alpha=50)).item()
    out["tmp_base"] = (0.2 * consistency_loss(batch["img"][:, :, 0],
                        disparities.permute(1,0,2,3,4), Flow_Model, alpha=50)).item()

    return dict(out)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=['LAS_stabilizer', 'raftstereo_stabilizer'])
    parser.add_argument("--ckpt_stereo", help="Checkpoint of stereo model")
    parser.add_argument('--ckpt_stb', help="Stabilizer Checkpoint")

    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size used during training."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=["things", "monkaa", "driving"],
        help="training datasets.",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="max learning rate.")

    parser.add_argument(
        "--num_steps", type=int, default=100000, help="length of training schedule."
    )
    parser.add_argument(
        "--save_steps", type=int, default=2500, help="length of training schedule."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[320, 720],
        help="size of the random image crops used during training.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=22,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )

    parser.add_argument(
        "--sample_len", type=int, default=1, help="length of training video samples"
    )
    parser.add_argument(
        "--validate_at_start", action="store_true", help="validate the model at start"
    )
    parser.add_argument("--save_freq", type=int, default=100, help="save frequency")

    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate every n epoch",
    )

    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers."
    )
    parser.add_argument('--max_batches', type=int, default=200, help='Maximum number of batches to process')
    args = parser.parse_args()
    return args

if __name__ == "__main__":    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()
    
    if args.name == "LAS_stabilizer":
        model = original_LAS(fnet_pretrained=False)
        import bidastabilizer_integration.video_datasets as datasets
        logging.info(f"Stereo model: LAS | video datasets 1")
    elif  args.name == "raftstereo_stabilizer":
        from bidastabilizer_integration.models.raft_stereo_model import RAFTStereoModel
        model = RAFTStereoModel().model
        import bidastabilizer_integration.video_datasets2 as datasets
        logging.info(f"Stereo model: RAFT-Stereo | video datasets 2")
    
    from bidastabilizer_integration.models.raft_model import RAFTModel
    raft = RAFTModel() # predict the optical flow 
    
    from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer
    model_stabilizer = BiDAStabilizer()
    
    model.to(device)
    model_stabilizer.to(device)
    model.eval()
    model_stabilizer.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model_stabilizer.parameters():
        p.requires_grad = False
    
    raft.to(device)
    
    train_loader = datasets.fetch_dataloader(args)
    
    if args.ckpt_stereo is not None:
        # Restore LAS / RAFT-Stereo checkpoint
        assert args.ckpt_stereo.endswith(".pth") or args.ckpt_stereo.endswith(
            ".pt"
        )
        logging.info("Loading checkpoint...")
        print("Loading checkpoint", args.ckpt_stereo)

        strict = True

        state_dict = torch.load(args.ckpt_stereo, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Done loading stero model checkpoint")
    
    if args.ckpt_stb is not None:
        assert args.ckpt_stb.endswith(".pth") or args.ckpt_stb.endswith(
            ".pt"
        )
        logging.info("Loading stabilizer checkpoint...")
        print("load Stabilizer parameters", args.ckpt_stb)
        strict = True

        state_dict = torch.load(args.ckpt_stb, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model_stabilizer.load_state_dict(state_dict, strict=strict)
        logging.info(f"Done loading stabilizer checkpoint")
    
    totals = defaultdict(float); n = 0
    n_skipped = 0
    rows = []
    for i_batch, batch in enumerate(tqdm(train_loader)):
        if batch is None:
            n_skipped += 1
            break
        elif i_batch >= args.max_batches:
            break
        for k, v in batch.items():
            batch[k] = v.cuda()
        r = diagnose_batch(batch, model, model_stabilizer, raft, args)
        rows.append({"i_batch": i_batch, **r})
        for k, v in r.items():
            totals[k] += v
        n += 1

    fve = 1.0 - totals["fve_num"] / max(totals["fve_den"], 1e-8)
    means = {k: v / n for k, v in totals.items() if not k.startswith("fve_")}
    
    file_path = f"is_stb_training_{args.name}.json"
    
    payload = {
    "args": vars(args),
    "n_batches": n,
    "n_skipped": n_skipped,
    "means": means,
    "fve": fve,
    "fve_num": totals["fve_num"],
    "fve_den": totals["fve_den"],
    "deltas": {
        "supervised": means["supervised_loss_stb"] - means["supervised_loss_base"],
        "epe": means["epe_stb"] - means["epe_base"],
        "tmp": means["tmp_stb"] - means["tmp_base"],
        },
    "per_batch": rows
    }
    
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    logging.info(f"Wrote {file_path}")
    logging.info(json.dumps(payload["deltas"], indent=2))