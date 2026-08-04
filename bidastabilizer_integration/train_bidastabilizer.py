# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import sys
import torch
import torch.optim as optim
import time

# from munch import DefaultMunch
# from pytorch_lightning.lite import LightningLite
import json
from torch.cuda.amp import GradScaler
from types import SimpleNamespace

from bidastabilizer_integration.train_utils.utils import (
    run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from bidastabilizer_integration.train_utils.logger import Logger
import importlib
from collections import defaultdict
# from bidavideo.evaluation.core.evaluator import Evaluator
from bidastabilizer_integration.train_utils.losses import sequence_loss, consistency_loss
import bidastabilizer_integration.video_datasets as datasets
autocast = torch.cuda.amp.autocast

from core.liteanystereo import original_LAS

def fetch_optimizer(args, model, model_stabilizer, Flow_model):
    """Create the optimizer and learning rate scheduler"""
    for name, param in Flow_model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model_stabilizer.named_parameters():
        if any([key in name for key in ['raft']]):
            param.requires_grad_(False)

    optimizer = optim.AdamW(
        model_stabilizer.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


def forward_batch(batch, model, model_stabilizer, Flow_Model, args, total_steps=None, skipped_frame_log=None):
    output = {}
    disparities_list = []
    b, T, *_ = batch["img"][:, :, 0].shape
    for i in range(T):        
        # ==== LAS MODEL INFERENCE ====
        flow_predictions,_ = model(batch["img"][:, :, 0][:, i], batch["img"][:, :, 1][:, i], max_disp=192, test_mode=False)
        disparities_list.append(flow_predictions)
    disparities = torch.stack(disparities_list, dim=0) # T B C H W


    print(f"LAS1 output: min={disparities.min().item()}, max={disparities.max().item()}")
    print(f"GT disp:     min={batch['disp'].min().item()}, max={batch['disp'].max().item()}")

    num_traj = len(batch["disp"][0]) # number of frames
    # Input: B T C H W    Output: T B C H W
    
    # Disparity from LAS model is positive, but BiDAStabilizer expects nagative disparity value and also GT is negative
    disparities_stb = model_stabilizer(batch["img"][:, :, 0], -disparities.permute(1,0,2,3,4)) # B T C H W 

    for i in range(num_traj):
        #eq.14 from paper 
        valid_i = batch['valid_disp'][:, i, 0]
        skipped_frames = []
        if valid_i.sum() == 0:
            skipped_frames.append(i)
            if skipped_frame_log is not None:
                skipped_frame_log.append({'step': total_steps, 'frame_index': i})
            # Prevent NaN values
            continue
        seq_loss, metrics = sequence_loss(
            disparities_stb[None][:, i], batch["disp"][:, i, 0], batch["valid_disp"][:, i, 0]
        )
        output[f"disp_stb_{i}"] = {"loss": seq_loss / num_traj, "metrics": metrics} # loss: average over T (num_traj) frames

    temporal_loss = 0.2 * consistency_loss(batch["img"][:, :, 0], disparities_stb.permute(1,0,2,3,4), Flow_Model, alpha=50)

    output[f"disp_temporal"] = {"loss": temporal_loss, "metrics": {"tc": temporal_loss.mean().item()}}
    output["disparity"] = {
        "predictions": torch.cat(
            [disparities[i] for i in range(num_traj)], dim=1).detach(),
    }
    output["skipped_frames"] = skipped_frames
    return output

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Add LAS model 
    model = original_LAS(fnet_pretrained=False)
    
    from bidastabilizer_integration.models.raft_model import RAFTModel
    raft = RAFTModel()

    from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer
    model_stabilizer = BiDAStabilizer()

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    raft.to(device)
    model_stabilizer.to(device)
    
    with open(args.ckpt_path + "/meta.json", "w") as file:
        json.dump(vars(args), file, sort_keys=True, indent=4)
        
    train_loader = datasets.fetch_dataloader(args)
    
    # if args.skip_frames_check:
    #     from collections import defaultdict
    #     skip_count = 0
    #     total_checks = 0
    #     per_frame_skips = defaultdict(int)

    #     for step, batch in enumerate(tqdm(train_loader)):
    #         if step >= 1000:
    #             break
    #         num_traj = batch["disp"].shape[1]
    #         for i in range(num_traj):
    #             valid_i = batch["valid_disp"][:, i, 0]
    #             total_checks += 1
    #             if valid_i.sum() == 0:
    #                 skip_count += 1
    #                 per_frame_skips[i] += 1

    #     print(f"\n{skip_count} / {total_checks} frame-checks were empty "
    #         f"({100 * skip_count / total_checks:.2f}%)")
    #     print(f"Per-frame-position breakdown: {dict(sorted(per_frame_skips.items()))}")

    #     import sys
    #     sys.exit(0)

    logging.info(f"Train loader size:  {len(train_loader)}")

    optimizer, scheduler = fetch_optimizer(args, model, model_stabilizer, raft)

    print("Parameter Count:", {count_parameters(model_stabilizer)})
    logging.info(f"Parameter Count:  {count_parameters(model_stabilizer)}")
    total_steps = 0
    logger = Logger(model_stabilizer, scheduler, args.ckpt_path)
    
    if args.restore_ckpt is not None:
        # Restore LAS checkpoint
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
            ".pt"
        )
        logging.info("Loading checkpoint...")
        print("Loading checkpoint", args.restore_ckpt)

        strict = True

        state_dict = torch.load(args.restore_ckpt, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Done loading checkpoint")
        
    if args.restore_stabilizer_ckpt is not None:
        assert args.restore_stabilizer_ckpt.endswith(".pth") or args.restore_stabilizer_ckpt.endswith(
            ".pt"
        )
        logging.info("Loading stabilizer checkpoint...")
        print("load Stabilizer parameters", args.restore_stabilizer_ckpt)
        strict = True

        state_dict = torch.load(args.restore_stabilizer_ckpt, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        model_stabilizer.load_state_dict(state_dict, strict=strict)
        logging.info(f"Done loading stabilizer checkpoint")

    model_stabilizer.train()

    
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)


    should_keep_training = True
    global_batch_num = 0
    epoch = -1
    
    start_time = time.time()
    skipped_frame_log = []
    while should_keep_training:
        epoch += 1
        for i_batch, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            if batch is None:
                print("batch is None")
                continue
            for k, v in batch.items():
                batch[k] = v.cuda()

            assert model_stabilizer.training
            output = forward_batch(batch, model, model_stabilizer, raft, args, total_steps, skipped_frame_log)

            loss = 0
            logger.update()
            for k, v in output.items():
                if "loss" in v:
                    loss += v["loss"]
                    logger.writer.add_scalar(
                        f"live_{k}_loss", v["loss"].item(), total_steps
                    )
                if "metrics" in v:
                    logger.push(v["metrics"], k)

            
            if len(output) > 1:
                logger.writer.add_scalar(
                    f"live_total_loss", loss.item(), total_steps
                )
            logger.writer.add_scalar(
                f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
            )
            global_batch_num += 1
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_stabilizer.parameters(), 1.0)

            scaler.step(optimizer)
            if total_steps < args.num_steps:
                scheduler.step()
            scaler.update()
            total_steps += 1

            if total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
                print(f"Skipped frames so far: {len(skipped_frame_log)} / {total_steps} steps")
                with open(f"{args.ckpt_path}/skipped_frames.json", "w") as f:
                    json.dump(skipped_frame_log, f, indent=2)
    
            if (i_batch >= len(train_loader) - 1) or (total_steps == 1 and args.validate_at_start):
                ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                save_path = Path(
                    f"{args.ckpt_path}/model_{args.name}_{ckpt_iter}.pth"
                )

                save_dict = {
                    "model": model_stabilizer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "total_steps": total_steps,
                }

                logging.info(f"Saving file {save_path}")
                torch.save(save_dict, save_path)
                
                with open(f"{args.ckpt_path}/skipped_frames.json", "w") as f:
                    json.dump(skipped_frame_log, f, indent=2)

            if total_steps > args.num_steps:
                should_keep_training = False
                break
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration:.2f}s")   
    
    logger.close()
    PATH = f"{args.ckpt_path}/{args.name}_final.pth"
    torch.save(model_stabilizer.state_dict(), PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="[raftstereo_stabilizer, igevstereo_stabilizer, LAS_stabilizer]")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--restore_stabilizer_ckpt", help="restore stabilizer checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    # Training parameters
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
    # Validation parameters
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of updates to the disparity field in each forward pass during validation.",
    )
    # Data augmentation
    parser.add_argument(
        "--img_gamma", type=float, nargs="+", default=None, help="gamma range"
    )
    parser.add_argument(
        "--saturation_range",
        type=float,
        nargs="+",
        default=None,
        help="color saturation",
    )
    parser.add_argument(
        "--do_flip",
        default=False,
        choices=["h", "v"],
        help="flip the images horizontally or vertically",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        nargs="+",
        default=[0, 0],
        help="re-scale the images randomly",
    )
    parser.add_argument(
        "--noyjitter",
        action="store_true",
        help="don't simulate imperfect rectification",
    )
    parser.add_argument(
        "--skip_frames_check", 
        action="store_true",
        help="Get empty valid-mask frames without re-training, then exit"
    )
    
    args = parser.parse_args()

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=args.ckpt_path + '/' + args.name + '.log',
        filemode='a',
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    train(args)