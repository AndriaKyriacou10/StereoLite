# evaluate_sceneflow.py
import argparse
import json
import os

import numpy as np
import torch

import bidastabilizer_integration.video_datasets as datasets
from bidastabilizer_integration.evaluation.core.evaluator import Evaluator
from bidastabilizer_integration.evaluation.utils.utils import aggregate_and_print_results
from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer
from core.liteanystereo import original_LAS

def run_eval(args):
    os.makedirs(args.exp_dir, exist_ok=True)

    with open(os.path.join(args.exp_dir, "expconfig.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    evaluator = Evaluator()
    
    # evaluator.setup_visualization(args)  # just needs .exp_dir / .visualize_interval / .render_bin_size

    model = original_LAS()
    model.cuda(0)

    model_stabilizer = None
    if args.stabilizer_ckpt is not None:
        model_stabilizer = BiDAStabilizer()
        model_stabilizer.cuda()
        state_dict = torch.load(args.stabilizer_ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_stabilizer.load_state_dict(state_dict, strict=True)
        print("Done loading stabilizer checkpoint:", args.stabilizer_ckpt)

    test_dataloader = datasets.SequenceSceneFlowDataset(
        {},
        dstype='frames_cleanpass',
        sample_len=args.sample_len,
        add_monkaa=False,
        add_driving=False,
        things_test=True,
    )

    evaluate_result = evaluator.evaluate_sequence(
        model,
        model_stabilizer,
        test_dataloader,
        is_real_data=True,
        exp_dir=args.exp_dir,
    )

    aggregate_result = aggregate_and_print_results(evaluate_result)

    result_file = os.path.join(args.exp_dir, "result_eval.json")
    print(f"Dumping eval results to {result_file}.")
    with open(result_file, "w") as f:
        json.dump(aggregate_result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="./eval_results_video_original")
    parser.add_argument("--model_weights", default = './checkpoints/LiteAnyStereo.pth' , required=True, help="checkpoint for the stereo model")
    parser.add_argument("--kernel_size", type=int, default=50)
    parser.add_argument("--stabilizer_ckpt", default='./checkpoints/disp_corr_run_b8/LAS_stabilizer_final.pth')
    parser.add_argument("--sample_len", type=int, default=9)
    parser.add_argument("--dstype", default='frames_cleanpass')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--visualize_interval", type=int, default=0,
                         help="0 disables visualization (and its pytorch3d point-cloud path in evaluator.py)")
    parser.add_argument("--render_bin_size", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    run_eval(args)