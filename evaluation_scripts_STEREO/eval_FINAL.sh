# Change the checkpoint paths below to point to the best checkpoints of your training runs for both the "with CV" and "no CV" models.

CKPT_NO_CV=./continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_best_no_cv_ContextNet.pth
CKPT_CV=./continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_best_cv_ContextNet.pth
[ -f "$CKPT_CV" ] || { echo "Checkpoint not found: $CKPT_CV"; exit 1; }

python evaluate_other_datasets.py --dataset middlebury_H --ckpt "$CKPT_CV"  --layer2 --cost_volume && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python evaluate_other_datasets.py --dataset eth3d --ckpt "$CKPT_CV" --layer2  --cost_volume && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python evaluate_other_datasets.py --dataset sceneflow --ckpt "$CKPT_CV"  --layer2  --cost_volume && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"


echo "--- Evaluation Completed ---"