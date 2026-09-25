# Change the checkpoint paths below to point to the best checkpoints of your training runs for both the "with CV" and "no CV" models.

CKPT_NO_CV=./checkpoints/stereoLite_FINAL_no_cv.pth
CKPT_CV=./checkpoints/stereoLite_FINAL_cv.pth
[ -f "$CKPT_CV" ] || { echo "Checkpoint not found: $CKPT_CV"; exit 1; }

python -m evaluation_STEREO.evaluate_stereoLite --dataset middlebury_H --ckpt "$CKPT_CV"  --layer2 --cost_volume && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt "$CKPT_CV" --layer2  --cost_volume && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python -m evaluation_STEREO.evaluate_stereoLite --dataset sceneflow --ckpt "$CKPT_CV"  --layer2  --cost_volume && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"


echo "--- Evaluation Completed ---"