CKPT_NO_CV=./checkpoints/train_continuous_best_no_cv.pth
CKPT_CV=./checkpoints/train_continuous_best_cv.pth

echo "With CV"
[ -f "$CKPT_CV" ] || { echo "Checkpoint not found: $CKPT_CV"; exit 1; }
python -m evaluation_STEREO.evaluate_stereoLite --dataset middlebury_H --ckpt "$CKPT_CV" --cost_volume && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt "$CKPT_CV" --cost_volume && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python -m evaluation_STEREO.evaluate_stereoLite --dataset sceneflow --ckpt "$CKPT_CV"  --cost_volume && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"

echo "NO CV"
[ -f "$CKPT_CV" ] || { echo "Checkpoint not found: $CKPT_NO_CV"; exit 1; }
python -m evaluation_STEREO.evaluate_stereoLite --dataset middlebury_H --ckpt "$CKPT_NO_CV" && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt "$CKPT_NO_CV" && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python -m evaluation_STEREO.evaluate_stereoLite --dataset sceneflow --ckpt "$CKPT_NO_CV" && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"
echo "--- Evaluation Completed ---"