# CKPT_NO_CV=./gru_context_features_injection/phase2_best_model.pth
# CKPT_CV=./gru_context_features_injection/phase1_best_model.pth

CKPT_NO_CV=./checkpoints/phase2_best_model_RUN2.pth
CKPT_CV=./checkpoints/phase1_best_model_RUN2.pth

[ -f "$CKPT_CV" ] || { echo "Checkpoint not found: $CKPT_CV"; exit 1; }
echo "Phase 1: With CV"
python -m evaluation_STEREO.evaluate_stereoLite --dataset middlebury_H --ckpt "$CKPT_CV" --cost_volume && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt "$CKPT_CV" --cost_volume && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python -m evaluation_STEREO.evaluate_stereoLite --dataset sceneflow --ckpt "$CKPT_CV"  --cost_volume && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"

echo "Phase 2: NO CV"
python -m evaluation_STEREO.evaluate_stereoLite --dataset middlebury_H --ckpt "$CKPT_NO_CV" && echo "[OK] middlebury_H" || echo "[FAIL] middlebury_H" 
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt "$CKPT_NO_CV" && echo "[OK] eth3d" || echo "[FAIL] eth3d"
python -m evaluation_STEREO.evaluate_stereoLite --dataset sceneflow --ckpt "$CKPT_NO_CV" && echo "[OK] sceneflow" || echo "[FAIL] sceneflow"


echo "--- Evaluation Completed ---"