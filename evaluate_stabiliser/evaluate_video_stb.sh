# Depending on the stereo model used, change the --name and --ckpt_model arguments accordingly. 
# The --ckpt_stb argument should point to the checkpoint of the stabilizer model you want to evaluate.

# Depending on the GPU memory available, you may need to adjust the --kernel_size argument.

# For LiteAnyStereo: --name LAS_stabilizer
# For RAFT-Stereo: --name raftstereo_stabilizer

echo "==== FallingThings3D Iter 25k Evaluation ===="
python -m evaluate_stabiliser.evaluation_og_stabilizer --name LAS_stabilizer --ckpt_model ./checkpoints/LiteAnyStereo.pth --ckpt_stb ./checkpoints/LAS_stabilizer_final.pth --dataset things --iter_name iter_25k


echo "==== Sintel Clean Iter 25k Evaluation ===="
python -m evaluate_stabiliser.evaluation_og_stabilizer --name LAS_stabilizer --ckpt_model ./checkpoints/LiteAnyStereo.pth --ckpt_stb ./checkpoints/LAS_stabilizer_final.pth --dataset sintel_clean --iter_name iter_25k --kernel_size 10

echo "Video Evaluation Completed"