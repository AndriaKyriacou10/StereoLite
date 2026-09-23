
# Depending on the stereo model used, change the --name and --ckpt_model arguments accordingly. 
# The --ckpt_stb argument should point to the checkpoint of the stabilizer model you want to evaluate.

# For LiteAnyStereo: --name LAS_stabilizer
# For RAFT-Stereo: --name raftstereo_stabilizer

echo "==== FallingThings3D Iter 25k Evaluation ===="
python evaluation_og_stabilizer.py --name LAS_stabilizer --ckpt_model ./checkpoints/LiteAnyStereo.pth --ckpt_stb ./checkpoints/iter_25k/LAS_stabilizer_final.pth --dataset things --iter_name iter_50k


echo "==== Sintel Clean Iter 25k Evaluation ===="
python evaluation_og_stabilizer.py --name LAS_stabilizer --ckpt_model ./checkpoints/LiteAnyStereo.pth --ckpt_stb ./checkpoints/iter_25k/LAS_stabilizer_final.pth --dataset sintel_clean --iter_name iter_50k --kernel_size 10

echo "Video Evaluation Completed"