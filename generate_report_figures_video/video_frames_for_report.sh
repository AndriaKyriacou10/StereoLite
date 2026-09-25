NAME=LAS_stabilizer
CKPT_MODEL=./checkpoints/LiteAnyStereo.pth
CKPT_STB=./checkpoints/LAS_stabilizer_final.pth

# echo "ambush_2"
# python video_frames_paper.py --name $NAME --ckpt_model $CKPT_MODEL --ckpt_stb $CKPT_STB \
#  --dataset sintel_clean --scene_name ambush_2 --start_frame 1 --end_frame 50

# echo "cave_2"
# python video_frames_paper.py --name $NAME --ckpt_model $CKPT_MODEL --ckpt_stb $CKPT_STB \
#  --dataset sintel_clean --scene_name cave_2 --start_frame 1 --end_frame 50

echo "bamboo_2"
python -m generate_report_figures_video.video_frames_paper --name $NAME --ckpt_model $CKPT_MODEL --ckpt_stb $CKPT_STB \
 --dataset sintel_clean --scene_name bamboo_2 --start_frame 1 --end_frame 50