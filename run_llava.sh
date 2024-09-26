#!/bin/bash

#SBATCH --job-name llava_video
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=50G
#SBATCH --time 6-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH -o /data/psh68380/repos/LLaVA-NeXT/%A-%x.out
#SBATCH -e /data/psh68380/repos/LLaVA-NeXT/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12345
# workspaceFolder="/data/psh68380/repos/LLaVA-NeXT"
# export PYTHONPATH="${workspaceFolder}:${workspaceFolder}/llava/eval"

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
# --data_root "/local_datasets/ai_hub_sketch_mw/01/train"
python -u /data/psh68380/repos/LLaVA-NeXT/run_llava-vid_for_videocbm.py \
--model_path "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2" \
--conv_mode "llava_v1" \
--answer_folder "cbm_concepts/k100_temporal5" \
--descriptor_type "temporal" \
--temperature 0 \
--max_new_tokens 512 \
--query "Please describe the main action in this video in a single sentence."
# --query "Please describe the main action in this video in a single sentence, starting the sentence with a gerund phrase, not a subject-verb structure."
# --query "Please describe the main action in this video in a single sentence using a gerund phrase."
# --query "Please provide a description of this video using a gerund phrase."
# --query "Please provide a brief summary of the content of this v/ideo in one sentence."
    
echo "Job finish"
exit 0