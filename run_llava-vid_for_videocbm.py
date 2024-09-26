# https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-Qwen2

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import argparse
import os
import pandas as pd

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
        
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=2)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def main(args):
    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    conv_template = args.conv_mode  # Make sure you use correct chat template for different models # llava/conversation.py 에서 확인(그냥 llava에서는 conv_llava_v1 사용함)
    max_frames_num = args.num_frames
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    model = model.to(torch.bfloat16)

    if args.dataset == 'k100':
        train_csv_path = '/data/psh68380/repos/Video-CBM_/data/video_annotation/kinetics100/train.csv'
        class_csv_path = '/data/psh68380/repos/Video-CBM_/data/kinetics100_classes.csv'
    elif args.dataset == 'k400':
        train_csv_path = '/data/psh68380/repos/Video-CBM_/data/video_annotation/kinetics400/train.csv'
        class_csv_path = '/data/psh68380/repos/Video-CBM_/data/kinetics400_classes.csv'

    df_train = pd.read_csv(train_csv_path, header=None, names=['video_path', 'label'])

    df_classes = pd.read_csv(class_csv_path)
    df_merged = pd.merge(df_train, df_classes, left_on='label', right_on='id')
    # 각 label당 n = 10개의 비디오 샘플 선택
    df_sampled = df_merged.groupby('label').apply(lambda x: x.sample(n=10, random_state=42, replace=True) if len(x) >= 10 else x).reset_index(drop=True)
    df_final = df_sampled[['video_path', 'name']]
    video_data = df_final.sort_values(by='name').reset_index(drop=True)
    
    for idx, row in video_data.iterrows():
        video_path = row['video_path']
        label = row['name']
        answer_file = os.path.join(args.answer_folder, f'{args.dataset}_ost_{args.descriptor_type}_concepts_{label}.txt')

        video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        ################
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        
        question = (DEFAULT_IMAGE_TOKEN + 
                    f"{time_instruciton}\n" +
                    args.query)
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        print(outputs)

        if idx % 10 == 0 and idx > 0:
            print(f'Finished answering {label}: {idx} / {len(video_data)/10}')

        with open(answer_file, 'a') as f:
            f.write(f"{outputs}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--dataset", type=str, default='k100')
    # parser.add_argument("--video_folder", type=str)
    parser.add_argument("--answer_folder", type=str, default="cbm_concepts/k100_temporal")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--descriptor_type", type=str, default="temporal")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--query", type=str, default="", help="question for asked to llava-video")

    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3) 
    parser.add_argument('--input_size', default=224, type=int,help='videos input size')
    parser.add_argument('--short_side_size', type=int, default=224)

    args = parser.parse_args()

    main(args)