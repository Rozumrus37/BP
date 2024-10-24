import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
from compute_iou import *
import sys
import argparse
from utilities_eval import *
from Sam2RealtimeTracker import Sam2RealtimeTracker
from tqdm import tqdm

from hq_sam.sam_hq.segment_anything import sam_model_registry, SamPredictor

torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False

from sam2.build_sam import build_sam2_video_realtime_predictor

BASE_dir = "/datagrid/personal/rozumrus/BP_dg/output_vot22ST"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alfa', type=float, default=0)
    parser.add_argument('--exclude_empty_masks', action="store_true")
    parser.add_argument('--no_memory', action="store_true")
    parser.add_argument('--num_obj_ptrs', type=int, default=16)
    parser.add_argument('--vis_out', action="store_true")
    parser.add_argument('--use_large_SAM2', action="store_true")
    parser.add_argument('--vis_IoU_graph', action="store_true")
    parser.add_argument('--save_ious', action="store_true")
    parser.add_argument('--memory_stride', type=int, default=1)
    parser.add_argument('--sequences')

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return args.alfa, args.exclude_empty_masks, args.no_memory, args.num_obj_ptrs, args.vis_out, args.use_large_SAM2, args.vis_IoU_graph, args.save_ious, args.memory_stride, args.sequences

(alfa, exclude_empty_masks, 
no_memory, num_obj_ptrs, vis_out,
use_large_SAM2, vis_IoU_graph, save_ious, memory_stride, sequences) = parse_args()


def run(seq, alfa=0.1, to_save_path="ious"):
    VIDEO_NAME = seq
    video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + VIDEO_NAME + "/color" 

    tracker = Sam2RealtimeTracker(image_path=os.path.join(video_dir, '00000001.jpg'), 
        VIDEO_NAME=VIDEO_NAME, alfa=alfa, 
        num_obj_ptrs=num_obj_ptrs, memory_temporal_stride_for_eval_r=memory_stride, use_large_SAM2=use_large_SAM2, exclude_empty_masks=exclude_empty_masks, no_memory=no_memory, 
        vis_out=vis_out, to_save_path=to_save_path)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names = frame_names[1:]
    frame_index = 1
    masks_out = []

    for imagefile in frame_names: 
        out_obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = tracker.track(os.path.join(video_dir, imagefile), frame_index)  
        masks_out.append(out_mask_logits)
        frame_index += 1

    return [tracker.mask_first_frame] + masks_out


SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
SEQ= ['flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
SEQ=['nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
#SEQ=['nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1'] 



if sequences != None:
    SEQ = sequences

all_ious = []

large_or_small = 'L' if use_large_SAM2 else 'S'
base_dir_save = os.path.join(BASE_dir, "alfa" + str(alfa) + "_nomem" + str(int(no_memory)) + "_excl_EM" + str(int(exclude_empty_masks)) + "_OP" + str(num_obj_ptrs) + "_" + large_or_small)
iou_save = base_dir_save + '/ious.txt'
save_names = base_dir_save + '/names.txt'


for seq in SEQ:
    to_save_path = os.path.join(base_dir_save, seq)

    masks_out = run(seq, alfa, to_save_path)

    iou_curr = get_iou(seq, masks_out)

    all_ious.append(iou_curr)

    print(f"IoU fpr {seq} is: {iou_curr}\n")

    print("----------\n")
    print("All ious:\n")

    for iou_i in all_ious:
        print(f"{iou_i}")

    if save_ious:
        with open(os.path.join(to_save_path, 'iou.txt'), 'a') as file:
            file.write(f"Sequence : {seq} with iou : {iou_curr}\n")

        with open(iou_save, 'a') as file:
            file.write(f"{iou_curr}\n")

        with open(save_names, 'a') as file:
            file.write(f"Sequence : {seq}\n")














