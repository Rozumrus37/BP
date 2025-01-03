import torch
from sam2.build_sam import build_sam2_video_predictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from compute_iou import *
from utilities_eval import *
from vis_outputs import *

import argparse
from tqdm import tqdm
import gc 
import torch.nn as nn
import csv

# use bfloat16 as in the SAM2 oficial example
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

# all sequences from the VOT2022
SEQ_VOT2022 = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
SEQ_VOT2020 = ["agility", "ants1", "ball2", "ball3", "basketball", "birds1", "bolt1", "book", "butterfly", "car1", "conduction1", "crabs1", "dinosaur", "dribble", "drone1", "drone_across", "drone_flip", "fernando", "fish1", "fish2", "flamingo1", "frisbee", "girl", "glove", "godfather", "graduate", "gymnastics1", "gymnastics2", "gymnastics3", "hand", "hand02", "hand2", "handball1", "handball2", "helicopter", "iceskater1", "iceskater2", "lamb", "leaves", "marathon", "matrix", "monkey", "motocross1", "nature", "polo", "rabbit", "rabbit2", "road", "rowing", "shaking", "singer2", "singer3", "soccer1", "soccer2", "soldier", "surfing", "tiger", "wheel", "wiper", "zebrafish1"]
SEQ=SEQ_VOT2022

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_empty_masks', action="store_true")
    parser.add_argument('--vis_out', action="store_true")
    parser.add_argument('--memory_stride', type=int, default=1)
    parser.add_argument('--sequences')
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--use_prev_box', action="store_true")
    parser.add_argument('--save_res_path', default="output.csv")
    parser.add_argument('--stack', default="vot2022ST")
    parser.add_argument('--thr_IoU_BB1_BBsm', type=float, default=0.1)
    parser.add_argument('--thr_Amask_to_Abb', type=float, default=0.1)

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return (args.exclude_empty_masks, args.thr_IoU_BB1_BBsm, args.thr_Amask_to_Abb, args.vis_out, args.memory_stride, args.factor, args.use_prev_box, args.save_res_path, args.stack, args.sequences)

(exclude_empty_masks, thr_IoU_BB1_BBsm, thr_Amask_to_Abb, vis_out, memory_stride, factor, use_prev_box, save_res_path, stack, sequences) = parse_args()

if sequences != None:
    SEQ = sequences
elif stack == "vot2020ST":
    SEQ = SEQ_VOT2020

def run_eval(seq):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_dir = f"/mnt/data_personal/rozumrus/BP_dg/{stack}/sequences/" + seq + "/color" 
    output_dir = "/mnt/data_personal/rozumrus/BP_dg/sam2.1_output/" + str(seq) 
    start_idx, OBJ_ID = 1, 1
    img_path_first_frame = os.path.join(video_dir, '00000001.jpg')
    frame_names = load_frames(video_dir)
    
    predictors, prev_bboxs, masks_all = [], [], []
    Fs = [4, 2, 1]

    for F in Fs:
        mask_first_frame = get_nth_mask(seq, 0, stack=stack)
        H, W = mask_first_frame.shape 
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
        inference_state = predictor.init_state()
        prev_bbox = None

        if use_prev_box:  
            min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame) # get the bbox of the first segmentation

            area_initial_bbox = abs(min_row - max_row) * abs(min_col - max_col) # area of the bbox of the first segmentaiton

            print(f"Bbox coords before increasing: {min_row}, {min_col}, {max_row}, {max_col}")
            print(f"H, W of the image: {H}, {W}")
            print(f"H, W of the original bbox: {max_col-min_col}, {max_row-min_row}")
            print(f"Initial bbox proportion to the image resol: {area_initial_bbox * 100 / (H * W * 1.0)}\n")

            # increase the area of the bbox by factor
            min_row, min_col, max_row, max_col = increase_bbox_area_for_parallel(H, W, min_row, min_col, max_row, max_col, factor=F)

            # enlarged bbox
            area_increased_bbox = abs(min_row - max_row) * abs(min_col - max_col)

            print(f"Bbox coords after increasing: {min_row}, {min_col}, {max_row}, {max_col}")
            print(f"H, W of the enlarged bbox: {max_col-min_col}, {max_row-min_row}")
            print(f"Increased bbox proportion to the image resol: {area_increased_bbox * 100 / (H * W * 1.0)}")

            # save the new enlarged bbox into bbox
            prev_bbox = (min_row, min_col, max_row, max_col)
            prev_bboxs.append(prev_bbox)

            # crop the first binary mask by enlarged bbox
            mask_first_frame = mask_first_frame[min_row:max_row, min_col:max_col]

        # load first frame into the inference_state
        predictor.load_first_frame(inference_state, img_path_first_frame, bbox=prev_bbox, frame_idx=start_idx-1)
        _, _, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_idx-1,
            obj_id=OBJ_ID,
            mask=mask_first_frame,
        )

        mask_full_size = get_full_size_mask(out_mask_logits, prev_bbox, H, W)

        predictors.append((predictor,inference_state))


    if vis_out:
        vis(mask_full_size, [OBJ_ID], 0, os.path.join(video_dir, frame_names[0]), output_dir)

        if use_prev_box or crop_gt:
            vis_cropped(mask_full_size, [OBJ_ID], "0", img_path_first_frame, prev_bbox, output_dir)


    for out_frame_idx in tqdm(range(start_idx, len(frame_names))):
        gt_i = get_nth_mask(seq, out_frame_idx, stack=stack)
        image_path = os.path.join(video_dir, frame_names[out_frame_idx]) # path to the current frame
        masks_i = []

        for i in range(len(Fs)):
            predictor, inference_state = predictors[i][0], predictors[i][1]

            # load current frame into the inference_state 
            predictor.load_first_frame(predictors[i][1], image_path, frame_idx=out_frame_idx, bbox=prev_bboxs[i])

            # get the output for the current frame
            out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = predictor.track(
                inference_state, 
                exclude_empty_masks=exclude_empty_masks, 
                memory_stride=memory_stride, 
                frame_idx=out_frame_idx, 
                video_H=H,
                video_W=W)

            # put the mask back into the original image dimensions
            mask_full_size = get_full_size_mask(out_mask_logits, prev_bboxs[i], H, W)
            masks_i.append(mask_full_size)


            if use_prev_box:
                temp_bbox = get_bounding_box(mask_full_size)

                if temp_bbox != None:
                    min_row, min_col, max_row, max_col = temp_bbox
                    min_row, min_col, max_row, max_col = increase_bbox_area_for_parallel(H, W, min_row, min_col, max_row, max_col, factor=Fs[i])
                    prev_bboxs[i] = (min_row, min_col, max_row, max_col)       
            

        best_score, best_mask = -1, masks_i[len(Fs)-1]
        mask_ori = masks_i[len(Fs)-1] # SAM2 original mask output

        for i in range(len(Fs)):
            IoU_mask_i_with_mask_ori = obatin_iou(masks_i[i], mask_ori) 
            mask_curr_bbox = get_bounding_box(masks_i[i]) 
            if mask_curr_bbox != None:
                H_curr_bbox, W_curr_bbox = mask_curr_bbox[0] - mask_curr_bbox[2], mask_curr_bbox[1] - mask_curr_bbox[3]
            else:
                H_curr_bbox, W_curr_bbox = 0, 0

            area_mask_to_area_whole_BB = H_curr_bbox * W_curr_bbox / ((H * W) / (Fs[i]**2))

            if (IoU_mask_i_with_mask_ori > thr_IoU_BB1_BBsm and (area_mask_to_area_whole_BB  <= thr_Amask_to_Abb)) or np.sum(mask_ori) == 0:
                #print(area_mask_to_area_whole_BB*100, IoU_mask_i_with_mask_ori)#np.sum(masks_i[i]) / (H*W/(Fs[i]**2)) * 100)
                best_mask = masks_i[i]
                break

            # iou_curr = obatin_iou(gt_i, masks_i[i])

            # if iou_curr > best_score:
            #     best_mask = masks_i[i]
            #     best_score = iou_curr


        if vis_out:
            vis(best_mask, out_obj_ids, out_frame_idx, image_path, output_dir)

        masks_all.append(best_mask)


    for (predictor, inference_state) in predictors:
        del predictor, inference_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    return [mask_first_frame] + masks_all

def main():
    all_ious = []

    # run through all sequences from VOT20/22
    for seq in SEQ:
        # run SAM2 on one sequence to get the output masks
        masks_all = run_eval(seq)

        # get mIoU between output masks and gt for the given sequnces
        iou_curr = get_iou(seq, masks_all,stack=stack)
        all_ious.append(iou_curr)

        # save the mIoU into the .csv file
        with open(save_res_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([seq, iou_curr])

        print(f"IoU for {seq} is: {iou_curr}")

    # output to stdout all mIoUs
    for iou_i in all_ious:
        print(f"{iou_i}")

    print(f"The mean after processing seqs is: {np.array(all_ious).mean()}")

    # write the mean of all mIoUs into the .csv file
    with open(save_res_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["mean", np.array(all_ious).mean()])

if __name__ == "__main__":
    main()
