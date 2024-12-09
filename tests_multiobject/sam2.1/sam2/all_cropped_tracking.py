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
SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_empty_masks', action="store_true")
    parser.add_argument('--vis_out', action="store_true")
    parser.add_argument('--memory_stride', type=int, default=1)
    parser.add_argument('--sequences')
    parser.add_argument('--crop_gt', action="store_true")
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--use_prev_box', action="store_true")
    parser.add_argument('--save_res_path', default="output.csv")
    parser.add_argument('--prev_mask_increase_when_empty', action="store_true")
    parser.add_argument('--use_log_memory_stride', action="store_true")
    parser.add_argument('--min_box_factor', type=float, default=256)

    parser.add_argument('--direct_comp_to_prev_pred', action="store_true")
    parser.add_argument('--alfa_flow', type=float, default=0.000001)
    parser.add_argument('--forward_of', action="store_true")
    parser.add_argument('--backward_of', action="store_true")
    parser.add_argument('--interpolation', type=str, default="bilinear")
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--close_trans', action="store_true")
    parser.add_argument('--open_trans', action="store_true")
    parser.add_argument('--oracle', action="store_true")
    parser.add_argument('--oracle_threshold', type=float, default=0)

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")


    return (args.exclude_empty_masks, args.vis_out, args.memory_stride, 
    args.crop_gt, args.factor, args.use_prev_box, 
    args.prev_mask_increase_when_empty, args.oracle, args.oracle_threshold, 
    args.alfa_flow, args.save_res_path, args.backward_of, args.direct_comp_to_prev_pred, 
    args.interpolation, args.kernel_size, args.close_trans, args.open_trans, args.forward_of, args.use_log_memory_stride, args.sequences)

(exclude_empty_masks, vis_out, memory_stride, 
crop_gt, factor, use_prev_box, 
prev_mask_increase_when_empty, oracle, oracle_threshold, 
alfa_flow, save_res_path, backward_of, direct_comp_to_prev_pred, 
interpolation, kernel_size, close_trans, open_trans, forward_of, use_log_memory_stride, sequences) = parse_args()

if sequences != None:
    SEQ = sequences

def run_eval(seq):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + seq + "/color"
    output_dir = "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + str(seq) 
    img_path_first_frame = os.path.join(video_dir, '00000001.jpg')
    frame_names = load_frames(video_dir)
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
    inference_state = predictor.init_state()

    start_idx = 1
    OBJ_ID = 1
    mask_first_frame = get_nth_mask(seq, 0)
    H, W = mask_first_frame.shape 
    prev_bbox = None
    prev_mask = None
    masks_all = []
    missed_best_mask = 0

    if crop_gt or use_prev_box:  
        min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame) # get the bbox of the first segmentation
        #import pdb; pdb.set_trace()

        area_initial_bbox = abs(min_row - max_row) * abs(min_col - max_col) # area of the bbox of the first segmentaiton

        print(f"Bbox coords before increasing: {min_row}, {min_col}, {max_row}, {max_col}")
        print(f"H, W of the image: {H}, {W}")
        print(f"H, W of the original bbox: {max_col-min_col}, {max_row-min_row}")
        print(f"Initial bbox proportion to the image resol: {area_initial_bbox * 100 / (H * W * 1.0)}\n")

        # increase the area of the bbox by factor
        min_row, min_col, max_row, max_col = increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=min_box_factor, factor=factor)
        #import pdb; pdb.set_trace()

        # enlarged bbox
        area_increased_bbox = abs(min_row - max_row) * abs(min_col - max_col)

        print(f"Bbox coords after increasing: {min_row}, {min_col}, {max_row}, {max_col}")
        print(f"H, W of the enlarged bbox: {max_col-min_col}, {max_row-min_row}")
        print(f"Increased bbox proportion to the image resol: {area_increased_bbox * 100 / (H * W * 1.0)}")

        # save the new enlarged bbox into bbox
        prev_bbox = (min_row, min_col, max_row, max_col)
        #import pdb; pdb.set_trace()

        # crop the first binary mask by enlarged bbox
        mask_first_frame = mask_first_frame[min_row:max_row, min_col:max_col]
        #import pdb; pdb.set_trace()


    # load first frame into the inference_state
    predictor.load_first_frame(inference_state, img_path_first_frame, bbox=prev_bbox, frame_idx=start_idx-1)
    _, _, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=start_idx-1,
        obj_id=OBJ_ID,
        mask=mask_first_frame,
    )

    #import pdb; pdb.set_trace()

    seq_to_pass = None 

    if oracle or forward_of or backward_of:
        seq_to_pass = seq 

    # put the cropped mask back into the original mask dimensions
    mask_full_size = get_full_size_mask(out_mask_logits, prev_bbox, H, W)


    if backward_of or forward_of or direct_comp_to_prev_pred:
        prev_mask = [mask_full_size]


    if vis_out:
        vis(mask_full_size, [OBJ_ID], 0, os.path.join(video_dir, frame_names[0]), output_dir)

        if use_prev_box or crop_gt:
            vis_cropped(img_path_first_frame, prev_bbox, output_dir, "0")


    for out_frame_idx in tqdm(range(start_idx, len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[out_frame_idx]) # path to the current frame

        # in case we use the previous box then visualize its cropped version
        if vis_out and use_prev_box and not crop_gt: 
            vis_cropped(image_path, prev_bbox, output_dir, out_frame_idx)

        # if crop by gt bbox
        if crop_gt:
            mask_curr = get_nth_mask(seq, out_frame_idx) # get the current gt mask
            temp_bbox = get_bounding_box(mask_curr) # get its bbox

            if temp_bbox != None:
                min_row, min_col, max_row, max_col  = temp_bbox # unpack
                min_row, min_col, max_row, max_col = increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=min_box_factor, factor=factor)
                
                # save the new enlarged bbox into bbox
                prev_bbox = (min_row, min_col, max_row, max_col)

                if vis_out:
                    vis_cropped(image_path, prev_bbox, output_dir, out_frame_idx)

        # load current frame into the inference_state 
        predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=prev_bbox)

        # get the output for the current frame
        out_frame_idx, out_obj_ids, out_mask_logits, _, miss, _ = predictor.track(inference_state, 
            exclude_empty_masks=exclude_empty_masks, 
            memory_stride=memory_stride, 
            frame_idx=out_frame_idx, 
            video_H=H,
            video_W=W,
            seq=seq_to_pass,
            oracle_threshold=oracle_threshold,
            prev_mask=prev_mask,
            alfa_flow=alfa_flow,
            direct_comp_to_prev_pred=direct_comp_to_prev_pred,
            backward_of=backward_of,
            interpolation=interpolation, 
            kernel_size=kernel_size, 
            close_trans=close_trans, 
            open_trans=open_trans,
            use_log_memory_stride=use_log_memory_stride,
            forward_of=forward_of,
            oracle=oracle)


        missed_best_mask += miss

        # put the mask back into the original image dimensions
        mask_full_size = get_full_size_mask(out_mask_logits, prev_bbox, H, W)
        masks_all.append(mask_full_size)

        if backward_of or forward_of or direct_comp_to_prev_pred:
            prev_mask = [mask_full_size]

        #import pdb; pdb.set_trace()

        if vis_out:
            vis(mask_full_size, out_obj_ids, out_frame_idx, image_path, output_dir)

        if use_prev_box:
            temp_bbox = get_bounding_box(mask_full_size)

            if temp_bbox != None:
                # import pdb; pdb.set_trace()
                min_row, min_col, max_row, max_col = temp_bbox
                min_row, min_col, max_row, max_col = increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=min_box_factor, factor=factor)
                prev_bbox = (min_row, min_col, max_row, max_col)       
            elif prev_mask_increase_when_empty:
                min_row, min_col, max_row, max_col = prev_bbox
                min_row, min_col, max_row, max_col = increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=min_box_factor, factor=2)
                prev_bbox = (min_row, min_col, max_row, max_col)


    del predictor, inference_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    if oracle:    
        print(f"For {seq} number of missed_best_mask is {missed_best_mask} and the percanrge is: {(missed_best_mask / len(masks_all))*100: .2f} %")

    return [mask_first_frame] + masks_all, (missed_best_mask / len(masks_all))*100



def main():
    all_ious = []
    total_miss = 0

    # run through all sequences from VOT2022
    for seq in SEQ:
        # run SAM2 on one sequence to get the output masks
        masks_all, missed_masks_perc = run_eval(seq)

        # get mIoU between output masks and gt for the given sequnces
        iou_curr = get_iou(seq, masks_all)
        all_ious.append(iou_curr)

        total_miss += missed_masks_perc

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

    if oracle:
        print("Mean miss: ", total_miss / len(SEQ))

if __name__ == "__main__":
    main()
