import torch
from sam2.build_sam import build_sam2_video_predictor
from utilities_eval import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from compute_iou import *
from utilities_eval import *

import argparse
from tqdm import tqdm

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']


# SEQ=['flamingo1', 'nature', 'girl']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_empty_masks', action="store_true")
    parser.add_argument('--vis_out', action="store_true")
    parser.add_argument('--memory_stride', type=int, default=1)
    parser.add_argument('--sequences')
    parser.add_argument('--crop_gt', action="store_true")
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--use_prev_box', action="store_true")

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return args.exclude_empty_masks, args.vis_out, args.memory_stride, args.crop_gt, args.factor, args.use_prev_box, args.sequences

exclude_empty_masks, vis_out, memory_stride, crop_gt, factor, use_prev_box, sequences = parse_args()

if sequences != None:
    SEQ = sequences

def run_eval(seq):

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" #/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt" #"./checkpoints/sam2_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" #"sam2_hiera_l.yaml" #"configs/sam2/sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
    video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + seq + "/color"
    output_dir = "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + str(seq)
    frame_names = load_frames(video_dir)
    inference_state = predictor.init_state()

    mask_first_frame = get_nth_mask(seq, 0)
    H, W = mask_first_frame.shape 
    bbox = None

    if crop_gt or use_prev_box:
        min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame)

        print(f"Number of occupied pixels initially: ", abs(min_col-max_col) * abs(min_row-max_row) * 100.0 / (H*W))

        min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=factor)

        bbox = (min_row, min_col, max_row, max_col)
        mask_first_frame = mask_first_frame[min_col:max_col, min_row:max_row]

        print(f"Number of occupied pixels by factor {factor}:", abs(min_col-max_col) * abs(min_row-max_row) * 100.0 / (H*W))

    image_path = os.path.join(video_dir, '00000001.jpg')

    predictor.load_first_frame(inference_state, image_path, bbox=bbox)
    _, _, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=np.array(mask_first_frame),
    )

    mask_full_size = get_full_size_mask(out_mask_logits, bbox, image_path, 0, H, W)
    
    if vis_out:
        vis(mask_full_size, [1], 0, os.path.join(video_dir, frame_names[0]), output_dir)

        img_pil_full_res = Image.open(image_path) 
        img_pil = img_pil_full_res.crop(bbox)
         
        img_pil.save(output_dir + '/z0_cropped' + str(0) +'.png')

    masks_all = []

    if use_prev_box:
        prev_bbox = (min_row, min_col, max_row, max_col) # bbox

    for out_frame_idx in tqdm(range(1, len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])

        # if out_frame_idx <= 14 or out_frame_idx == 16 or out_frame_idx == 17:
        #     mask_curr = get_nth_mask(seq, out_frame_idx)
        #     # if get_bounding_box(mask_curr) == None:
        #     #     predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=o)
        #     #     _, _, out_mask_logits = predictor.add_new_mask(
        #     #         inference_state=inference_state,
        #     #         frame_idx=out_frame_idx,
        #     #         obj_id=1,
        #     #         mask=mask_curr,
        #     #     )
        #     #     continue

        #     min_row, min_col, max_row, max_col = get_bounding_box(mask_curr)

        #     min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=factor)

        #     bbox = (min_row, min_col, max_row, max_col)
        #     mask_curr = mask_curr[min_col:max_col, min_row:max_row]

        #     predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=bbox)
        #     _, _, out_mask_logits = predictor.add_new_mask(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=1,
        #         mask=np.array(mask_curr),
        #     )

        #     mask_full_size = get_full_size_mask(out_mask_logits, bbox, image_path, out_frame_idx, seq)
        #     masks_all.append(mask_full_size)

        #     if vis_out:
        #         vis(mask_full_size, [1], out_frame_idx, image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq)

        #     continue



        if vis_out:

            if prev_bbox != None:
                print("Area: ", out_frame_idx, abs(prev_bbox[0] - prev_bbox[2]) * abs(prev_bbox[1] - prev_bbox[3]))

            img_pil_full_res = Image.open(image_path) 
            img_pil = img_pil_full_res.crop(prev_bbox)
             
            img_pil.save(output_dir + '/z_cropped' + str(out_frame_idx) +'.png')




        bbox = None

        if use_prev_box:
            bbox = prev_bbox

        if crop_gt:
            mask_curr = get_nth_mask(seq, out_frame_idx)
            bbox = get_bounding_box(mask_curr)
     
            if bbox != None:

                if vis_out:
                    img_pil_full_res = Image.open(image_path) 
                    img_pil = img_pil_full_res.crop(bbox)
                     
                    img_pil.save(output_dir + '/z_cropped' + str(out_frame_idx) +'.png')


                min_row, min_col, max_row, max_col = bbox 
                min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor)
                bbox = min_row, min_col, max_row, max_col 

        # if out_frame_idx == 16 or out_frame_idx == 17 or out_frame_idx == 17:
        #     bbox = None

        predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=bbox)

        out_frame_idx, out_obj_ids, out_mask_logits = predictor.track(inference_state, 
            exclude_empty_masks=exclude_empty_masks, memory_stride=memory_stride, frame_idx=out_frame_idx)

        mask_full_size = get_full_size_mask(out_mask_logits, bbox, image_path, out_frame_idx, H, W)
        masks_all.append(mask_full_size)

        if vis_out:
            vis(mask_full_size, out_obj_ids, out_frame_idx, image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq)

        if use_prev_box:
            bbox = get_bounding_box(mask_full_size)

            if bbox != None:
                min_row, min_col, max_row, max_col = bbox
                min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor)

                area_i = None 

                if prev_bbox != None:
                    area_i = abs(prev_bbox[0] - prev_bbox[2]) * abs(prev_bbox[1] - prev_bbox[3])

                if min_row-max_row != 0 and min_col-max_col != 0:
                    prev_bbox = (min_row, min_col, max_row, max_col)

                area_i_next = abs(prev_bbox[0] - prev_bbox[2]) * abs(prev_bbox[1] - prev_bbox[3])

                # if area_i != None and area_i_next > area_i:
                #     min_row, min_col, max_row, max_col = prev_bbox
                #     min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, 1.0 * area_i / area_i_next)

                #     if min_row-max_row != 0 and min_col-max_col != 0:
                #         prev_bbox = (min_row, min_col, max_row, max_col)
            else:
                # min_row, min_col, max_row, max_col = prev_bbox
                # min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, 2)

                # area_i = None 

                # if prev_bbox != None:
                #     area_i = abs(prev_bbox[0] - prev_bbox[2]) * abs(prev_bbox[1] - prev_bbox[3])

                # if min_row-max_row != 0 and min_col-max_col != 0:
                #     prev_bbox = (min_row, min_col, max_row, max_col)

                # print(out_frame_idx)
                prev_bbox = None


        
    return [mask_first_frame] + masks_all


def vis(mask_full_size, out_obj_ids, ann_frame_idx, image, to_save_path):
    image = Image.open(image)

    plt.clf()
    plt.cla()

    plt.imshow(image)

    show_mask(mask_full_size, plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)


all_ious = []
cnt = 1

for seq in SEQ:
    masks_all = run_eval(seq)

    iou_curr = get_iou(seq, masks_all)

    all_ious.append(iou_curr)

    print(f"IoU fpr {seq} is: {iou_curr}\n")

    print("----------\n")
    print("All ious:\n")

    for iou_i in all_ious:
        print(f"{iou_i}")


    print(f"The mean after processing {cnt} seqs is: {np.array(all_ious).mean()}")

    cnt+=1



