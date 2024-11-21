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

from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import torch.nn as nn

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model_dino = AutoModel.from_pretrained('facebook/dinov2-giant').to('cuda')

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument('--use_square_box', action="store_true")
    parser.add_argument('--no_mask_set_larger_prev_bbox', action="store_true")
    parser.add_argument('--no_mask_set_whole_image', action="store_true")
    parser.add_argument('--double_memory_bank', action="store_true")
    parser.add_argument('--oracle_threshold', type=float, default=20)
    parser.add_argument('--uncroped_mask_for_double_MB', action="store_true")
    parser.add_argument('--use_prev_mask', action="store_true")
    parser.add_argument('--oracle', action="store_true")

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return (args.exclude_empty_masks, args.vis_out, args.memory_stride, 
    args.crop_gt, args.factor, args.use_prev_box, args.use_square_box, 
    args.no_mask_set_larger_prev_bbox, args.no_mask_set_whole_image, 
    args.double_memory_bank, args.uncroped_mask_for_double_MB, args.oracle, args.oracle_threshold, args.use_prev_mask, args.sequences)

(exclude_empty_masks, vis_out, memory_stride, 
crop_gt, factor, use_prev_box, use_square_box, 
no_mask_set_larger_prev_bbox, no_mask_set_whole_image, 
double_memory_bank, uncroped_mask_for_double_MB, oracle, oracle_threshold, use_prev_mask, sequences) = parse_args()

if sequences != None:
    SEQ = sequences

def run_eval(seq):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # sam2_checkpoint =  "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt" 
    # model_cfg = "configs/sam2/sam2_hiera_l.yaml" 

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
    video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + seq + "/color"
    output_dir = "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + str(seq) + "_oracle" if oracle else "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + str(seq) 
    frame_names = load_frames(video_dir, double_memory_bank)
    inference_state = predictor.init_state()
    mask_first_frame = get_nth_mask(seq, 0)
    H, W = mask_first_frame.shape 
    bbox = None
    prev_mask = None
    USE_PREV_BOX = use_prev_box
    

    if crop_gt or use_prev_box:
        min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame)

        area = abs(min_row-max_row) * abs(min_col-max_col)

        print(min_row, min_col, max_row, max_col, area / (H*W*1.0), H, W)

        if use_square_box:
            min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=factor)
        else:
            min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=factor)


        area = abs(min_row-max_row) * abs(min_col-max_col)

        print(min_row, min_col, max_row, max_col, area / (H*W*1.0))
        
        # if area / (H*W*1.0) >= 0.1:
        #     USE_PREV_BOX = False
        # else:
        bbox = (min_row, min_col, max_row, max_col)
        mask_first_frame = mask_first_frame[min_col:max_col, min_row:max_row]

    image_path = os.path.join(video_dir, '00000001.jpg')
    start_idx = 1

    if double_memory_bank:
        predictor.load_first_frame(inference_state, image_path, bbox=bbox,frame_idx=0)
        _, _, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=np.array(mask_first_frame),
        )

        mask_full_size = get_full_size_mask(out_mask_logits, bbox, H, W)

        predictor.load_first_frame(inference_state, image_path, bbox=None, frame_idx=1)
        _, _, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=1,
            obj_id=1,
            mask=np.array(get_nth_mask(seq, 0)),
        )
        start_idx = 2

        if vis_out:
            vis(get_full_size_mask(out_mask_logits, None, H, W), [1], 0, 
                image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/full_resol_" + seq)
    else:
        predictor.load_first_frame(inference_state, image_path, bbox=bbox,frame_idx=0)
        _, _, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=np.array(mask_first_frame),
        )

        mask_full_size = get_full_size_mask(out_mask_logits, bbox, H, W)

    prev_mask = [mask_full_size]

    if vis_out:
        vis(mask_full_size, [1], 0, os.path.join(video_dir, frame_names[0]), output_dir)

        img_pil_full_res = Image.open(image_path) 
        img_pil = img_pil_full_res.crop(bbox)
         
        img_pil.save(output_dir + '/cropped' + str(0) +'.png')

    masks_all = []
    prev_bbox = None
    missed_best_mask = 0

    if USE_PREV_BOX or crop_gt:
        prev_bbox = (min_row, min_col, max_row, max_col)

    previous_masks = [prev_mask]

    for out_frame_idx in tqdm(range(start_idx, len(frame_names), start_idx)):
        # print(out_frame_idx)
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])

        if vis_out and USE_PREV_BOX:
            img_pil_full_res = Image.open(image_path) 
            img_pil = img_pil_full_res.crop(prev_bbox)
            img_pil.save(output_dir + '/cropped' + str(out_frame_idx) +'.png')

        bbox = None

        if USE_PREV_BOX:
            bbox = prev_bbox

        if crop_gt:
            mask_curr = get_nth_mask(seq, out_frame_idx)
            bbox = get_bounding_box(mask_curr)
     
            if bbox != None:
                min_row, min_col, max_row, max_col = bbox 
                if use_square_box:
                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, 
                        min_col, min_row, max_col, max_row, factor=factor)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, 
                        min_col, min_row, max_col, max_row, factor=factor)

                bbox = min_row, min_col, max_row, max_col 

        predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=bbox)

        seq_to_pass = None

        if oracle:
            seq_to_pass = seq

        if not use_prev_mask:
            previous_masks = None 
            prev_mask = None


        out_frame_idx, out_obj_ids, out_mask_logits, out_curr, miss, video_res_masks_second_best = predictor.track(inference_state, 
            exclude_empty_masks=exclude_empty_masks, 
            memory_stride=memory_stride, 
            frame_idx=out_frame_idx, 
            double_memory_bank=double_memory_bank,
            video_H=H,
            video_W=W,
            seq=seq_to_pass,
            oracle_threshold=oracle_threshold,
            prev_mask=prev_mask, #previous_masks,#prev_mask, , #
            processor_dino=processor_dino,
            model_dino=model_dino)

        missed_best_mask += miss

        mask_full_size = get_full_size_mask(out_mask_logits, bbox, H, W)
        
        mask_full_size_second_best = None 


        if out_frame_idx == 19:



            # Save the array to a text file
            np.savetxt('array.txt', mask_full_size, fmt='%d')

            print(mask_full_size.shape)
            # print(mask_full_size)

        if video_res_masks_second_best != None:
            mask_full_size_second_best = get_full_size_mask(video_res_masks_second_best, bbox, H, W) 

        if double_memory_bank:
            predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=None)
            predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx+1, bbox=None)

            out_frame_idx, out_obj_ids, out_mask_logits, out_curr, miss, video_res_masks_second_best = predictor.track(inference_state, 
                exclude_empty_masks=exclude_empty_masks, 
                memory_stride=memory_stride, 
                frame_idx=out_frame_idx, 
                prev_out=out_curr, 
                double_memory_bank=double_memory_bank,
                video_H=H,
                video_W=W,
                seq=seq_to_pass,
                oracle_threshold=oracle_threshold,
                prev_mask=None)

            if uncroped_mask_for_double_MB:
                mask_full_size = get_full_size_mask(out_mask_logits, None, H, W)

            if vis_out:
                vis(get_full_size_mask(out_mask_logits, None, image_path, H, W), out_obj_ids, out_frame_idx//2, 
                    image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/full_resol_" + seq)

        
        masks_all.append(mask_full_size)
        prev_mask = [mask_full_size]
        previous_masks = [mask_first_frame] + masks_all[-6:]


        if vis_out:
            if oracle:
                vis(mask_full_size, out_obj_ids, out_frame_idx//2 if double_memory_bank else out_frame_idx, 
                image_path, output_dir)

                vis(mask_full_size_second_best, out_obj_ids, out_frame_idx//2 if double_memory_bank else out_frame_idx, 
                    image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq + "_sam2_best_pred")
            else:
                vis(mask_full_size, out_obj_ids, out_frame_idx//2 if double_memory_bank else out_frame_idx, 
                image_path, output_dir)



        if USE_PREV_BOX:
            bbox = get_bounding_box(mask_full_size)

            if bbox != None:
                min_row, min_col, max_row, max_col = bbox
                
                if use_square_box:
                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=factor)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=factor)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    prev_bbox = (min_row, min_col, max_row, max_col)    
                else:
                    print("Mask prediction is just a line or empty!") 
            elif no_mask_set_larger_prev_bbox:
                min_row, min_col, max_row, max_col = prev_bbox
                
                if use_square_box:
                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=2)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=2)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    prev_bbox = (min_row, min_col, max_row, max_col)
                else:
                    print("Mask prediction is just a line or empty!")
            elif no_mask_set_whole_image:
                prev_bbox = None
    
    if oracle:    
        print(f"For {seq} number of missed_best_mask is {missed_best_mask} and the percanrge is: {(missed_best_mask / len(masks_all))*100: .2f} %")
    return [mask_first_frame] + masks_all, (missed_best_mask / len(masks_all))*100

def vis(mask_full_size, out_obj_ids, ann_frame_idx, image, to_save_path):
    image = Image.open(image)
    plt.clf()
    plt.cla()
    plt.axis('off')
    plt.imshow(image)


    show_mask(mask_full_size, plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)

all_ious = []

total_miss = 0

for seq in SEQ:
    masks_all, missed_masks_perc = run_eval(seq)
    iou_curr = get_iou(seq, masks_all)
    all_ious.append(iou_curr)

    total_miss += missed_masks_perc

    print(f"IoU for {seq} is: {iou_curr}")

if sequences == None:
    for iou_i in all_ious:
        print(f"{iou_i}")

    print(f"The mean after processing seqs is: {np.array(all_ious).mean()}")

if oracle:
    print("Mean miss: ", total_miss / len(SEQ))



