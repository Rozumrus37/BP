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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']

# SEQ=['flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
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
    parser.add_argument('--use_square_box', action="store_true")
    parser.add_argument('--no_mask_set_full_image', action="store_true")

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return args.exclude_empty_masks, args.vis_out, args.memory_stride, args.crop_gt, args.factor, args.use_prev_box, args.use_square_box, args.no_mask_set_full_image, args.sequences

exclude_empty_masks, vis_out, memory_stride, crop_gt, factor, use_prev_box, use_square_box, no_mask_set_full_image, sequences = parse_args()

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

    
    image_path = os.path.join(video_dir, '00000001.jpg')


    predictor.load_first_frame(inference_state, image_path, bbox=bbox,frame_idx=0)
    _, _, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=np.array(mask_first_frame),
    )

  

    mask_full_size = get_full_size_mask(out_mask_logits, bbox, image_path, 0, H, W)

    # predictor.load_first_frame(inference_state, image_path, bbox=None, frame_idx=1)
    # _, _, out_mask_logits = predictor.add_new_mask(
    #     inference_state=inference_state,
    #     frame_idx=1,
    #     obj_id=1,
    #     mask=np.array(get_nth_mask(seq, 0)),
    # )

    
    if vis_out:
        vis(mask_full_size, [1], 0, os.path.join(video_dir, frame_names[0]), output_dir)

        img_pil_full_res = Image.open(image_path) 
        img_pil = img_pil_full_res.crop(bbox)
         
        img_pil.save(output_dir + '/z0_cropped' + str(0) +'.png')

    masks_all = []

    prev_bbox = None
    
    if use_prev_box or crop_gt:
        prev_bbox = (min_row, min_col, max_row, max_col) # bbox

    for out_frame_idx in tqdm(range(1, len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])

        if vis_out:
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
                if use_square_box:
                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=factor)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=factor)
                bbox = min_row, min_col, max_row, max_col 

        predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx, bbox=bbox)

        out_frame_idx, out_obj_ids, out_mask_logits = predictor.track(inference_state, 
            exclude_empty_masks=exclude_empty_masks, memory_stride=memory_stride, frame_idx=out_frame_idx)

        mask_full_size = get_full_size_mask(out_mask_logits, bbox, image_path, out_frame_idx, H, W)
        bbox = get_bounding_box(mask_full_size)

        if bbox != None:
            print("Before bbox: ", obatin_iou(mask_full_size, get_nth_mask(seq, out_frame_idx)))
            checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            predictor2 = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
            input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))

            ones_indices = np.argwhere(mask_full_size == 1)
            zeros_indices = np.argwhere(mask_full_size == 0)

            # Select 5 points from ones and 10 from zeros
            ones_points = ones_indices[np.random.choice(len(ones_indices), 1, replace=False)]
            zeros_points = zeros_indices[np.random.choice(len(zeros_indices), 5, replace=False)]

            # Convert points into desired format
            image2_pts = np.array([[[y, x]] for x, y in np.vstack((ones_points, zeros_points))])
            image2_labels = np.array([[1]] * len(ones_points) + [[0]] * len(zeros_points))


            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor2.set_image(image)
                masks, _, _ = predictor2.predict(point_coords=image2_pts, point_labels=image2_labels)

            # import pdb; pdb.set_trace()
            print("AFter bbox: ", obatin_iou(masks[0], get_nth_mask(seq, out_frame_idx)))
        
        masks_all.append(mask_full_size)



        if vis_out:
            vis(mask_full_size, out_obj_ids, out_frame_idx, image_path, "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq)

        if use_prev_box:
            bbox = get_bounding_box(mask_full_size)

            if bbox != None:
                min_row, min_col, max_row, max_col = bbox
                
                if use_square_box:

                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=factor)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=factor)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    prev_bbox = (min_row, min_col, max_row, max_col)
                    
            elif no_mask_set_full_image:
                min_row, min_col, max_row, max_col = prev_bbox
                
                if use_square_box:
                    min_col, min_row, max_col, max_row = increase_bbox_to_square(H, W, min_col, min_row, max_col, max_row, factor=2)
                else:
                    min_col, min_row, max_col, max_row = increase_bbox_area(H, W, min_col, min_row, max_col, max_row, factor=2)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    prev_bbox = (min_row, min_col, max_row, max_col)
        
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

    for iou_i in all_ious:
        print(f"{iou_i}")

    print("----------\n")
    # print("All ious:\n")

for iou_i in all_ious:
    print(f"{iou_i}")


print(f"The mean after processing seqs is: {np.array(all_ious).mean()}")




