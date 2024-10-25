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

SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'frisbee', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']

#SEQ= ['agility', 'frisbee', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
SEQ=['drone1']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_empty_masks', action="store_true")
    parser.add_argument('--vis_out', action="store_true")
    parser.add_argument('--memory_stride', type=int, default=1)
    parser.add_argument('--sequences')

    args = parser.parse_args()

    if args.sequences != None:
        args.sequences = args.sequences.split(",")

    return args.exclude_empty_masks, args.vis_out, args.memory_stride, args.sequences

exclude_empty_masks, vis_out, memory_stride, sequences = parse_args()

if sequences != None:
    SEQ = sequences


def run_eval(seq):

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" #/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt" #"./checkpoints/sam2_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" #"sam2_hiera_l.yaml" #"configs/sam2/sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')

    video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + seq + "/color"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # print(frame_names)
    frame_names = [i for i in frame_names if i[0] != '.']


    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # inference_state = predictor.init_state(video_path=video_dir)

    inference_state = predictor.init_state()

    mask_first_frame = get_nth_mask(seq, 0)

    # min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame)


    predictor.load_first_frame(inference_state, os.path.join(video_dir, '00000001.jpg'))#, bbox=(min_row, min_col, max_row, max_col))


    #mask_first_frame = mask_first_frame[min_col:max_col, min_row:max_row]

    _, _, out_mask_logits = predictor.add_new_mask(
    	inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=np.array(mask_first_frame),
    )

 
    vis(out_mask_logits, [1], 0, os.path.join(video_dir, frame_names[0]), "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq)

    masks_all = []

    #for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, exclude_empty_masks=exclude_empty_masks, memory_stride=memory_stride):
    # vis(out_mask_logits, out_obj_ids, out_frame_idx, os.path.join(video_dir, frame_names[0]), "output_sam2.1")

    for out_frame_idx in tqdm(range(1, len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[out_frame_idx])
        
        predictor.load_first_frame(inference_state, image_path, frame_idx=out_frame_idx)

        out_frame_idx, out_obj_ids, out_mask_logits = predictor.track(inference_state, exclude_empty_masks=exclude_empty_masks, memory_stride=memory_stride, frame_idx=out_frame_idx)

        masks_all.append((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])

        # cnt+=1
        vis(out_mask_logits, out_obj_ids, out_frame_idx, os.path.join(video_dir, frame_names[out_frame_idx]), "/datagrid/personal/rozumrus/BP_dg/sam2.1_output/" + seq)

    return [mask_first_frame] + masks_all

       

def vis(out_mask_logits, out_obj_ids, ann_frame_idx, image, to_save_path):
    image = Image.open(image)

    # mask_first_frame = get_nth_mask(seq, 0)

    # min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame)


    # image = image.crop((min_row, min_col, max_row, max_col))


    plt.clf()
    plt.cla()

    plt.imshow(image)

    # import pdb; pdb.set_trace()

    # H, W = mask_first_frame.shape

    # filled_mask = np.zeros((H, W))
    # filled_mask[min_col:max_col, min_row:max_row] = (out_mask_logits[0] > 0).cpu().numpy()






    show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)



all_ious = []

for seq in SEQ:
    masks_all = run_eval(seq)

    iou_curr = get_iou(seq, masks_all)

    all_ious.append(iou_curr)

    print(f"IoU fpr {seq} is: {iou_curr}\n")

    print("----------\n")
    print("All ious:\n")

    for iou_i in all_ious:
        print(f"{iou_i}")








