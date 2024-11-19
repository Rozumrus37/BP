import sys
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from compute_iou import *
from utilities_eval import *
from tqdm import tqdm
import argparse

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor
import time

parser = argparse.ArgumentParser()
parser.add_argument('--sam2_original', action="store_true")
parser.add_argument('--sam2_RR', action="store_true")
parser.add_argument('--exclude_empty_masks', action="store_true")
parser.add_argument('--memory_stride', type=int, default=1)
parser.add_argument('--factor', type=float, default=1)


args = parser.parse_args()

# sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
# model_cfg = "sam2_hiera_s.yaml"
sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/sam2.1/sam2/checkpoints/sam2.1_hiera_large.pt" #/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt" #"./checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

class SAM2Tracker(object):

    def __init__(self, image, mask, cnt):
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) 

        self.inference_state = self.predictor.init_state()
        self.memory_stride = args.memory_stride
        self.factor = args.factor
        self.prev_mask_increase_when_empty = True
        self.exclude_empty_masks = args.exclude_empty_masks
        self.use_RR_sam2 = args.sam2_RR
        self.prev_bbox = None
        self.prev_mask = None 

        iimage = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
        iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2RGB)

        self.H, self.W, _ = iimage.shape

        bbox = None 

        if self.use_RR_sam2:
            min_row, min_col, max_row, max_col = get_bounding_box(mask)
            min_col, min_row, max_col, max_row = increase_bbox_area(self.H, self.W, min_col, min_row, max_col, max_row, factor=self.factor)

            bbox = (min_row, min_col, max_row, max_col)
            mask = mask[min_col:max_col, min_row:max_row]

        self.predictor.load_first_frame(self.inference_state, image, bbox=bbox)
        _, _, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=1,
            mask=np.array(mask),
        )

        self.prev_mask = mask

        if self.use_RR_sam2:
            self.prev_bbox = (min_row, min_col, max_row, max_col) 

    def track(self, image, c):
        bbox = self.prev_bbox
        self.predictor.load_first_frame(self.inference_state, image, frame_idx=c, bbox=bbox)
        output = None 

        if self.use_RR_sam2:
            out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = self.predictor.track(self.inference_state, 
                exclude_empty_masks=self.exclude_empty_masks, memory_stride=self.memory_stride, frame_idx=c)

            mask_full_size = get_full_size_mask(out_mask_logits, bbox, self.H, self.W)

            bbox = get_bounding_box(mask_full_size)

            if bbox != None:
                min_row, min_col, max_row, max_col = bbox
                
                min_col, min_row, max_col, max_row = increase_bbox_area(self.H, self.W, min_col, min_row, max_col, max_row, factor=self.factor)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    self.prev_bbox = (min_row, min_col, max_row, max_col)
                    
            elif self.prev_mask_increase_when_empty:
                min_row, min_col, max_row, max_col = self.prev_bbox
                
                min_col, min_row, max_col, max_row = increase_bbox_area(self.H, self.W, min_col, min_row, max_col, max_row, factor=2)

                if min_row-max_row != 0 and min_col-max_col != 0:
                    self.prev_bbox = (min_row, min_col, max_row, max_col)

            output = np.array(mask_full_size).astype(np.uint8) 
        else:
            out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = self.predictor.track(self.inference_state, frame_idx=c, prev_mask=self.prev_mask)
            output = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]             


        self.prev_mask = output   

        return output
        

    def vis_segm(self, image, mask, output_file):
        non_zero_indices = np.nonzero(mask)
        coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
 
        image_with_dots = image.copy()
        
        for (y, x) in coordinates:
            cv2.circle(image_with_dots, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
        
        cv2.imwrite(output_file, image_with_dots)
        print(f"Image with black dots saved to {output_file}")


if args.sam2_original:
    path_pred = "/datagrid/personal/rozumrus/BP_dg/sav_test/pred_sam2"
elif args.sam2_RR and args.memory_stride == 7:
    path_pred = "/datagrid/personal/rozumrus/BP_dg/sav_test/pred_sam2_RR"
elif args.sam2_RR and args.memory_stride == 1:
    path_pred = "/datagrid/personal/rozumrus/BP_dg/sav_test/pred_sam2_RR_mem_stride1"
else:
    path_pred = "/datagrid/personal/rozumrus/BP_dg/sav_test/temp"

path_imgs = "/datagrid/personal/rozumrus/BP_dg/sav_test/sav_val/JPEGImages_24fps"
path_gt = "/datagrid/personal/rozumrus/BP_dg/sav_test/sav_val/Annotations_6fps"

global_cnt = 0

for video_name in sorted(os.listdir(path_gt)):
    if not os.path.isdir(os.path.join(path_gt, video_name)):
        continue

    global_cnt += 1

    # if global_cnt <= 151 and args.sam2_RR:
    #     continue 

    # if global_cnt > 3:
    #     break
  


    print(f"Processed video is {video_name}")
    trackers = []

    cnt, c = 0, 1

    video_dir_gt = os.path.join(path_gt, video_name)

    for obj_id in sorted(os.listdir(video_dir_gt)):
        if not os.path.isdir(os.path.join(path_gt, video_name)):
            continue

        gt_mask_path = os.path.join(video_dir_gt, obj_id, "00000.png")
        imagefile = os.path.join(path_imgs, video_name, "00000.jpg")



        gt_mask = Image.open(gt_mask_path).convert("1")  
        gt_mask = np.array(gt_mask)
        gt_mask = gt_mask.astype(int)

        trackers.append(SAM2Tracker(imagefile, gt_mask, cnt))
        cnt+=1

    video_dir = os.path.join(path_imgs, video_name)

    pred_current_video_path = os.path.join(path_pred, video_name)

    if not os.path.exists(pred_current_video_path):
        os.makedirs(pred_current_video_path)

    counter = 0
    for tracker in trackers:
        obj_id_path = os.path.join(pred_current_video_path, "00" + str(counter))

        if not os.path.exists(obj_id_path):
            os.makedirs(obj_id_path)

        counter+=1

    c = 1
    for  frame in tqdm(sorted(os.listdir(video_dir))):
        if frame[0] == '.' or frame=="00000.jpg":
            continue 

        out = []
 
        for tracker in trackers:
            # print(c, video_dir, frame)
            # import pdb; pdb.set_trace()

            out.append(tracker.track(os.path.join(video_dir, frame), c))

        counter = 0

        for img_np in out:
            img_np = img_np * 255
            image = Image.fromarray(img_np.astype(np.uint8), 'L')  # 'L' mode for grayscale (black and white)

            image.save(os.path.join(pred_current_video_path, "00" + str(counter),  os.path.splitext(frame)[0] + ".png"))
            counter+=1

        c+=1

    del trackers
    torch.cuda.empty_cache()

           



# while True:
#     imagefile = handle.frame()
#     if not imagefile:
#         break

#     out = []
#     for tracker in trackers:
#         out.append(tracker.track(imagefile, c))

#     c += 1
#     handle.report(out)










