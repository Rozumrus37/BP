import vot_utils
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

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor
import time


# sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
# model_cfg = "sam2_hiera_s.yaml"
sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/sam2.1/sam2/checkpoints/sam2.1_hiera_large.pt" #/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt" #"./checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


class SAM2Tracker(object):

    def __init__(self, image, mask, cnt):
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) 

        self.inference_state = self.predictor.init_state()
        self.no_mask_set_full_image = True
        self.memory_stride = 7
        self.factor = 100
        self.prev_mask_increase_when_empty = True
        self.exclude_empty_masks = True
        self.use_RR_sam2 = False
        self.prev_bbox = None

        iimage = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
        iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2RGB)

        self.H, self.W, _ = iimage.shape

        mask = self.pad_mask_to_image_size(mask, (iimage.shape[0], iimage.shape[1]))

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

        if self.use_RR_sam2:
            self.prev_bbox = (min_row, min_col, max_row, max_col) 

    def pad_mask_to_image_size(self, mask, image_size):
        mask_rows, mask_cols = mask.shape
        image_rows, image_cols = image_size
        
        pad_rows = image_rows - mask_rows
        pad_cols = image_cols - mask_cols

        padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return padded_mask

    def track(self, image, c):
        bbox = self.prev_bbox
        self.predictor.load_first_frame(self.inference_state, image, frame_idx=c, bbox=bbox)
        output = None 

        if self.use_RR_sam2:
            out_frame_idx, out_obj_ids, out_mask_logits, _, _ = self.predictor.track(self.inference_state, 
                exclude_empty_masks=self.exclude_empty_masks, memory_stride=self.memory_stride, frame_idx=c)

            mask_full_size = get_full_size_mask(out_mask_logits, bbox, image, self.H, self.W)

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
            out_frame_idx, out_obj_ids, out_mask_logits, _, _ = self.predictor.track(self.inference_state, frame_idx=c)
            output = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]             

        return output
        

    def vis_segm(self, image, mask, output_file):
        non_zero_indices = np.nonzero(mask)
        coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
 
        image_with_dots = image.copy()
        
        for (y, x) in coordinates:
            cv2.circle(image_with_dots, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
        
        cv2.imwrite(output_file, image_with_dots)
        print(f"Image with black dots saved to {output_file}")


handle = vot_utils.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

trackers = []

cnt, c = 0, 1

for object in objects:
    trackers.append(SAM2Tracker(imagefile, object, cnt))
    cnt += 1

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    out = []
    for tracker in trackers:
        out.append(tracker.track(imagefile, c))

    c += 1
    handle.report(out)

