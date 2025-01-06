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


""" SAM2 parallel tracking using cropped window;
The commented parts of the code correspond to tracking by cropping controlled by the enlarging factor """

class SAM2Tracker(object):

    def __init__(self, image, mask_init, cnt):
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) 

        self.predictors = []
        self.prev_bboxs = [] 
        self.masks_all = []
        self.Fs = [2, 1]

        self.inference_state = self.predictor.init_state()
        self.memory_stride = 5
        self.prev_mask_increase_when_empty = False
        self.factor = 9
        self.exclude_empty_masks = True
        self.use_RR_sam2 = True
        self.prev_bbox = None
        self.no_mask_set_whole_image = False
        self.min_box_factor = 512

        self.thr_IoU_BB1_BBsm = 0.5
        self.thr_Amask_to_Abb = 0.1
            
        iimage = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
        iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2RGB)

        self.H, self.W, _ = iimage.shape

    
        for F in self.Fs:
            mask = self.pad_mask_to_image_size(mask_init, (iimage.shape[0], iimage.shape[1]))

            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
            inference_state = predictor.init_state()

            min_row, min_col, max_row, max_col = get_bounding_box(mask)
            min_row, min_col, max_row, max_col = increase_bbox_area_for_parallel(self.H, self.W, min_row, min_col, max_row, max_col, factor=F)

            prev_bbox = (min_row, min_col, max_row, max_col)
            self.prev_bboxs.append(prev_bbox)

            mask = mask[min_row:max_row, min_col:max_col]

            predictor.load_first_frame(inference_state, image, bbox=prev_bbox)

            _, _, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=mask,
            )

            self.predictors.append((predictor,inference_state))


        # self.prev_mask = mask 
        # bbox = None 

        # if self.use_RR_sam2:
        #     min_row, min_col, max_row, max_col = get_bounding_box(mask)
        #     min_row, min_col, max_row, max_col = increase_bbox_area(self.H, self.W, min_row, min_col, max_row, max_col, min_box_factor=self.min_box_factor, factor=self.factor)

        #     bbox = (min_row, min_col, max_row, max_col)
        #     mask = mask[min_row:max_row, min_col:max_col]

        # self.predictor.load_first_frame(self.inference_state, image, bbox=bbox)
        # _, _, out_mask_logits = self.predictor.add_new_mask(
        #     inference_state=self.inference_state,
        #     frame_idx=0,
        #     obj_id=1,
        #     mask=np.array(mask),
        # )

        # if self.use_RR_sam2:
        #     self.prev_bbox = (min_row, min_col, max_row, max_col) 

    def pad_mask_to_image_size(self, mask, image_size):
        mask_rows, mask_cols = mask.shape
        image_rows, image_cols = image_size
        
        pad_rows = image_rows - mask_rows
        pad_cols = image_cols - mask_cols

        padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return padded_mask

    def track(self, image, c):
        bbox = self.prev_bbox
        # self.predictor.load_first_frame(self.inference_state, image, frame_idx=c, bbox=bbox)
        output = None 
        masks_i = []

        for i in range(len(self.Fs)):
            predictor, inference_state = self.predictors[i][0], self.predictors[i][1]

            predictor.load_first_frame(self.predictors[i][1], image, frame_idx=c, bbox=self.prev_bboxs[i])

            out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = predictor.track(
                inference_state, 
                exclude_empty_masks=self.exclude_empty_masks, 
                memory_stride=self.memory_stride, 
                frame_idx=c, 
                video_H=self.H,
                video_W=self.W)

            mask_full_size = get_full_size_mask(out_mask_logits, self.prev_bboxs[i], self.H, self.W)
            masks_i.append(mask_full_size)

            temp_bbox = get_bounding_box(mask_full_size)

            if temp_bbox != None:
                min_row, min_col, max_row, max_col = temp_bbox
                min_row, min_col, max_row, max_col = increase_bbox_area_for_parallel(self.H, self.W, min_row, min_col, max_row, max_col, factor=self.Fs[i])
                self.prev_bboxs[i] = (min_row, min_col, max_row, max_col)       

        best_score, best_mask = -1, masks_i[len(self.Fs)-1]
        mask_ori = masks_i[len(self.Fs)-1] # SAM2 original mask output

        for i in range(len(self.Fs)):
            IoU_mask_i_with_mask_ori = obatin_iou(masks_i[i], mask_ori) 
            mask_curr_bbox = get_bounding_box(masks_i[i]) 
            if mask_curr_bbox != None:
                H_curr_bbox, W_curr_bbox = mask_curr_bbox[0] - mask_curr_bbox[2], mask_curr_bbox[1] - mask_curr_bbox[3]
            else:
                H_curr_bbox, W_curr_bbox = 0, 0

            area_mask_to_area_whole_BB = H_curr_bbox * W_curr_bbox / ((self.H * self.W) / (self.Fs[i]**2))

            if (IoU_mask_i_with_mask_ori > self.thr_IoU_BB1_BBsm and (area_mask_to_area_whole_BB  <= self.thr_Amask_to_Abb)) or np.sum(mask_ori) == 0:
                # print(np.sum(masks_i[i]) / (H*W/(Fs[i]**2)) * 100)
                best_mask = masks_i[i]
                # best_score = iou_curr
                break
        
        
        # if self.use_RR_sam2:
        #     out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = self.predictor.track(self.inference_state, 
        #         exclude_empty_masks=self.exclude_empty_masks, memory_stride=self.memory_stride, frame_idx=c, prev_mask=None)

        #     mask_full_size = get_full_size_mask(out_mask_logits, bbox, self.H, self.W)

        #     bbox = get_bounding_box(mask_full_size)

        #     if bbox != None:
        #         min_row, min_col, max_row, max_col = bbox
                
        #         min_row, min_col, max_row, max_col = increase_bbox_area(self.H, self.W, min_row, min_col, max_row, max_col, min_box_factor=self.min_box_factor, factor=self.factor)

        #         if min_row-max_row != 0 and min_col-max_col != 0:
        #             self.prev_bbox = (min_row, min_col, max_row, max_col)
                    
        #     elif self.prev_mask_increase_when_empty:
        #         min_row, min_col, max_row, max_col = self.prev_bbox
                
        #         min_row, min_col, max_row, max_col = increase_bbox_area(self.H, self.W, min_row, min_col, max_row, max_col, min_box_factor=self.min_box_factor, factor=2)

        #         if min_row-max_row != 0 and min_col-max_col != 0:
        #             self.prev_bbox = (min_row, min_col, max_row, max_col)

        #     elif self.no_mask_set_whole_image:
        #         self.prev_bbox = None

        #     output = np.array(mask_full_size).astype(np.uint8) 
        # else:
        #     out_frame_idx, out_obj_ids, out_mask_logits, _, _, _ = self.predictor.track(self.inference_state, frame_idx=c, prev_mask=None,  
        #         exclude_empty_masks=self.exclude_empty_masks, memory_stride=self.memory_stride)
        #     output = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]  

        # self.prev_mask = output           

        return np.array(best_mask).astype(np.uint8) #output
        

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

