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

from hq_sam.sam_hq.segment_anything import sam_model_registry, SamPredictor

torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False

from sam2.build_sam import build_sam2_video_realtime_predictor

USE_HQ = False

if USE_HQ:
    sam_checkpoint = "hq_sam/sam_hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

class Sam2RealtimeTracker(object):

    def __init__(self, image_path, VIDEO_NAME, alfa=0, num_obj_ptrs=16, memory_temporal_stride_for_eval_r=1, use_large_SAM2=True, exclude_empty_masks=True, no_memory=False, vis_out=False, to_save_path=None):

        if use_large_SAM2:
            model_cfg = "sam2_hiera_l.yaml"
            sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt"
        else:
            model_cfg = "sam2_hiera_s.yaml"
            sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
        
        predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint)
        self.inference_state = predictor.init_state()
        self.predictor = predictor
        self.video_name = VIDEO_NAME
        self.alfa = alfa
        self.predictor.load_first_frame(self.inference_state, image_path)
        self.best_masklets_prev = []
        self.to_save_path = to_save_path
        self.num_obj_ptrs = num_obj_ptrs
        self.exclude_empty_masks = exclude_empty_masks
        self.no_memory = no_memory
        self.vis_out = vis_out
        self.memory_temporal_stride_for_eval_r = memory_temporal_stride_for_eval_r

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
        self.mask_first_frame = get_nth_mask(VIDEO_NAME, 0)

        _, out_obj_ids, out_mask_logits, ious_output, low_res_masks = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=np.array(self.mask_first_frame),#
            num_of_obj_ptrs_in_sam2=num_obj_ptrs,
            memory_temporal_stride_for_eval_r=memory_temporal_stride_for_eval_r,
        )

        if vis_out:
            self.vis(out_mask_logits, out_obj_ids, str(0), image_path, self.to_save_path)

        self.prev_out_mask_logits = [out_mask_logits]
        self.ious = []

    def track(self, image_path, out_frame_idx):
        self.predictor.load_first_frame(self.inference_state, image_path, frame_idx=out_frame_idx)

        prev_mask = self.prev_out_mask_logits[-1]

        (frame_idx,
        obj_ids, 
        out_mask_logits, 
        iou_output_scores_RR_added, 
        object_score_logits, 
        best_masklets,
        worst_res_masks, 
        second_best_res_masks, 
        low_res_output_mask,
        IoU_prev_curr) = self.predictor.track(
            self.inference_state, 
            image_path, 
            start_frame_idx=out_frame_idx, 
            points=None, labels=None, 
            best_masklets=self.best_masklets_prev, 
            prev_mask=prev_mask, 
            alfa=self.alfa, 
            exclude_empty_masks=self.exclude_empty_masks,
            no_memory_sam2=self.no_memory,
            num_of_obj_ptrs_in_sam2=self.num_obj_ptrs,
            memory_temporal_stride_for_eval_r=self.memory_temporal_stride_for_eval_r)

        self.best_masklets_prev = best_masklets
        self.prev_out_mask_logits.append(out_mask_logits)

        if USE_HQ:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = get_bounding_box((out_mask_logits[0] > 0).cpu().numpy()[0]) 
            if output != None:
                a, b, c, d = output
                input_box = np.array([a, b, c, d])
                input_point, input_label = None, None
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=low_res_output_mask[0],
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=False,
                )
                self.vis_sam_hq(masks, obj_ids, out_frame_idx, self.video_name, image_path)
        else:
            masks = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]

            if self.vis_out:
                self.vis(out_mask_logits, obj_ids, out_frame_idx, image_path, self.to_save_path)
                # self.vis(second_best_res_masks, obj_ids, out_frame_idx, self.video_name + "_second_best", image_path)
                # self.vis(worst_res_masks, obj_ids, out_frame_idx, self.video_name + "_third_best", image_path)

        self.ious.append(IoU_prev_curr)

        return obj_ids, masks, iou_output_scores_RR_added, object_score_logits


    def vis_sam_hq(self, out_mask_logits, out_obj_ids, ann_frame_idx, output_dir, image):
        image = Image.open(image)

        plt.clf()
        plt.cla()

        plt.imshow(image)

        show_mask(out_mask_logits[0], plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, output_dir=output_dir)

    def vis(self, out_mask_logits, out_obj_ids, ann_frame_idx, image, to_save_path):
        image = Image.open(image)

        plt.clf()
        plt.cla()

        plt.imshow(image)
        show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)
