
# ml PyTorch/2.3.0-foss-2023a-CUDA-12.3.0
# ml OpenCV/4.9.0-foss-2023a-CUDA-12.3.0-contrib
# ml  torchvision/0.18.0-foss-2023a-CUDA-12.3.0 //torchvision/0.18.0-foss-2023b-CUDA-12.4.0
# ml Hydra/1.3.2-GCCcore-12.3.0  


# from PIL import Imag




import vot_utils
import sys
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time


sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"


"""""" 

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


class NCCTracker2(object):

    def __init__(self, image, mask):
        predictor.load_first_frame(image)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array([[600, 400], [600, 300], [600, 350]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 1, 1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    def track(self, image):
        out_obj_ids, out_mask_logits = predictor.track(image)

        return out_mask_logits

    def _rect_from_mask(self, mask):
        '''
        create an axis-aligned rectangle from a given binary mask
        mask in created as a minimal rectangle containing all non-zero pixels
        '''
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    def _mask_from_rect(self, rect, output_size):
        '''
        create a binary mask from a given rectangle
        rect: axis-aligned rectangle [x0, y0, width, height]
        output_sz: size of the output [width, height]
        '''
        mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
        x0 = max(int(round(rect[0])), 0)
        y0 = max(int(round(rect[1])), 0)
        x1 = min(int(round(rect[0] + rect[2])), output_size[0])
        y1 = min(int(round(rect[1] + rect[3])), output_size[1])
        mask[y0:y1, x0:x1] = 1
        return mask


handle = vot_utils.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)

trackers = [NCCTracker2(image, object) for object in objects]

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    handle.report([tracker.track(image) for tracker in trackers])
