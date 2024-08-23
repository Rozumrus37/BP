
import vot_utils
import sys
import cv2
import numpy as np
import os
# import torch
import matplotlib.pyplot as plt
# from sam2_tracker_vot import NccTracker2
# from PIL import Image

#  ml PyTorch/1.12.0-foss-2021a-CUDA-11.3.1
#  ml torchvision/0.13.0-foss-2021a-CUDA-11.3.1

class NCCTracker(object):

    def __init__(self, image, mask):
        region = self._rect_from_mask(mask)
        region = vot_utils.Rectangle(region[0], region[1], region[2], region[3])
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot_utils.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        #return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1])
        return self._mask_from_rect([left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]], (image.shape[1], image.shape[0]))

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




# import vot_utils
# import sys
# import cv2
# import numpy as np
# import os
# import torch
# import matplotlib.pyplot as plt
# import imageio
# from PIL import Image

# # use bfloat16 for the entire notebook
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# from sam2.build_sam import build_sam2_camera_predictor
# import time


# sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
# model_cfg = "sam2_hiera_s.yaml"


# """""" 

# predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


# class NCCTracker(object):

#     def __init__(self, image, mask):
#         predictor.load_first_frame(image)
#         if_init = True

#         ann_frame_idx = 0  # the frame index we interact with
#         ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
#         # Let's add a positive click at (x, y) = (210, 350) to get started
#         points = np.array([[600, 400], [600, 300], [600, 350]], dtype=np.float32)
#         # for labels, `1` means positive click and `0` means negative click
#         labels = np.array([1, 1, 1], np.int32)
#         _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#             frame_idx=ann_frame_idx,
#             obj_id=ann_obj_id,
#             points=points,
#             labels=labels,
#         )

#     def track(self, image):
#         out_obj_ids, out_mask_logits = predictor.track(image)

#         return out_mask_logits

#     def _rect_from_mask(self, mask):
#         '''
#         create an axis-aligned rectangle from a given binary mask
#         mask in created as a minimal rectangle containing all non-zero pixels
#         '''
#         x_ = np.sum(mask, axis=0)
#         y_ = np.sum(mask, axis=1)
#         x0 = np.min(np.nonzero(x_))
#         x1 = np.max(np.nonzero(x_))
#         y0 = np.min(np.nonzero(y_))
#         y1 = np.max(np.nonzero(y_))
#         return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

#     def _mask_from_rect(self, rect, output_size):
#         '''
#         create a binary mask from a given rectangle
#         rect: axis-aligned rectangle [x0, y0, width, height]
#         output_sz: size of the output [width, height]
#         '''
#         mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
#         x0 = max(int(round(rect[0])), 0)
#         y0 = max(int(round(rect[1])), 0)
#         x1 = min(int(round(rect[0] + rect[2])), output_size[0])
#         y1 = min(int(round(rect[1] + rect[3])), output_size[1])
#         mask[y0:y1, x0:x1] = 1
#         return mask


handle = vot_utils.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)

trackers = [NCCTracker(image, object) for object in objects]

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    handle.report([tracker.track(image) for tracker in trackers])




# import vot_utils
# import cv2

# # An example of how to use the toolkit in Python. This is the simplest possible example
# # which only reports the initialization region for all frames. It only supports single
# # channel (RGB) frames.

# handle = vot_utils.VOT("mask", multiobject=True)
# objects = handle.objects()

# imagefile = handle.frame()

# image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)

# while True:
#     imagefile = handle.frame()
#     if not imagefile:
#         # Terminate if no new frame was received.
#         break
#     image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
#     handle.report(objects)
