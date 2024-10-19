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
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True

# from sam2.build_sam import build_sam2_camera_predictor
from sam2.build_sam import build_sam2_video_realtime_predictor

import time


sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"


class SAM2Tracker(object):

    def __init__(self, image, mask, cnt):
        

        self.predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint) # build_sam2_camera_predictor

        self.inference_state = self.predictor.init_state()


        #self.predictor.load_first_frame(image)
        
        self.predictor.load_first_frame(self.inference_state, image)
        
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started
        iimage = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
        iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2RGB)


        mask = self.pad_mask_to_image_size(mask, (iimage.shape[0], iimage.shape[1]))

        _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=mask
        )

        #self.vis_segm(image, (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], "/home.stud/rozumrus/BP/tests_multiobject/out_sam2_25maskletN/" + str(cnt) + "init.png")

    def pad_mask_to_image_size(self, mask, image_size):
        # Get the dimensions of the mask and the desired image size
        mask_rows, mask_cols = mask.shape
        image_rows, image_cols = image_size
        
        # Calculate the amount of padding needed for rows and columns
        pad_rows = image_rows - mask_rows
        pad_cols = image_cols - mask_cols

        # Pad the mask with zeros to the right and bottom
        padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return padded_mask

    def track(self, image,c):
        #out_obj_ids, out_mask_logits = self.predictor.track(image)
        self.predictor.load_first_frame(self.inference_state, image, frame_idx=c)

        # if c == 1:
        #     # ann_frame_idx = 1  # the frame index we interact with
        #     # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        #     # # Let's add a positive click at (x, y) = (210, 350) to get started
        #     # points = np.array([[209, 109]], dtype=np.float32)
        #     # # for labels, `1` means positive click and `0` means negative click
        #     # labels = np.array([1], np.int32)

        #     ann_frame_idx = 1  # the frame index we interact with
        #     ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        #     # Let's add a positive click at (x, y) = (210, 350) to get started
        #     points = np.array([[209, 109]], dtype=np.float32)
        #     # for labels, `1` means positive click and `0` means negative click
        #     labels = np.array([1], np.int32)
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_points_or_box(
        #         inference_state=self.inference_state,
        #         frame_idx=ann_frame_idx,
        #         obj_id=ann_obj_id,
        #         points=points,
        #         labels=labels,
        #     )



        frame_idx, obj_ids, out_mask_logits, iou_output_scores_RR_added = self.predictor.track(self.inference_state, image, start_frame_idx=c)

        #self.vis_segm(image, (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], "/home.stud/rozumrus/BP/tests_multiobject/out_sam2_25maskletN/" + str(c) + "_sam2_out.png")

        return (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
        
        #return (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]

    def vis_segm(self, image, mask, output_file):
        non_zero_indices = np.nonzero(mask)
        coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
        # print("LIST", len(coordinates), coordinates)
        
        # Make a copy of the image to draw on
        image_with_dots = image.copy()
        
        # Draw black dots on the image at each coordinate
        for (y, x) in coordinates:
            cv2.circle(image_with_dots, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
        
        # Save the resulting image
        cv2.imwrite(output_file, image_with_dots)
        print(f"Image with black dots saved to {output_file}")


handle = vot_utils.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

# image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

trackers = []

cnt, c = 0, 1

for object in objects:
    trackers.append(SAM2Tracker(imagefile, object, cnt))
    cnt += 1

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    # image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out = []
    for tracker in trackers:
        # c += 1
        out.append(tracker.track(imagefile, c))


    c += 1
    handle.report(out)


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
# torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = False #True
#     torch.backends.cudnn.allow_tf32 = False #True

# # from sam2.build_sam import build_sam2_camera_predictor
# from sam2.build_sam import build_sam2_video_realtime_predictor

# import time


# sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, last_char='', name="0"):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         cmap = plt.get_cmap("tab10")
#         cmap_idx = 0 if obj_id is None else obj_id
#         color = np.array([*cmap(cmap_idx)[:3], 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     print("Mask output shape is: ", mask_image.shape, mask.shape)

#     ax.imshow(mask_image)
#     plt.savefig('/home.stud/rozumrus/BP/tests_multiobject/trackers/vot24_all/' + name + '_' + last_char + '.png') #img_' + str(ann_frame_idx) + '_' + last_char + '.png')



# def show_points(coords, labels, ax, marker_size=200):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



# class SAM2Tracker(object):

#     def __init__(self, image, masks):#, cnt):
        

#         self.predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint) 

#         self.inference_state = self.predictor.init_state()
        
#         self.predictor.load_first_frame(self.inference_state, image)
#         self.prev_logit = -1000
            
#         cnt = 0
#         iimage = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
#         iimage = cv2.cvtColor(iimage, cv2.COLOR_BGR2RGB)
#         if_init = True

#         for object in objects:

#             ann_frame_idx = 0  
#             ann_obj_id = cnt #2  
#             mask = object

#             mask = self.pad_mask_to_image_size(mask, (iimage.shape[0], iimage.shape[1]))


            
#             _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
#                 inference_state=self.inference_state,
#                 frame_idx=ann_frame_idx,
#                 obj_id=ann_obj_id,
#                 mask=mask
#             )

#             cnt+=1

#         # with open('/home.stud/rozumrus/BP/tests_multiobject/trackers/first_ball3_mask.txt', 'a') as file:

#         #     for i in range(iimage.shape[0]):
#         #         file.write(f"[")
#         #         for j in range(iimage.shape[1]):
#         #             file.write(f"{mask[i, j]}")
#         #             if j != iimage.shape[1] - 1:
#         #                 file.write(f",")
#         #         file.write(f"],\n")
        

#         # with open('/home.stud/rozumrus/BP/tests_multiobject/trackers/vot2024_all.txt', 'a') as file:
#         #     file.write(f"For the frame: 0_{cnt}, iou is: {ious_output} INIT\n")

#         #self.vis_segm(image, (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], "/home.stud/rozumrus/BP/tests_multiobject/out_sam2_25maskletN/" + str(cnt) + "init.png")

#     def pad_mask_to_image_size(self, mask, image_size):
#         # Get the dimensions of the mask and the desired image size
#         mask_rows, mask_cols = mask.shape
#         image_rows, image_cols = image_size
        
#         # Calculate the amount of padding needed for rows and columns
#         pad_rows = image_rows - mask_rows
#         pad_cols = image_cols - mask_cols

#         # Pad the mask with zeros to the right and bottom
#         padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

#         return padded_mask

#     def track(self, image, out_frame_idx):
#         #out_obj_ids, out_mask_logits = self.predictor.track(image)
#         self.predictor.load_first_frame(self.inference_state, image, frame_idx=out_frame_idx)

#         frame_idx, obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = self.predictor.track(self.inference_state, image, start_frame_idx=out_frame_idx)


#         # with open('/home.stud/rozumrus/BP/tests_multiobject/trackers/vot2024_all1prompt.txt', 'a') as file:
#         #     file.write(f"For the frame: {frame_idx}, iou is: {iou_output_scores_RR_added} and occlusion scores are: {object_score_logits}  \n")
#             # file.write(f"Shapes: {(out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0].shape}")

#         # plt.clf()
#         # plt.cla()
#         # plt.imshow(Image.open(image))

#         # show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=obj_ids[0], ann_frame_idx=out_frame_idx, last_char=str(obj_id), name=str(out_frame_idx))

#         #self.vis_segm(image, (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], "/home.stud/rozumrus/BP/tests_multiobject/vot_output_2_1modified/" + str(c) + "_sam2_out.png")

#         return {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)[0] for i, out_obj_id in enumerate(obj_ids)} 

#         #(out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
        
#         #return (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]

#     def vis_segm(self, image, mask, output_file):
       
#         non_zero_indices = np.nonzero(mask)
#         coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
#         # print("LIST", len(coordinates), coordinates)
        
#         # Make a copy of the image to draw on
#         image_with_dots = image.copy()
        
#         # Draw black dots on the image at each coordinate
#         for (y, x) in coordinates:
#             cv2.circle(image_with_dots, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
        
#         # Save the resulting image
#         cv2.imwrite(output_file, image_with_dots)
#         print(f"Image with black dots saved to {output_file}")


# handle = vot_utils.VOT("mask", multiobject=True)
# objects = handle.objects()

# imagefile = handle.frame()

# # tracker = SAM2Tracker(imagefile, objects, cnt)
# # image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# tracker = SAM2Tracker(imagefile, objects)

# cnt, c = 0, 1

# while True:
#     imagefile = handle.frame()
#     if not imagefile:
#         break

#     # image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     out = tracker.track(imagefile, c)
#     out = [out[i] for i in out]


 
#     c += 1
#     handle.report(out)



# # trackers = []

# # cnt, c = 0, 1

# # for object in objects:
# #     trackers.append(SAM2Tracker(imagefile, object, cnt))
# #     cnt += 1

# # while True:
# #     imagefile = handle.frame()
# #     if not imagefile:
# #         break

# #     # image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
# #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #     out = []
# #     obj_id = 0

# #     for tracker in trackers:
# #         out.append(tracker.track(imagefile, c, obj_id))
# #         obj_id += 1

# #     c += 1
# #     handle.report(out)


# # handle = vot_utils.VOT("mask")
# # selection = handle.region()

# # imagefile = handle.frame()
# # if not imagefile:
# #     sys.exit(0)

# # tracker =SAM2Tracker(imagefile, selection, 0)

# # c = 1

# # while True:
# #     imagefile = handle.frame()
# #     if not imagefile:
# #         break

# #     m = tracker.track(imagefile, c)
# #     c+=1
# #     handle.report(m, 1)
