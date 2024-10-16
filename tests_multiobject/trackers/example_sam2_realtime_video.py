import sys
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import operator
from first_mask_book import *


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True

from sam2.build_sam import build_sam2_video_realtime_predictor #build_sam2_camera_predictor
import time


sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"



def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, last_char='', name="0"):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("Mask output shape is: ", mask_image.shape, mask.shape)

    ax.imshow(mask_image)
    plt.savefig('JM_ball3/' + name +'_' + str(ann_frame_idx) +'.png') #img_' + str(ann_frame_idx) + '_' + last_char + '.png')



def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint)
inference_state = predictor.init_state()

class SAM2Tracker(object):

    def __init__(self, image):
        self.predictor = predictor

        self.predictor.load_first_frame(inference_state, image)
        if_init = True


        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # # Let's add a positive click at (x, y) = (210, 350) to get started
        # points = np.array([[244, 87]], dtype=np.float32)
        # # for labels, `1` means positive click and `0` means negative click
        # labels = np.array([1], np.int32)
        # _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_points(
        #     inference_state=inference_state,
        #     frame_idx=ann_frame_idx,
        #     obj_id=ann_obj_id,
        #     points=points,
        #     labels=labels,
        #     # frame_idx=ann_frame_idx,
        #     # obj_id=ann_obj_id,
        #     # points=points,
        #     # labels=labels,
        # )

        # print("Shapes are: ", out_mask_logits.shape, out_mask_logits[0].shape, (out_mask_logits[0] > 0.0).cpu().numpy().shape, len(out_mask_logits))
        # print("First iteration IoUs: ", ious_output)

        image = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))

        # plt.imshow(image)
        # show_points(points, labels, plt.gca())
        # #show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, last_char='bboxes')
        mask_input = get_first_mask_book()

        # for i in range(1000):

        _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=np.array(mask_input),#out_mask_logits[0][0],
        )



        plt.clf()
        plt.cla()

        plt.imshow(image)
        # show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=11, last_char='masks')

        _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=np.array(mask_input),#out_mask_logits[0][0],
        )


        print("Second iteration IoUs: ", ious_output)
        print("Shapes are: ", out_mask_logits.shape, out_mask_logits[0].shape, (out_mask_logits[0] > 0.0).cpu().numpy().shape, len(out_mask_logits))


    def track(self, image, out_frame_idx):
        #out_obj_ids, out_mask_logits, iou_output_scores_RR_added = self.predictor.track(image)
        points, labels = None, None

        self.predictor.load_first_frame(inference_state, image, frame_idx=out_frame_idx)

        pts = [(2.955, 140), (3.059, 125), (3.06, 130), (3.133, 139), (3.223, 126)]
        pointss = [(328, 219)]

      
        # if out_frame_idx == 140:

        #     ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        #     # Let's add a positive click at (x, y) = (210, 350) to get started
        #     points = np.array([[328, 219]], dtype=np.float32)
        #     # for labels, `1` means positive click and `0` means negative click
        #     labels = np.array([1], np.int32)
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_points_or_box(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=ann_obj_id,
        #         points=points,
        #         labels=labels,
        #     )

        # if out_frame_idx == 125:
        #     ann_obj_id = 1
        #     #points = np.array([[345, 222], [331, 238]], dtype=np.float32)
            
        #     box = np.array([308, 179, 391, 275], dtype=np.float32)

        #     labels = np.array([1, 1], np.int32)
        #     _, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_points_or_box(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=ann_obj_id,
        #         box=box,
        #     )



        # plt.imshow(image)
        # show_points(points, labels, plt.gca())
        # #show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, last_char='bboxes')
        mask_input = get_first_mask_book()



        # if out_frame_idx <= 5:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=1,
        #         mask=np.array(mask_input),#out_mask_logits[0][0],
        #     )


        # if out_frame_idx == 140:
        #     ann_obj_id = 1
        #     #points = np.array([[328, 219]], dtype=np.float32)
        #     #labels = np.array([1], np.int32)
        #     box = np.array([293,176,363,258], dtype=np.float32)

        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_points_or_box(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=ann_obj_id,
        #         box=box,
        #     )
        
        # if out_frame_idx >= 50 and out_frame_idx <= 50:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=inference_state,
        #         frame_idx=out_frame_idx,
        #         obj_id=1,
        #         mask=np.array(mask_input),
        #     )   



        frame_idx, obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = self.predictor.track(inference_state, image, start_frame_idx=out_frame_idx, points=None, labels=None)



        show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=obj_ids[0], ann_frame_idx=out_frame_idx, last_char='bboxes', name=str(out_frame_idx))

        return obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits

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



video_segments = {} 


video_dir = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/ball3/color" #"book_images_large"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names = [i for i in frame_names if i[0] != '.']

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

frame_names = []

for i in range(100):
    frame_names.append('00000001.jpg')
frame_names = ['00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg', '00000001.jpg']

imagefile= frame_names[0]
print(imagefile)

image = cv2.imread(os.path.join(video_dir, imagefile), cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tracker = SAM2Tracker(os.path.join(video_dir, imagefile)) #image)

frame_index = 1

ious_scores = []
object_score_logits_scores =[]

# import pdb; pdb.set_trace();
frame_names =  frame_names[1:]
minimum =[]

plt.close("all")
for imagefile in frame_names: # frames 2...N
    image = cv2.imread(os.path.join(video_dir, imagefile), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.clf()
    plt.cla()
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_index-1])))

    out_obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = tracker.track(os.path.join(video_dir, imagefile), frame_index)  #image, frame_index

    ious_scores.append(iou_output_scores_RR_added)
    object_score_logits_scores.append(object_score_logits)
    print("Occlusion: ", object_score_logits)
    # minimum.append((np.array(object_score_logits.cpu())[0][0], frame_index))


    frame_index += 1


with open('IoU_scores/jm_occlusion_2.txt', 'w') as file:
    cnt = 0

    for i, j in zip(ious_scores, object_score_logits_scores):
        cnt+=1
        file.write(f"For the frame: {cnt}, iou is: {i} and occlusion scores are: {j} \n")


# minimum = sorted(minimum, key=operator.itemgetter(0))

# print(minimum[:5])












