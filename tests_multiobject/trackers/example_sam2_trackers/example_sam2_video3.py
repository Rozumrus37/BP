import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
from compute_iou import *

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True

from sam2.build_sam import build_sam2_video_realtime_predictor


def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, output_dir="temp"):
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
    plt.savefig(output_dir + str(ann_frame_idx) + '.png')



class SAM2Tracker(object):

    def __init__(self, image, VIDEO_NAME):
        model_cfg = "sam2_hiera_s.yaml"
        sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
        
        predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint)
        self.inference_state = predictor.init_state()

        self.predictor = predictor


        self.video_name = VIDEO_NAME

        self.predictor.load_first_frame(self.inference_state, image)
        self.best_masklets_prev = []
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
        self.mask_first_frame = get_nth_mask(VIDEO_NAME, 0)

        _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=np.array(self.mask_first_frame),#out_mask_logits[0][0],
        )

        self.prev_out_mask_logits = out_mask_logits

        # image = Image.open(image)

        # plt.clf()
        # plt.cla()

        # plt.imshow(image)

        # show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=0, output_dir=self.video_name)

    

    def track(self, image_path, out_frame_idx):
        self.predictor.load_first_frame(self.inference_state, image_path, frame_idx=out_frame_idx)

        # if out_frame_idx == 17:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=17,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )
        # elif out_frame_idx == 23:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=23,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )
        # elif out_frame_idx == 24:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=24,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )

        # if out_frame_idx == 35:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=35,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )

        # if out_frame_idx == 41:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=41,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )

        # if out_frame_idx == 42:
        #     _, out_obj_ids, out_mask_logits, ious_output = self.predictor.add_new_mask(
        #         inference_state=self.inference_state,
        #         frame_idx=42,
        #         obj_id=1,
        #         mask=self.prev_out_mask_logits[0][0],
        #     )




        frame_idx, obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits, best_masklets, worst_res_masks, second_best_res_masks = self.predictor.track(self.inference_state, image_path, start_frame_idx=out_frame_idx, points=None, labels=None, best_masklets=self.best_masklets_prev, prev_mask=self.prev_out_mask_logits)

        self.best_masklets_prev = best_masklets
        self.prev_out_mask_logits = out_mask_logits


        # image = Image.open(image_path)

        # plt.clf()
        # plt.cla()

        # plt.imshow(image)

        # show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=obj_ids[0], ann_frame_idx=out_frame_idx, output_dir=self.video_name + "_output_sam2_best_mask/")


        # image = Image.open(image_path)

        # plt.clf()
        # plt.cla()

        # plt.imshow(image)

        # show_mask((worst_res_masks[0] > 0).cpu().numpy(), plt.gca(), obj_id=obj_ids[0], ann_frame_idx=out_frame_idx, output_dir=self.video_name + "_output_sam2_best2_mask/")



        # image = Image.open(image_path)

        # plt.clf()
        # plt.cla()

        # plt.imshow(image)

        # show_mask((second_best_res_masks[0] > 0).cpu().numpy(), plt.gca(), obj_id=obj_ids[0], ann_frame_idx=out_frame_idx, output_dir=self.video_name + "_output_sam2_best3_mask/")


        print("BESTSTSS", self.best_masklets_prev)

        return obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits



def run(seq):
    VIDEO_NAME = seq
  

    video_dir =  "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + VIDEO_NAME + "/color" 


    tracker = SAM2Tracker(os.path.join(video_dir, '00000001.jpg'), VIDEO_NAME)


    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_names = frame_names[1:]
    frame_index = 1
    masks_out = []


    for imagefile in frame_names: # frames 2...N
        out_obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = tracker.track(os.path.join(video_dir, imagefile), frame_index)  #image, frame_index

        masks_out.append((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])

        frame_index += 1

    return [tracker.mask_first_frame] + masks_out

SEQ = ['book', 'agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']

# SEQ = ['animal'] #, 'flamingo1', 'frisbee', 'girl', 'nature']
# SEQ=['agility']
# SEQ=['hand', 'animal', 'animal']

SEQ.reverse()

for seq in SEQ:
    masks_out = run(seq)

    iou_curr = get_iou(seq, masks_out)
    print("IoU is: ", iou_curr)

    with open('/home.stud/rozumrus/BP/tests_multiobject/trackers/iou_ALL3_pickbest_4.txt', 'a') as file:
        file.write(f"Sequence : {seq} with iou : {iou_curr} \n")























