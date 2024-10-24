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

from sam2.build_sam import build_sam2_video_predictor


def run(seq):
    VIDEO_NAME = seq
    sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    video_dir =  "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + VIDEO_NAME + "/color" 

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')

    def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0):
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
        plt.savefig('output_book/' + str(ann_frame_idx) + '.png')


    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    mask_first_frame = get_nth_mask(VIDEO_NAME, 0)

    _, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=mask_first_frame,
    )

    plt.clf()
    plt.cla()
    # plt.imshow(Image.open(image))

    # show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=99)

    print("First iteration IoUs: ", ious_output)
    print("Shapes are: ", out_mask_logits.shape, out_mask_logits[0].shape, (out_mask_logits[0] > 0.0).cpu().numpy().shape, len(out_mask_logits))


    ious_scores = []
    masks_out = []

    for out_frame_idx, out_obj_ids, out_mask_logits, iou_output_scores_RR_added in predictor.propagate_in_video(inference_state):
        masks_out.append((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])

    return masks_out

SEQ = ['kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']


for seq in SEQ:
    masks_out = run(seq)

    iou_curr = get_iou(seq, masks_out)
    print("IoU is: ", iou_curr)

    with open('/home.stud/rozumrus/BP/tests_multiobject/trackers/ious_.txt', 'a') as file:
        file.write(f"Sequence : {seq} with iou : {iou_curr} \n")























