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

from hq_sam.sam_hq.segment_anything import sam_model_registry, SamPredictor

USE_HQ = False
BASE_dir = "/datagrid/personal/rozumrus/BP_dg/output_vot22ST"
save_path_endwords = "_output"
ALFA=0.5
SEQ_prompted = None 
exclude_empty_masks = False
no_memory_sam2=False

# if len(sys.argv) > 1:
#     save_path_endwords = sys.argv[1]

#     if len(sys.argv) > 2:
#         ALFA = float(sys.argv[2])

#         if len(sys.argv) > 3:
#             exclude_empty_masks = int(sys.argv[3])

#             if len(sys.argv) > 4:
#                 no_memory_sam2 = int(sys.argv[4])

#                 if len(sys.argv) > 5:
#                     SEQ_prompted = sys.argv[5:]

parser = argparse.ArgumentParser()
parser.add_argument('--alfa', type=float)
parser.add_argument('--exclude_empty_masks', type=bool)
parser.add_argument('--no_memory', type=bool)
parser.add_argument('--sequences')

args = parser.parse_args()

(ALFA, 
exclude_empty_masks, 
no_memory_sam2, 
SEQ_prompted) = args.alfa, args.exclude_empty_masks, args.no_memory, args.sequences

SEQ_prompted = SEQ_prompted.split()


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True

from sam2.build_sam import build_sam2_video_realtime_predictor


def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, output_dir="temp", save_path_endwords=save_path_endwords):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    DIR = os.path.join(BASE_dir, output_dir + "_alfa" + str(ALFA) + "_nomem" + str(no_memory_sam2) + "_excl_emp_masks" + str(exclude_empty_masks))# save_path_endwords)

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    final_path = os.path.join(DIR, str(ann_frame_idx) + '.png')

    ax.imshow(mask_image)
    plt.savefig(final_path)
    
def get_bounding_box(segmentation):
    # Get the indices of the 1s in the array
    cols, rows = np.where(segmentation == True)
    
    # If there are no 1s, return None or a default bbox
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    # Get the minimum and maximum row and column indices
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Return the top-left corner and bottom-right corner
    return (min_row, min_col, max_row, max_col)


sam_checkpoint = "hq_sam/sam_hq/pretrained_checkpoint/sam_hq_vit_l.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

class SAM2Tracker(object):

    def __init__(self, image_path, VIDEO_NAME, alfa=0):
        model_cfg = "sam2_hiera_s.yaml"
        sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
        
        predictor = build_sam2_video_realtime_predictor(model_cfg, sam2_checkpoint)
        self.inference_state = predictor.init_state()
        self.predictor = predictor
        self.video_name = VIDEO_NAME
        self.alfa = alfa
        self.predictor.load_first_frame(self.inference_state, image_path)
        self.best_masklets_prev = []

        if_init = True
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        
        self.mask_first_frame = get_nth_mask(VIDEO_NAME, 0)

        _, out_obj_ids, out_mask_logits, ious_output, low_res_masks = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=np.array(self.mask_first_frame),#out_mask_logits[0][0],
        )

        self.prev_out_mask_logits = [out_mask_logits]
        self.ious = []

        # sam_checkpoint = "hq_sam/sam_hq/pretrained_checkpoint/sam_hq_vit_l.pth"
        # model_type = "vit_l"
        # device = "cuda"
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        # predictor = SamPredictor(sam)

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # import pdb; pdb.set_trace()

        # a, b, c, d = get_bounding_box((out_mask_logits[0] > 0).cpu().numpy()[0]) # np.array([[199,47,291,114]])

        # input_box = np.array([a, b, c, d])

        # input_point, input_label = None, None
        # predictor.set_image(image)
        # # import pdb; pdb.set_trace()
        # masks, scores, logits = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     # mask_input=low_res_masks[0],
        #     box = input_box,
        #     multimask_output=False,
        #     hq_token_only= False,
        # )

        # import pdb; pdb.set_trace()


        # self.vis_sam_hq(masks, out_obj_ids, 888, self.video_name, image_path)


        # self.vis(out_mask_logits, out_obj_ids, 0, self.video_name, image_path)
    

    def track(self, image_path, out_frame_idx):
        self.predictor.load_first_frame(self.inference_state, image_path, frame_idx=out_frame_idx)

        prev_mask = self.prev_out_mask_logits[-1] #if len(self.prev_out_mask_logits) < 2 else self.prev_out_mask_logits[-2]

        print("YES", len(self.prev_out_mask_logits))

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
            exclude_empty_masks=exclude_empty_masks,
            no_memory_sam2=no_memory_sam2)


        self.best_masklets_prev = best_masklets
        
        self.prev_out_mask_logits.append(out_mask_logits)

        #self.vis(out_mask_logits, obj_ids, out_frame_idx, self.video_name, image_path)
        print("Occlusion score: ", object_score_logits)

        print("BESTSTSS", self.best_masklets_prev)


        if USE_HQ:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = get_bounding_box((out_mask_logits[0] > 0).cpu().numpy()[0]) # np.array([[199,47,291,114]])
            if output != None:
                a, b, c, d = output
                input_box = np.array([a, b, c, d])

                input_point, input_label = None, None
                predictor.set_image(image)
                # import pdb; pdb.set_trace()
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=low_res_output_mask[0],
                    box = input_box,
                    multimask_output=False,
                    hq_token_only=False,
                )
                # self.vis_sam_hq(masks, obj_ids, out_frame_idx, self.video_name, image_path)
        else:
            masks = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
            self.vis(out_mask_logits, obj_ids, out_frame_idx, self.video_name, image_path)
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

    def vis(self, out_mask_logits, out_obj_ids, ann_frame_idx, output_dir, image):
        image = Image.open(image)

        plt.clf()
        plt.cla()

        plt.imshow(image)

        show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, output_dir=output_dir)


def run(seq, alfa=0.1):
    VIDEO_NAME = seq
    video_dir =  "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/" + VIDEO_NAME + "/color" 
    tracker = SAM2Tracker(os.path.join(video_dir, '00000001.jpg'), VIDEO_NAME, alfa)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_names = frame_names[1:]
    frame_index = 1
    masks_out = []

    for imagefile in frame_names[:60]:#frame_names: # frames 2...N
        out_obj_ids, out_mask_logits, iou_output_scores_RR_added, object_score_logits = tracker.track(os.path.join(video_dir, imagefile), frame_index)  #image, frame_index

        masks_out.append(out_mask_logits)

        # masks_out.append((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])

        frame_index += 1



    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the numbers with their indices on the x-axis
    ax.plot(range(1, len(tracker.ious) + 1), tracker.ious, marker='o')

    # Labeling the axes
    ax.set_xlabel('frame idx')
    ax.set_ylabel('IoU')


    # Save the plot
    file_path = 'ious_1_50_book.png'
    plt.savefig(file_path)

    # Show the plot
    plt.show()


    return [tracker.mask_first_frame] + masks_out


ALL_SEQ = ['book', 'agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']
SEQ= ['agility', 'book', 'conduction1', 'drone1', 'flamingo1', 'hand', 'hand2', 'rowing', 'zebrafish1']
SEQ=['book', 'conduction1']


if SEQ_prompted != None:
    SEQ = SEQ_prompted

# SEQ=ALL_SEQ
# SEQ.reverse()
# alfas = [0.5, 0.1, 0.25, 1, 2, 5, 10, 20]

# for alfa in alfas:
for seq in SEQ:
    masks_out = run(seq, ALFA)

    iou_curr = get_iou(seq, masks_out)
    print("IoU is: ", iou_curr)

    with open(os.path.join(BASE_dir, seq + "_alfa" + str(ALFA) + "_nomem" + str(no_memory_sam2) + "_excl_emp_masks" + str(exclude_empty_masks) + '.txt'), 'a') as file:
        file.write(f"Sequence : {seq} with iou : {iou_curr}\n")



    
























