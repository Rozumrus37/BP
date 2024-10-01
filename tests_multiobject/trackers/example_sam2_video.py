import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')


def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, last_char=''):
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
    plt.savefig('iou1prompt_video_withN/img_' + str(ann_frame_idx) + '_' + last_char + '.png')



def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


video_dir =  "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/book/color" 

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[244, 87]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

print("Shapes are: ", out_mask_logits.shape, out_mask_logits[0].shape, (out_mask_logits[0] > 0.0).cpu().numpy().shape, len(out_mask_logits))
print("First iteration IoUs: ", ious_output)

image = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))

plt.imshow(image)
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, last_char='bboxes')


_, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    mask=out_mask_logits[0][0],
)


# ann_frame_idx = 1  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[209, 109]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )


plt.clf()
plt.cla()

plt.imshow(image)
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, last_char='masks')


print("Second iteration IoUs: ", ious_output)
print("Shapes are: ", out_mask_logits.shape, out_mask_logits[0].shape, (out_mask_logits[0] > 0.0).cpu().numpy().shape, len(out_mask_logits))



ious_scores = []


# print("YESYESYESYEYEYES")
# ann_frame_idx = 1  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[209, 109]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits, ious_output = predictor.add_new_points_or_box(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits, iou_output_scores_RR_added in predictor.propagate_in_video(inference_state):



    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    ious_scores.append(iou_output_scores_RR_added)


with open('IoU_scores/iou1prompt_video_withN.txt', 'w') as file:
    # Loop through a range of numbers (e.g., 1 to 10)
    cnt = 0

    for i in ious_scores:
        # Write each number to the file, followed by a newline
        file.write(f"For the frame: {cnt}, iou is: {i}\n")
        cnt+=1


plt.close("all")
for out_frame_idx in range(0, len(frame_names)):
    plt.clf()
    plt.cla()
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id, ann_frame_idx=out_frame_idx, last_char='masks')




# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }



#  are:  tensor([[0.5059, 0.2368, 0.8276]], device='cuda:0', dtype=torch.float16)

# or([[0.9512, 0.9536, 0.9570]]
# IoU scores for frame  48 are:  tensor([[0.9062, 0.9141, 0.9233]], device='cuda:0', dtype=torch.float16) 15
# IoU scores for frame  48 are:  tensor([[0.9053, 0.9097, 0.9204]], device='cuda:0', dtype=torch.float16) 7

# IoU scores for frame  60 are:  tensor([[0.2002, 0.9141, 0.8965]], device='cuda:0', dtype=torch.float16)
#   65 are:  tensor([[0.6567, 0.0859, 0.6523]], device='cuda:0', dtype=torch.float16)


# Frame  43 are:  tensor([[0.8989, 0.9121, 0.9146]], device='cuda:0', dtype=torch.float16)
# IoU scores for frame  43 are:  tensor([[0.8960, 0.9097, 0.9136]], device='cuda:0', dtype=torch.float16)

# IoU scores for frame  57 are:  tensor([[0.2595, 0.8740, 0.8696]], device='cuda:0', dtype=torch.float16)
# IoU scores for frame  57 are:  tensor([[0.2455, 0.8711, 0.8647]], device='cuda:0', dtype=torch.float16)

# print(out_mask_logits[0] )
# logits_tf = (out_mask_logits[0]).cpu().numpy()[0]
# output = (out_mask_logits[0]).cpu().numpy()
# output2 = (out_mask_logits[0] > 0).cpu().numpy()
# print("output1: ", output.shape)
# print("output2: ", output2.shape)

# IoU scores for frame  66 are:  tensor([[0.5171, 0.2145, 0.8506]], device='cuda:0', dtype=torch.float16) 25
# IoU scores for frame  66 are:  tensor([[0.7666, 0.0728, 0.6050]], device='cuda:0', dtype=torch.float16) 7

