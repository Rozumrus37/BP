import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
from compute_iou import *


def increase_bbox_area(H, W, min_row, min_col, max_row, max_col, factor=2):
    # Calculate the center of the original rectangle
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    # Calculate the current width and height
    height = max_row - min_row
    width = max_col - min_col

    # Double the width and height by multiplying by sqrt(2)
    scale_factor = factor**0.5
    new_height = height * scale_factor
    new_width = width * scale_factor

    # Calculate new coordinates by expanding around the center
    new_min_row = center_row - new_height / 2
    new_min_col = center_col - new_width / 2
    new_max_row = center_row + new_height / 2
    new_max_col = center_col + new_width / 2


     # Apply boundary constraints to keep the box within the image
    if new_min_row < 0:
        new_min_row = 0
    if new_min_col < 0:
        new_min_col = 0
    if new_max_row > H:
        new_max_row = H-1
    if new_max_col > W:
        new_max_col = W-1


    return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)


def increase_bbox_to_square(H, W, min_row, min_col, max_row, max_col, factor=2):
    # Calculate the center of the original rectangle
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    # Calculate the current width and height
    height = max_row - min_row
    width = max_col - min_col

    # Double the width and height by multiplying by sqrt(2)
    new_height = factor*max(height, width) #height * scale_factor
    new_width = factor*max(height, width) # width * scale_factor

    # Calculate new coordinates by expanding around the center
    new_min_row = center_row - new_height / 2
    new_min_col = center_col - new_width / 2
    new_max_row = center_row + new_height / 2
    new_max_col = center_col + new_width / 2


     # Apply boundary constraints to keep the box within the image
    if new_min_row < 0:
        new_min_row = 0
    if new_min_col < 0:
        new_min_col = 0
    if new_max_row > H:
        new_max_row = H-1
    if new_max_col > W:
        new_max_col = W-1


    return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)



def get_full_size_mask(out_mask_logits, bbox, image, ann_frame_idx, H, W):
    # image = Image.open(image)
    # mask_curr = get_nth_mask(seq, ann_frame_idx)
    # H, W = mask_curr.shape

    if bbox != None:
        min_row, min_col, max_row, max_col = bbox
        filled_mask = np.zeros((H,W))

        filled_mask[min_col:max_col, min_row:max_row] = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
    else:
        filled_mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]

    return filled_mask

def check_boundaries(H, W, min_row, min_col, max_row, max_col):
    if min_row < 0 or min_col < 0:
        return False  # Bounding box cannot start from a negative position
    if max_row >= W or max_col >= H:
        return False  # Bounding box cannot exceed the height or width
    if min_row > max_row or min_col > max_col:
        return False  # Invalid bounding box dimensions

    return True  # Bounding box fits within the given dimensions

def load_frames(video_dir):
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # print(frame_names)
    frame_names = [i for i in frame_names if i[0] != '.']

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names


def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, to_save_path=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    DIR = to_save_path

    if not os.path.exists(DIR):
        os.makedirs(DIR)

    final_path = os.path.join(DIR, str(ann_frame_idx) + '.png')

    ax.imshow(mask_image)
    plt.savefig(final_path)
    
def get_bounding_box(segmentation):
    cols, rows = np.where(segmentation == True)
    
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    return (min_row, min_col, max_row, max_col)

def vis_IoU_graph(ious, file_path):
    fig, ax = plt.subplots()

    ax.plot(range(1, len(ious) + 1), ious, marker='o')

    ax.set_xlabel('frame idx')
    ax.set_ylabel('IoU')

    file_path = file_path +'.png' 
    plt.savefig(file_path)

    plt.show()


def create_video_from_frames(dir_path, output_video='output_video.mp4', fps=5):
    images = [img for img in os.listdir(dir_path) if img.endswith((".png", ".jpg", ".jpeg")) and not img.startswith('.')]
    images.sort(key=lambda p: int(os.path.splitext(p)[0])) # Sort files by name to maintain the correct frame order

    if not images:
        print("No image files found in the directory.")
        return

    # Read the first image to get the dimensions
    first_image = cv2.imread(os.path.join(dir_path, images[0]))
    height, width, layers = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop over all images and write them to the video
    for image in images:
        img = cv2.imread(os.path.join(dir_path, image))
        video.write(img)

    # Release the video writer
    video.release()
    print(f"Video created successfully: {output_video}")

# Example usage:
# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/ants1', 
#     output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/ants1.mp4')

# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/zebrafish1',output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/zebrafish1.mp4')

