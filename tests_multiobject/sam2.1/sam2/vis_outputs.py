import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
from compute_iou import *
from vot.region import io
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
from PIL import ImageEnhance
import re

""" Extract the frames from the video into the directory """
def extract_frames(video_path, output_folder, target_fps=12):
    vid = cv2.VideoCapture(video_path)
    original_fps = vid.get(cv2.CAP_PROP_FPS)  
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    save_frame_number = 0

    frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1

    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Frame interval for extraction: {frame_interval}")

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            frame_name = f"{save_frame_number:04d}.jpg"
            output_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(output_path, frame)
            print(f"saved frame {frame_name}")
            save_frame_number += 1

        frame_number += 1

    vid.release()

def vis_mem_stride_tune():
    N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'log2']
    mIoU = [73.04, 72.95, 72.74, 73.27, 73.73, 74.01, 73.36, 73.77, 73.97, 73.63, 73.67]

    # Convert 'log2' to a numeric value for plotting (optional)
    N_numeric = list(range(1, len(N))) + [11]  # 'log2' is treated as index 11 for consistent spacing

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(N_numeric, mIoU, marker='o', label='mIoU (%)', color='dodgerblue')

    # Custom X-axis labels to include 'log2'
    plt.xticks(N_numeric, N)

    y_ticks = plt.yticks()[0].tolist()  
    y_ticks.append(73.04)  

    plt.yticks(sorted(y_ticks)[:len(y_ticks)-1])
    plt.axhline(y=73.04, color='r', linestyle='--', label="baseline")

    # Labels and title
    plt.xlabel('Memory stride', fontsize=12)
    plt.ylabel('mIoU (%)', fontsize=12)
    plt.title('mIoU on VOT2022ST for different memory strides', fontsize=14)
    plt.grid(True)

    # Highlight the maximum mIoU
    max_mIoU = max(mIoU)
    max_idx = mIoU.index(max_mIoU)
    plt.scatter(N_numeric[max_idx], max_mIoU, color='red')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()
    plt.savefig("tuning_MS.png")

# [75.9, 75.9, 75.7, 75.8, 75.7, 75.6, 75.8]
# [4, 5, 6, 7, 8, 9, 10]

def vis_BB_N_tune():


    N = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]
    minBB_0 = [43.2, 54.7, 62.0, 66.6, 66.7, 68.6, 69.6, 71.3, 74.2, 71.4, 73.2, 73.0, 72.9, 73.1, 72.1, 71.8, 73.6]
    minBB_128 = [66.9, 68.0, 68.6, 72.1, 70.1, 72.1, 73.5, 74.2, 74.9, 73.6, 74.7, 74.0, 73.6]

    minBB_256 = [74.4, 74.2, 75.3, 73.8, 74.3, 72.6, 74.7, 74.5, 74.9, 73.9, 74.0, 73.5, 73.5, 73.7, 73.7, 73.3, 73.1]

    minBB_512 = [74.7, 75.4, 75.3, 75.1, 75.1, 75.1, 75.2, 75.3, 75.1, 74.9, 74.0, 72.9, 73.8]


    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(N, minBB_0, marker='o', label='m: 0')
    plt.plot(N[:12] + [20], minBB_128, marker='o', label='m: 128')
    plt.plot(N, minBB_256, marker='o', label='m: 256')
    plt.plot(N[:12] + [20], minBB_512, marker='o', label='m: 512')

    plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    y_ticks = plt.yticks()[0].tolist()  
    y_ticks.append(73.04)  

    plt.yticks(sorted(y_ticks)[:len(y_ticks)-1])
    plt.axhline(y=73.04, color='r', linestyle='--', label="original")

    # Labels and title
    plt.xlabel('N: the factor for enlarging every side of prev. bbox', fontsize=12)
    plt.ylabel('mIoU', fontsize=12)
    plt.title('mIoU on VOT2022ST for frames cropped by enlarged prev. bbox', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
    plt.savefig("tuning_N_with_mBB.png")


def create_video():
    # Set the path to the directory containing the frames
    frames_path = '/mnt/data_personal/rozumrus/BP_dg/sam2.1_output/drone_flip/cropped_with_darkened' # vot2020ST/sequences/drone_flip/color'  # Replace with the path to your frames
    output_video_path = 'drone_flip_512_output.mp4'
    frame_rate = 15

    def sort_by_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    frame_files = sorted(
        [f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=sort_by_key
    )

    # # Get all image file names in the frames directory
    # frame_files = sorted(
    #     [f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # )

    frame_files = [i for i in frame_files if i[0] != '.']

    print(frame_files)

    # Ensure there are frames to create a video
    if not frame_files:
        raise ValueError("No frames found in the specified directory.")

    # Get the size of the frames
    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        out.write(frame)

    # Release the video writer object
    out.release()

    print(f"Video saved as {output_video_path}")



""" Create the video from the images in the directory """
def create_video_from_frames(dir_path, output_video='output_video.mp4', fps=5):
    images = [img for img in os.listdir(dir_path) if img.endswith((".png", ".jpg", ".jpeg")) and not img.startswith('.')]
    images.sort(key=lambda p: int(os.path.splitext(p)[0])) 

    if not images:
        print("No image files found in the directory.")
        return

    first_image = cv2.imread(os.path.join(dir_path, images[0]))
    height, width, layers = first_image.shape


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(dir_path, image))
        video.write(img)

    video.release()
    print(f"Video created successfully: {output_video}")

""" Visualize the mask in the plt with the image """
def show_mask(mask, ax, obj_id=None, random_color=False, ann_frame_idx=0, to_save_path=None):
    ax.set_axis_off()
    plt.margins(0,0)

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
    plt.savefig(final_path, pad_inches=0, bbox_inches='tight', transparent=True)


def darken_outside_bbox(img_pil_full_res, bbox, output_path, darkness_factor=0.2):
    """
    Darken the area outside the bounding box in the given image.

    Parameters:
        img_pil_full_res (PIL.Image.Image): The input image.
        bbox (tuple): A tuple (min_row, min_col, max_row, max_col) defining the bounding box.
        output_path (str): Path to save the resulting image.
        darkness_factor (float): Factor by which to darken the outside area (0.0 - fully black, 1.0 - no change).
    """
    min_row, min_col, max_row, max_col = bbox

    # Create a mask with the same size as the image
    mask = Image.new("L", img_pil_full_res.size, 0)

    # Draw a white rectangle for the bounding box area on the mask
    bbox_area = Image.new("L", (max_col - min_col, max_row - min_row), 255)
    mask.paste(bbox_area, (min_col, min_row))

    # Darken the entire image
    darkened_img = ImageEnhance.Brightness(img_pil_full_res).enhance(darkness_factor)

    # Composite the images: keep the original inside the bounding box, darken outside
    result_img = Image.composite(img_pil_full_res, darkened_img, mask)

    # Save the result
    result_img.save(output_path)


""" Visualize cropped image by the bbox """
def vis_cropped(mask_full_size, out_obj_ids, out_frame_idx, image_path, bbox, output_dir):
    img_pil_full_res = Image.open(image_path)  
    min_row, min_col, max_row, max_col = bbox 

    img_pil = img_pil_full_res.crop((min_col, min_row, max_col, max_row)) # PIL requires W1, H1, W2, H2 format for bbox

    img_pil.save(output_dir + '/cropped' + str(out_frame_idx) +'.png')

    # darken_outside_bbox(img_pil_full_res, bbox, output_dir + '/cropped_with_darkened' + str(out_frame_idx) +'.png')

    vis_for_darkened_cropped(mask_full_size, out_obj_ids, out_frame_idx, image_path, output_dir + '/cropped_with_darkened', bbox)


def vis_for_darkened_cropped(mask_full_size, out_obj_ids, ann_frame_idx, image, to_save_path, bbox):
    img_pil_full_res = Image.open(image)

    min_row, min_col, max_row, max_col = bbox

    # Create a mask with the same size as the image
    mask = Image.new("L", img_pil_full_res.size, 0)

    # Draw a white rectangle for the bounding box area on the mask
    bbox_area = Image.new("L", (max_col - min_col, max_row - min_row), 255)
    mask.paste(bbox_area, (min_col, min_row))

    # Darken the entire image
    darkened_img = ImageEnhance.Brightness(img_pil_full_res).enhance(0.2)

    # Composite the images: keep the original inside the bounding box, darken outside
    image = Image.composite(img_pil_full_res, darkened_img, mask)


    plt.clf()
    plt.cla()
    plt.axis('off')
    plt.imshow(image)

    show_mask(mask_full_size, plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)



""" Opens the image and calls show_mask to visualize the output mask """
def vis(mask_full_size, out_obj_ids, ann_frame_idx, image, to_save_path):
    image = Image.open(image)
    plt.clf()
    plt.cla()
    plt.axis('off')
    plt.imshow(image)

    show_mask(mask_full_size, plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)

# vis_mem_stride_tune()

# create_video()

vis_BB_N_tune()


# extract_frames("/datagrid/personal/rozumrus/BP_dg/collected_videos/rolling_wallnuts.mov", "/datagrid/personal/rozumrus/BP_dg/collected_videos/rolling_wallnuts")
# Example usage:
# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/ants1', 
#     output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/ants1.mp4')

# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/zebrafish1',output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/zebrafish1.mp4')



