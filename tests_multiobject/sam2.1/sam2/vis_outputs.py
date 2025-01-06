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


def vis_mem_stride_tune():
    N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'log2']
    mIoU = [73.04, 72.95, 72.74, 73.27, 73.73, 74.01, 73.36, 73.77, 73.97, 73.63, 73.67]

    N_numeric = list(range(1, len(N))) + [11] 

    plt.figure(figsize=(10, 6))
    plt.plot(N_numeric, mIoU, marker='o', label='mIoU (%)', color='dodgerblue')

    plt.xticks(N_numeric, N)

    y_ticks = plt.yticks()[0].tolist()  
    y_ticks.append(73.04)  

    plt.yticks(sorted(y_ticks)[:len(y_ticks)-1])
    plt.axhline(y=73.04, color='r', linestyle='--', label="baseline")

    plt.xlabel('Memory stride', fontsize=12)
    plt.ylabel('mIoU (%)', fontsize=12)
    plt.title('mIoU on VOT2022ST for different memory strides', fontsize=14)
    plt.grid(True)

    max_mIoU = max(mIoU)
    max_idx = mIoU.index(max_mIoU)
    plt.scatter(N_numeric[max_idx], max_mIoU, color='red')

    plt.legend()

    plt.show()
    plt.savefig("tuning_MS.png")


def vis_BB_N_tune():
    N = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]
    minBB_0 = [43.2, 54.7, 62.0, 66.6, 66.7, 68.6, 69.6, 71.3, 74.2, 71.4, 73.2, 73.0, 72.9, 73.1, 72.1, 71.8, 73.6]
    minBB_128 = [66.9, 68.0, 68.6, 72.1, 70.1, 72.1, 73.5, 74.2, 74.9, 73.6, 74.7, 74.0, 73.6]

    minBB_256 = [74.4, 74.2, 75.3, 73.8, 74.3, 72.6, 74.7, 74.5, 74.9, 73.9, 74.0, 73.5, 73.5, 73.7, 73.7, 73.3, 73.1]

    minBB_512 = [74.7, 75.4, 75.3, 75.1, 75.1, 75.1, 75.2, 75.3, 75.1, 74.9, 74.0, 72.9, 73.8]

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

    plt.xlabel('N: the factor for enlarging every side of prev. bbox', fontsize=12)
    plt.ylabel('mIoU', fontsize=12)
    plt.title('mIoU on VOT2022ST for frames cropped by enlarged prev. bbox', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig("tuning_N_with_mBB.png")


def create_video():
    frames_path = '/mnt/data_personal/rozumrus/BP_dg/sam2.1_output/drone_flip/cropped_with_darkened' 
    output_video_path = 'drone_flip_512_output.mp4'
    frame_rate = 15

    def sort_by_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    frame_files = sorted(
        [f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=sort_by_key
    )

    frame_files = [i for i in frame_files if i[0] != '.']

    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        out.write(frame)

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
    min_row, min_col, max_row, max_col = bbox

    mask = Image.new("L", img_pil_full_res.size, 0)

    bbox_area = Image.new("L", (max_col - min_col, max_row - min_row), 255)
    mask.paste(bbox_area, (min_col, min_row))

    darkened_img = ImageEnhance.Brightness(img_pil_full_res).enhance(darkness_factor)

    result_img = Image.composite(img_pil_full_res, darkened_img, mask)

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

    mask = Image.new("L", img_pil_full_res.size, 0)

    bbox_area = Image.new("L", (max_col - min_col, max_row - min_row), 255)
    mask.paste(bbox_area, (min_col, min_row))

    darkened_img = ImageEnhance.Brightness(img_pil_full_res).enhance(0.2)

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



