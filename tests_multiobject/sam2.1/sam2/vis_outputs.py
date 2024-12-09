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

def create_video():
	# Set the path to the directory containing the frames
	frames_path = '/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color'  # Replace with the path to your frames
	output_video_path = 'hand2_output.mp4'
	frame_rate = 24  # Set the desired frame rate (e.g., 24 fps)

	# Get all image file names in the frames directory
	frame_files = sorted(
	    [f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
	)

	frame_files = [i for i in frame_files if i[0] != '.']

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

""" Visualize cropped image by the bbox """
def vis_cropped(image_path, bbox, output_dir, out_frame_idx):
    img_pil_full_res = Image.open(image_path)  
    min_row, min_col, max_row, max_col = bbox 

    img_pil = img_pil_full_res.crop((min_col, min_row, max_col, max_row)) # PIL requires W1, H1, W2, H2 format for bbox

    img_pil.save(output_dir + '/cropped' + str(out_frame_idx) +'.png')


""" Opens the image and calls show_mask to visualize the output mask """
def vis(mask_full_size, out_obj_ids, ann_frame_idx, image, to_save_path):
    image = Image.open(image)
    plt.clf()
    plt.cla()
    plt.axis('off')
    plt.imshow(image)

    show_mask(mask_full_size, plt.gca(), obj_id=out_obj_ids[0], ann_frame_idx=ann_frame_idx, to_save_path=to_save_path)


# extract_frames("/datagrid/personal/rozumrus/BP_dg/collected_videos/rolling_wallnuts.mov", "/datagrid/personal/rozumrus/BP_dg/collected_videos/rolling_wallnuts")
# Example usage:
# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/ants1', 
#     output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/ants1.mp4')

# create_video_from_frames('/datagrid/personal/rozumrus/BP_dg/output_vot22ST/alfa0.0_nomem0_excl_EM0_OP16_L/zebrafish1',output_video='/datagrid/personal/rozumrus/BP_dg/output_vot22ST/zebrafish1.mp4')



