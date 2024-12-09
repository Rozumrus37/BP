import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os 
from vot.region import io
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss


""" Get the bounding box of the given segmentation """
def get_bounding_box(segmentation):
    rows, cols = np.where(segmentation == True)
    
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    return (min_row, min_col, max_row, max_col)


""" Insert the cropped segmentaiton back to the original dimensions of the image """
def get_full_size_mask(out_mask_logits, bbox, H, W):
    if bbox != None:
        min_row, min_col, max_row, max_col = bbox
        filled_mask = np.zeros((H,W))

        filled_mask[min_row:max_row, min_col:max_col] = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
    else:
        filled_mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]

    return filled_mask


""" Increase the height and width of the original bounding box by the factor and align 
the new boudning box to the orignal bounding box's center """
def increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=256, factor=2):
    # calculate the center of the original bbox
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    # calculate the height and width of the original bbox
    height = max_row - min_row
    width = max_col - min_col

    # multiply height and width by the scale factor
    scale_factor = factor ** 0.5
    new_height = height * scale_factor
    new_width = width * scale_factor

    MIN_LEN = min_box_factor

    if new_height < MIN_LEN:
        new_height = MIN_LEN
    
    if new_width < MIN_LEN:
        new_width = MIN_LEN

    # calculate new coordinates of the bbox by aligning to the original bbox center
    new_min_row = center_row - new_height / 2
    new_min_col = center_col - new_width / 2
    new_max_row = center_row + new_height / 2
    new_max_col = center_col + new_width / 2

    # check the boundaries and adjust if needed
    if new_min_row < 0:
        new_min_row = 0
    if new_min_col < 0:
        new_min_col = 0
    if new_max_row > H:
        new_max_row = H
    if new_max_col > W:
        new_max_col = W

    return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)

""" Similiar to increase_bbox_area but make a square """
def increase_bbox_to_square(H, W, min_row, min_col, max_row, max_col, factor=2):
    # calculate the center of the original bbox
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    # calculate the height and width of the original bbox
    height = max_row - min_row
    width = max_col - min_col

    # find the maximum between height and weight and assign the largest as new square side
    new_height = factor * max(height, width) # height * scale_factor
    new_width = factor * max(height, width) # width * scale_factor

    # calculate new coordinates of the bbox by aligning to the original bbox center
    new_min_row = center_row - new_height / 2
    new_min_col = center_col - new_width / 2
    new_max_row = center_row + new_height / 2
    new_max_col = center_col + new_width / 2

    MIN_LEN = 256

    if new_height < MIN_LEN:
        new_height = MIN_LEN
    
    if new_width < MIN_LEN:
        new_width = MIN_LEN

    # check the boundaries and adjust if needed
    if new_min_row < 0:
        new_min_row = 0
    if new_min_col < 0:
        new_min_col = 0
    if new_max_row > H:
        new_max_row = H
    if new_max_col > W:
        new_max_col = W

    return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)


""" Load the frames names from the directory into the list. Then sort them in ascending order """
def load_frames(video_dir, double_memory=False):
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    # Remove hidden files
    if double_memory:
        frame_names = [i for i in frame_names if i[0] != '.'] + [i for i in frame_names if i[0] != '.']
    else:
        frame_names = [i for i in frame_names if i[0] != '.']

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return  frame_names


""" DINO methods starts here (some experiments have been made with the model) """
def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)

def extract_features_dino(image):
    processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')

    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to('cuda')
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

""" DINO methods finish here """

