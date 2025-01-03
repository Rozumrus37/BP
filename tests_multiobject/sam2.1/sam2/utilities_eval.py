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

from typing import Tuple

# def get_adjusted_bbox(
#     img_width: int,
#     img_height: int,
#     bbox: Tuple[int, int, int, int]
# ) -> Tuple[int, int, int, int]:

# def increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=256, factor=4):
#     """
#     Adjusts a bounding box to have a new size where its width and height are
#     half of the image dimensions (W / 2, H / 2). If the new box exceeds the
#     image boundaries, it is shifted to fit within the image.

#     Parameters:
#         H (int): Height of the image.
#         W (int): Width of the image.
#         min_row (int): Minimum row (y_min) of the current bounding box.
#         min_col (int): Minimum column (x_min) of the current bounding box.
#         max_row (int): Maximum row (y_max) of the current bounding box.
#         max_col (int): Maximum column (x_max) of the current bounding box.

#     Returns:
#         Tuple[int, int, int, int]: Adjusted bounding box as (new_min_row, new_min_col, new_max_row, new_max_col).
#     """
#     # Calculate the center of the current bounding box

#     for factor in [1]:
#         center_row = (min_row + max_row) // 2
#         center_col = (min_col + max_col) // 2

#         # Define new dimensions for the bounding box (half the image width and height)


#         new_height = H // factor
#         new_width = W // factor

#         if new_height 

#         # Calculate the new bounding box coordinates
#         new_min_row = center_row - new_height // 2
#         new_min_col = center_col - new_width // 2
#         new_max_row = new_min_row + new_height
#         new_max_col = new_min_col + new_width

#         # Adjust if the bounding box exceeds the image boundaries
#         if new_min_row < 0:
#             shift = -new_min_row
#             new_min_row += shift
#             new_max_row += shift
#         if new_max_row > H:
#             shift = new_max_row - H
#             new_min_row -= shift
#             new_max_row -= shift
#         if new_min_col < 0:
#             shift = -new_min_col
#             new_min_col += shift
#             new_max_col += shift
#         if new_max_col > W:
#             shift = new_max_col - W
#             new_min_col -= shift
#             new_max_col -= shift

#         area = abs(min_row-max_row)*abs(min_col-max_col)*100 / (abs(new_min_row-new_max_row)* abs(new_min_col- new_max_col))

#         if area <= 5:
#             # print(area, factor)
#             break

#     # if new_min_row < 0:
#     #     new_min_row = 0
#     # if new_min_col < 0:
#     #     new_min_col = 0
#     # if new_max_row > H:
#     #     new_max_row = H
#     # if new_max_col > W:
#     #     new_max_col = W



    


#     return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)(6+x) / 3 = 5 6+x=15 x = 3 


# k1 = AR_img 20 / 10 = 2 k1 = W / H
# k2 = AR_bbox 19 / 6 = 3   k2 = (W_bbox)/(H_bbox)  W_bbox / (H_bbox+M) = k1 -> M=k1*H_bbox-W_bbox

# k1*H_bbox + M*k1 = W_bbox -> M = (W_bbox - k1*H_bbox) / k1
# 19 / 


# k1 > k2
# W_bbox_in_AR_img = k1*H_bbox
# H_bbox_in_AR_img = H_bbox

# k2 > k1
# W_bbox_in_AR_img = W_bbox
# H_bbox_in_AR_img = W_bbox / k1

# 1. k2 < k1 then  
# 2. k1 > k2

# """ Increase the height and width of the original bounding box by the factor and align 
# the new boudning box to the orignal bounding box's center """
# def increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=256, factor=2):
#     # calculate the center of the original bbox
#     center_row = (min_row + max_row) / 2
#     center_col = (min_col + max_col) / 2

#     # calculate the height and width of the original bbox
#     height = max_row - min_row
#     width = max_col - min_col

#     # multiply height and width by the scale factor
#     scale_factor = factor ** 0.5
#     new_height = height * scale_factor
#     new_width = width * scale_factor

#     min_box_width = W / 2
#     min_box_height = H / 2

#     k1 = W / H
#     k2 = new_width / new_height

#     MIN_LEN = min_box_factor

#     # print(height, width, new_height, new_width)

#     # if new_height < MIN_LEN:
#     #     new_height = MIN_LEN
    
#     # if new_width < MIN_LEN:
#     #     new_width = MIN_LEN

#     # if new_height < MIN_LEN and new_width < MIN_LEN:
#     #     new_height = MIN_LEN
#     #     new_width = MIN_LEN
#     # elif new_height < MIN_LEN:
#     #     if new_width < H:
#     #         new_height = new_width
#     # elif new_width < MIN_LEN:
#     #     if new_height < W:
#     #         new_width = new_height

#     if k1 > k2:
#         new_width = k1 * new_height

#     if k2 > k1:
#         new_height = new_width / k1


#     if new_height < min_box_height and new_width < min_box_width:
#         new_height = min_box_height
#         new_width = min_box_width
#     # elif new_height < min_box_height: #or new_width < min_box_width:
#     #     # new_height = min_box_height
#     #     new_height = width / k1   #min_box_width
#     # elif new_width < min_box_width:
#     #     new_width = k1 * height


    
#     # if :
        


#     # if new_height != new_width:
#     #     new_height = max(new_height, new_width)
#     #     new_width = max(new_width, new_height)

#     # if new_height < min_box_height:
#     #     new_height = min_box_height
    
#     # if new_width < min_box_width:
#     #     new_width = min_box_width



#     # new_height = 512
#     # new_width = 512
#     # calculate new coordinates of the bbox by aligning to the original bbox center
#     new_min_row = center_row - new_height / 2
#     new_min_col = center_col - new_width / 2
#     new_max_row = center_row + new_height / 2
#     new_max_col = center_col + new_width / 2

#     # check the boundaries and adjust if needed
#     # if new_min_row < 0:
#     #     new_min_row = 0
#     # if new_min_col < 0:
#     #     new_min_col = 0
#     # if new_max_row > H:
#     #     new_max_row = H
#     # if new_max_col > W:
#     #     new_max_col = W

#     # print("First", abs(new_min_row-new_max_row), abs(new_min_col-new_max_col), abs(new_min_col-new_max_col)/abs(new_min_row-new_max_row))

#     if new_min_row < 0:
#         shift = -new_min_row
#         new_min_row += shift
#         new_max_row += shift
#     if new_max_row > H:
#         shift = new_max_row - H
#         new_min_row -= shift
#         new_max_row -= shift
#     if new_min_col < 0:
#         shift = -new_min_col
#         new_min_col += shift
#         new_max_col += shift
#     if new_max_col > W:
#         shift = new_max_col - W
#         new_min_col -= shift
#         new_max_col -= shift


#     if abs(new_min_row-new_max_row) > H or abs(new_min_col-new_max_col) > W:
#         new_min_row = 0
#         new_min_col = 0

#         new_max_row = H
#         new_max_col = W


#     # if new_min_row < 0:
#     #     new_min_row = 0
#     # if new_min_col < 0:
#     #     new_min_col = 0
#     # if new_max_row > H:
#     #     new_max_row = H
#     # if new_max_col > W:
#     #     new_max_col = W

#     # print("Second", abs(new_min_row-new_max_row), abs(new_min_col-new_max_col), abs(int(new_min_col)-int(new_max_col))/abs(int(new_min_row)-int(new_max_row)))

#     return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)


def increase_bbox_area_for_parallel(H, W, min_row, min_col, max_row, max_col, factor=2):
    # calculate the center of the original bbox
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2

    # calculate the height and width of the original bbox
    height = max_row - min_row
    width = max_col - min_col

    new_height = H * 1.0 / factor 
    new_width = W * 1.0 / factor

    # calculate new coordinates of the bbox by aligning to the original bbox center
    new_min_row = center_row - new_height / 2
    new_min_col = center_col - new_width / 2
    new_max_row = center_row + new_height / 2
    new_max_col = center_col + new_width / 2

    if new_min_row < 0:
        shift = -new_min_row
        new_min_row += shift
        new_max_row += shift
    if new_max_row > H:
        shift = new_max_row - H
        new_min_row -= shift
        new_max_row -= shift
    if new_min_col < 0:
        shift = -new_min_col
        new_min_col += shift
        new_max_col += shift
    if new_max_col > W:
        shift = new_max_col - W
        new_min_col -= shift
        new_max_col -= shift

    return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)



# def increase_bbox_area(H, W, min_row, min_col, max_row, max_col, min_box_factor=256, factor=2):
#     # calculate the center of the original bbox
#     center_row = (min_row + max_row) / 2
#     center_col = (min_col + max_col) / 2

#     # calculate the height and width of the original bbox
#     height = max_row - min_row
#     width = max_col - min_col

#     # multiply height and width by the scale factor
#     scale_factor = factor ** 0.5
#     new_height = height * scale_factor
#     new_width = width * scale_factor

#     HWs = [(H / 8.0, W / 8.0), (H / 4.0, W / 4.0), (H / 2.0, W / 2.0), (H, W)]

#     new_height = H / factor 
#     new_width = W / factor

#     # MIN_LEN = min_box_factor

#     # if new_height < MIN_LEN:
#     #     new_height = MIN_LEN
    
#     # if new_width < MIN_LEN:
#     #     new_width = MIN_LEN

#     # calculate new coordinates of the bbox by aligning to the original bbox center
#     new_min_row = center_row - new_height / 2
#     new_min_col = center_col - new_width / 2
#     new_max_row = center_row + new_height / 2
#     new_max_col = center_col + new_width / 2

    
#     # if new_min_row < 0:
#     #     new_min_row = 0
#     # if new_min_col < 0:
#     #     new_min_col = 0
#     # if new_max_row > H:
#     #     new_max_row = H
#     # if new_max_col > W:
#     #     new_max_col = W


#     # H_bbox_new = new_max_row - new_min_row
#     # W_bbox_new = new_max_col - new_min_col

#     # for (h, w) in HWs:
#         # if H_bbox_new <= h and W_bbox_new <= w:
#         #     print(H_bbox_new, h, W_bbox_new, w)
#     # new_min_row = center_row - h / 2
#     # new_min_col = center_col - w / 2
#     # new_max_row = center_row + h / 2
#     # new_max_col = center_col + w / 2

#     if new_min_row < 0:
#         shift = -new_min_row
#         new_min_row += shift
#         new_max_row += shift
#     if new_max_row > H:
#         shift = new_max_row - H
#         new_min_row -= shift
#         new_max_row -= shift
#     if new_min_col < 0:
#         shift = -new_min_col
#         new_min_col += shift
#         new_max_col += shift
#     if new_max_col > W:
#         shift = new_max_col - W
#         new_min_col -= shift
#         new_max_col -= shift
#     # break

#     # print("H and W: ", int(new_max_row)-int(new_min_row), int(new_max_col)-int(new_min_col), "object", height, width)
#     # check the boundaries and adjust if needed
#     # if new_min_row < 0:
#     #     new_min_row = 0
#     # if new_min_col < 0:
#     #     new_min_col = 0
#     # if new_max_row > H:
#     #     new_max_row = H
#     # if new_max_col > W:
#     #     new_max_col = W

#     return int(new_min_row), int(new_min_col), int(new_max_row), int(new_max_col)

    

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

