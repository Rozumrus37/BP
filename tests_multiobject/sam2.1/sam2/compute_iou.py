from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import sys
import numpy as np
from vot.region import io
import cv2


""" Processing the VOT format binary gt mask """
def apply_mask_to_image(binary_mask, offset, file_path):
    image = Image.open(file_path)
    mask_indices = np.argwhere(binary_mask == 1)

    pixs = []
    for index in mask_indices:
        x, y = index 
        x += offset[1]
        y += offset[0]
        if 0 < y <= image.width and 0 < x <= image.height:
            pixs.append((y, x))
    return pixs
    
""" Get n-th mask for the given sequence """
def get_nth_mask(gt_folder_name, n):
    # path to groundtruth directory
    file_path = os.path.join("/datagrid/personal/rozumrus/BP_dg/votstval/sequences", gt_folder_name, 'groundtruth_object1.txt')

    # path to the image directory  
    file_path_img = os.path.join("/datagrid/personal/rozumrus/BP_dg/votstval/sequences", gt_folder_name, 'color/00000001.jpg')

    with open(file_path, 'r') as file:
        lines_gt = file.readlines()

    s = lines_gt[n]

    binary_mask = np.array(io.parse_region(s)._mask)
    offset = io.parse_region(s)._offset
    pixs = apply_mask_to_image(binary_mask, offset, file_path_img)

    image = Image.open(file_path_img)
    output_mask = np.zeros((image.height, image.width))

    for i in pixs:
        output_mask[i[1], i[0]] = 1

    return output_mask

""" Calculate IoU for two binary masks """
def obatin_iou(array1, array2):
    array1_bool = array1.astype(bool)
    array2_bool = array2.astype(bool)
    
    intersection = np.logical_and(array1_bool, array2_bool)
    union = np.logical_or(array1_bool, array2_bool)

    # if both gts and predictions are empty, then IoU is 1
    if np.sum(union) == 0:
        return 1.0

    iou = np.sum(intersection) / np.sum(union)
    
    return iou

""" Compute IoU between predicted masks and gt masks """
def get_iou(gt_folder_name, predicted_masks):
    file_path = os.path.join("/datagrid/personal/rozumrus/BP_dg/votstval/sequences", gt_folder_name, 'groundtruth_object1.txt')   

    with open(file_path, 'r') as file:
        lines_gt = file.readlines()

    strings_gt = [line.strip() for line in lines_gt[0:]]

    iou = []

    for i in range(1, len(strings_gt)):
        binary_mask = get_nth_mask(gt_folder_name, i)
        iou.append(obatin_iou(predicted_masks[i], binary_mask))

    iou = np.array(iou)
    return iou.mean() * 100

