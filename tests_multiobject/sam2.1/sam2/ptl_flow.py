import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os 
from torchvision.models.optical_flow import raft_large
from PIL import Image
from torchvision.utils import flow_to_image
from skimage.transform import resize_local_mean
from torchvision.io import read_image 
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import cv2

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ptlflow.get_model('sea_raft_m', 'sintel').to(device) # ptlflow.get_model('raft', 'sintel').to(device)
model.eval()


def get_mask_ptl_flow(prev_frame_path, next_frame_path,
    prev_segmentation_mask, frame_idx=None, 
    seq=None, direct_comp_to_prev_pred=False, 
    interpolation='bilinear', kernel_size=3, 
    close_trans=False, open_trans=False):
    
    if direct_comp_to_prev_pred:
        return prev_segmentation_mask
    
    img1 = cv2.imread(prev_frame_path)
    img2 = cv2.imread(next_frame_path)

    H, W, _ = img2.shape

    # IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
    io_adapter = IOAdapter(model, img1.shape[:2])
    inputs = io_adapter.prepare_inputs([img1, img2])

    predictions = model({'images':inputs['images'].to(device)}) # load to cuda 

    flow = predictions['flows'][0, 0]

    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

    mask_indices = prev_segmentation_mask > 0  # Get indices of the segmented region
    flow_x, flow_y = [], []

    masked_flow = flow[:, mask_indices]  # Select masked values directly
    flow_x = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
    flow_y = masked_flow[1].cpu().numpy() # Extract y-component and convert to list

    x_coords_masked = x_coords[mask_indices].cpu().numpy()
    y_coords_masked = y_coords[mask_indices].cpu().numpy()

    x_coords_mapped = (x_coords_masked + flow_x).round()
    y_coords_mapped = (y_coords_masked + flow_y).round()

    x_coords_mapped = np.clip(x_coords_mapped, 0, W - 1)
    y_coords_mapped = np.clip(y_coords_mapped, 0, H - 1)

    mapped_mask = np.zeros((H, W), dtype=np.uint8)
    mapped_mask[y_coords_mapped.astype(int), x_coords_mapped.astype(int)] = 1

    if open_trans:
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        next_mask = cv2.morphologyEx(next_mask, cv2.MORPH_OPEN, kernel, iterations=1) 

    if close_trans:
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        next_mask = cv2.morphologyEx(next_mask, cv2.MORPH_CLOSE, kernel, iterations=1) 

    return  mapped_mask

