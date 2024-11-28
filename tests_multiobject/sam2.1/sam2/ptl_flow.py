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
try:
    from ptlflow.utils.io_adapter import IOAdapter
    from ptlflow.utils import flow_utils
except e:
    pass

# Load the two images
img1 = cv2.imread('/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/animal/color/00000001.jpg')
img2 = cv2.imread('/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/animal/color/00000001.jpg')

# Get an initialized model from PTLFlow
model = ptlflow.get_model('raft_small', 'things')
model.eval()

print("ok")
# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
io_adapter = IOAdapter(model, img1.shape[:2])
inputs = io_adapter.prepare_inputs([img1, img2])

predictions = model(inputs)

flow = predictions['flows'][0, 0]

import pdb; pdb.set_trace()

model = ptlflow.get_model('sea_raft')
model.eval()

# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
io_adapter = IOAdapter(model, img1.shape[:2])
inputs = io_adapter.prepare_inputs([img1, img2])

predictions = model(inputs)

flow = predictions['flows'][0, 0]

import pdb; pdb.set_trace()

model = ptlflow.get_model('sea_raft_l')
model.eval()

# IOAdapter is a helper to transform the two images into the input format accepted by PTLFlow models
io_adapter = IOAdapter(model, img1.shape[:2])
inputs = io_adapter.prepare_inputs([img1, img2])

predictions = model(inputs)

flow = predictions['flows'][0, 0]

import pdb; pdb.set_trace()


# def get_mask(prev_frame_path, next_frame_path, prev_segmentation_mask, frame_idx=None, seq=None, direct_comp_to_prev_pred=False, 
#     interpolation='bilinear', kernel_size=3, close_trans=False, open_trans=False):

#     if direct_comp_to_prev_pred:
#         return prev_segmentation_mask
#     # next_frame_path = "tennis_shift20.jpg" #"shifted_by_10_frame1.jpg"
#     # prev_frame_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/animal/color/00000001.jpg"
#     # next_frame_path =  "animal1_shifted_10.jpg"#/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/anima/color/00000002.jpg"

#     frame1 = read_image(prev_frame_path) 
#     frame2 = read_image(next_frame_path) 

#     img1_batch = torch.stack([frame1])
#     img2_batch = torch.stack([frame2])

#     W_original, H_original = Image.open(prev_frame_path).size

#     if interpolation == 'bilinear':
#         H_resized, W_resized = H_original + (((H_original // 8) + 1) * 8 - H_original) % 8, W_original + (((W_original // 8) + 1) * 8 - W_original) % 8
#         img1_batch, img2_batch = preprocess_resize(img1_batch, img2_batch, H_resized, W_resized) #padder)
#     elif interpolation == 'replicate':
#         padder = InputPadder(frame1.shape)
#         img1_batch, img2_batch = preprocess_padder(img1_batch, img2_batch, padder)
#         H_resized, W_resized = img1_batch.shape[2], img1_batch.shape[3] 


#     list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

#     predicted_flows = list_of_flows[-1] # obtain last of 12 iteration
#     flow = predicted_flows[0]  # Shape: (2, H_resized, W_resized)
    
#     if interpolation == 'bilinear':
#         prev_segmentation_mask = F.resize(torch.tensor(prev_segmentation_mask).unsqueeze(0).unsqueeze(0), size=[H_resized, W_resized], antialias=False) # 
#     elif interpolation == 'replicate':
#         prev_segmentation_mask = padder.pad(torch.tensor(prev_segmentation_mask).unsqueeze(0).unsqueeze(0))[0] 

#     y_coords, x_coords = torch.meshgrid(torch.arange(H_resized, device=device), torch.arange(W_resized, device=device), indexing="ij")
   
#     mask_indices = prev_segmentation_mask.squeeze(0).squeeze(0).cpu().numpy() > 0  # Get indices of the segmented region
#     flow_x, flow_y = [], []

#     masked_flow = flow[:, mask_indices]  # Select masked values directly
#     flow_x = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
#     flow_y = masked_flow[1].cpu().numpy() # Extract y-component and convert to list

#     x_coords_masked = x_coords[mask_indices].cpu().numpy()
#     y_coords_masked = y_coords[mask_indices].cpu().numpy()

#     x_coords_mapped = (x_coords_masked + flow_x).round()
#     y_coords_mapped = (y_coords_masked + flow_y).round()

#     x_coords_mapped = np.clip(x_coords_mapped, 0, W_resized - 1)
#     y_coords_mapped = np.clip(y_coords_mapped, 0, H_resized - 1)

#     mapped_mask = np.zeros((H_resized, W_resized), dtype=np.uint8)
#     mapped_mask[y_coords_mapped.astype(int), x_coords_mapped.astype(int)] = 1

#     mapped_mask = torch.tensor(mapped_mask)
#     mapped_mask = mapped_mask.unsqueeze(0).unsqueeze(0)

#     if interpolation == 'bilinear':
#         mapped_mask = F.resize(mapped_mask, size=[H_original, W_original], antialias=False)
#     elif interpolation == 'replicate':
#         mapped_mask = padder.unpad(mapped_mask)

#     mapped_mask = mapped_mask.squeeze(0).squeeze(0)
#     next_mask = mapped_mask.cpu().numpy()

#     if open_trans:
#         kernel = np.ones((kernel_size, kernel_size), np.uint8) 
#         next_mask = cv2.morphologyEx(next_mask, cv2.MORPH_OPEN, kernel, iterations=1) 

#     if close_trans:
#         kernel = np.ones((kernel_size, kernel_size), np.uint8) 
#         next_mask = cv2.morphologyEx(next_mask, cv2.MORPH_CLOSE, kernel, iterations=1) 

#     vis_out = False
#     # For visualization
#     if vis_out:
#         mapped_mask_img = Image.fromarray(next_mask* 255)  # Scale to 0-255 for visibility
#         mapped_mask_img.save("of_bilinear_" + seq +"/"+ str(frame_idx) + "mapped_segmentation_mask.png")

#     return  next_mask

