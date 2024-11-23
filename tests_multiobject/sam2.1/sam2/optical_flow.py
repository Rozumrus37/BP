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

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [torch.nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def preprocess(img1_batch, img2_batch, padder):
    img1_batch, img2_batch = padder.pad(img1_batch, img2_batch)
    # img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    # img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)

def get_mask(prev_frame_path, next_frame_path, prev_segmentation_mask, frame_idx=None, seq=None):
    # return prev_segmentation_mask
    # next_frame_path = "shifted_by_10_frame1.jpg"
    # prev_frame_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color/00000001.jpg"

    frame1 = read_image(prev_frame_path) 
    frame2 = read_image(next_frame_path) 

    img1_batch = torch.stack([frame1])
    img2_batch = torch.stack([frame2])

    padder = InputPadder(frame1.shape)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch, padder)

    # print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    # print(f"type = {type(list_of_flows)}")
    # print(f"length = {len(list_of_flows)} = number of iterations of the model")

    predicted_flows = list_of_flows[-1]
    # print(f"dtype = {predicted_flows.dtype}")
    # print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    # print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

    W_original, H_original = Image.open(prev_frame_path).size

    H_resized, W_resized = img1_batch.shape[2], img1_batch.shape[3] 

    predicted_flows = list_of_flows[-1] # obtain last of 12 iteration
    flow = predicted_flows[0]  # Shape: (2, H_resized, W_resized)
   
    prev_segmentation_mask = padder.pad(torch.tensor(prev_segmentation_mask).unsqueeze(0).unsqueeze(0))[0] 
    y_coords, x_coords = torch.meshgrid(torch.arange(H_resized, device=device), torch.arange(W_resized, device=device), indexing="ij")
   
    mask_indices = prev_segmentation_mask.squeeze(0).squeeze(0).cpu().numpy() > 0  # Get indices of the segmented region
    flow_x, flow_y = [], []

    masked_flow = flow[:, mask_indices]  # Select masked values directly
    flow_x = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
    flow_y = masked_flow[1].cpu().numpy() # Extract y-component and convert to list

    x_coords_masked = x_coords[mask_indices].cpu().numpy()
    y_coords_masked = y_coords[mask_indices].cpu().numpy()

    x_coords_mapped = (x_coords_masked + flow_x).round()
    y_coords_mapped = (y_coords_masked + flow_y).round()

    x_coords_mapped = np.clip(x_coords_mapped, 0, W_resized - 1)
    y_coords_mapped = np.clip(y_coords_mapped, 0, H_resized - 1)

    mapped_mask = np.zeros((H_resized, W_resized), dtype=np.uint8)
    mapped_mask[y_coords_mapped.astype(int), x_coords_mapped.astype(int)] = 1

    mapped_mask = torch.tensor(mapped_mask)
    mapped_mask = mapped_mask.unsqueeze(0).unsqueeze(0)

    mapped_mask = padder.unpad(mapped_mask)
    mapped_mask = mapped_mask.squeeze(0).squeeze(0)
    next_mask = mapped_mask.cpu().numpy()

    vis_out = False
    # For visualization
    if vis_out:
        mapped_mask_img = Image.fromarray(next_mask* 255)  # Scale to 0-255 for visibility
        mapped_mask_img.save("of_" + seq +"/"+ str(frame_idx) + "z_mapped_segmentation_mask.png")

    return next_mask
