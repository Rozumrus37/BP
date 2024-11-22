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

device = "cuda" if torch.cuda.is_available() else "cpu"


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



def preprocess(batch, H, W):
    transforms = T.Compose(
        [
            # T.ConvertImageDtype(torch.float32),
            # T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(H, W)), # 520, 960
        ]
    )
    batch = transforms(batch)
    return batch

def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def get_mask(prev_frame_path, next_frame_path, prev_segmentation_mask, frame_idx=None, vis_out=None, seq=None):
    # return prev_segmentation_mask
    # next_frame_path = "shifted_by_10_frame1.jpg"
    # prev_frame_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color/00000001.jpg"


    frame1 = read_image(prev_frame_path) 
    frame2 = read_image(next_frame_path) 


    img1_batch = torch.stack([frame1])
    img2_batch = torch.stack([frame2])

    from torchvision.models.optical_flow import Raft_Large_Weights

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    padder = InputPadder(frame1.shape)

    def preprocess(img1_batch, img2_batch):
        
        img1_batch, img2_batch = padder.pad(img1_batch, img2_batch)

        # img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
        # img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
        return transforms(img1_batch, img2_batch)


    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

    from torchvision.models.optical_flow import raft_large

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    print(f"type = {type(list_of_flows)}")
    print(f"length = {len(list_of_flows)} = number of iterations of the model")

    predicted_flows = list_of_flows[-1]
    print(f"dtype = {predicted_flows.dtype}")
    print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")


    W_original, H_original = Image.open(prev_frame_path).size



    H_resized, W_resized = img1_batch.shape[2], img1_batch.shape[3] #(H_original//8)*8, (W_original//8)*8

    predicted_flows = list_of_flows[-1] # obtain last of 12 iteration
    flow = predicted_flows[0]  # Shape: (2, H_resized, W_resized)
   
    prev_segmentation_mask = padder.pad(torch.tensor(prev_segmentation_mask).unsqueeze(0).unsqueeze(0))[0] #resize_local_mean(prev_segmentation_mask, (H_resized, W_resized)) # resize to match resized prev_frame, next_frame
    y_coords, x_coords = torch.meshgrid(torch.arange(H_resized, device=device), torch.arange(W_resized, device=device), indexing="ij")

   

    mask_indices = prev_segmentation_mask.squeeze(0).squeeze(0).cpu().numpy() > 0  # Get indices of the segmented region
    flow_x, flow_y = [], []


    masked_flow = flow[:, mask_indices]  # Select masked values directly
    flow_x = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
    flow_y = masked_flow[1].cpu().numpy() # Extract y-component and convert to list



    # import pdb; pdb.set_trace()

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
    # mapped_mask = torch.nn.functional.interpolate(mapped_mask, size=(H_original, W_original), mode='bilinear', align_corners=False)
    mapped_mask = mapped_mask.squeeze(0).squeeze(0)
    next_mask = mapped_mask.cpu().numpy()

    # # # For visualization
    # # next_mask = prev_segmentation_mask
    mapped_mask_img = Image.fromarray(next_mask* 255)  # Scale to 0-255 for visibility
    mapped_mask_img.save("of_" + seq +"/"+ str(frame_idx) + "z_mapped_segmentation_mask.png")

    return next_mask


# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
# import torchvision.transforms as T
# import os 
# from torchvision.models.optical_flow import raft_large
# from PIL import Image
# from torchvision.utils import flow_to_image
# from skimage.transform import resize_local_mean

# device = "cuda" if torch.cuda.is_available() else "cpu"


# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """
#     def __init__(self, dims, mode='sintel'):
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
#         pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
#         if mode == 'sintel':
#             self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
#         else:
#             self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

#     def pad(self, *inputs):
#         return [torch.nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

#     def unpad(self,x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]



# def preprocess(batch, H, W):
#     transforms = T.Compose(
#         [
#             # T.ConvertImageDtype(torch.float32),
#             # T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
#             T.Resize(size=(H, W)), # 520, 960
#         ]
#     )
#     batch = transforms(batch)
#     return batch

# def load_image(imfile):
#     img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(device)


# def get_mask(prev_frame_path, next_frame_path, prev_segmentation_mask, frame_idx=None, vis_out=None, seq=None):
#     # return prev_segmentation_mask
#     next_frame_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color/00000002.jpg" #"shifted_by_10_frame1.jpg"
#     prev_frame_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color/00000001.jpg"

#     frame1 = load_image(prev_frame_path) #Image.open(prev_frame_path).convert("RGB")
#     frame2 = load_image(next_frame_path) #Image.open(next_frame_path).convert("RGB")

#     W_original, H_original = Image.open(prev_frame_path).size

#     padder = InputPadder(frame1.shape)
#     frame1, frame2 = padder.pad(frame1, frame2)

#     # import pdb; pdb.set_trace()

#     H_resized, W_resized = frame1[0][0].shape[0], frame1[0][0].shape[1] #(H_original//8)*8, (W_original//8)*8

#     # frame1 = T.ToTensor()(frame1).unsqueeze(0) 
#     # frame2 = T.ToTensor()(frame2).unsqueeze(0) 

#     # import pdb; pdb.set_trace()

#     # img1_batch = preprocess(frame1, H_resized, W_resized).to(device)
#     # img2_batch = preprocess(frame2, H_resized, W_resized).to(device)



#     model = raft_large(pretrained=True, progress=False).to(device)
#     model = model.eval()

#     # extracting flow
#     list_of_flows = model(frame1.to(device), frame2.to(device))
#     predicted_flows = list_of_flows[-1] # obtain last of 12 iteration
#     flow = predicted_flows[0]  # Shape: (2, H_resized, W_resized)
#     flow_imgs = flow_to_image(predicted_flows) # image of the flow
    
#     import pdb; pdb.set_trace()

#     prev_segmentation_mask = padder.pad(torch.tensor(prev_segmentation_mask).unsqueeze(0).unsqueeze(0))[0] #resize_local_mean(prev_segmentation_mask, (H_resized, W_resized)) # resize to match resized prev_frame, next_frame
#     y_coords, x_coords = torch.meshgrid(torch.arange(H_resized, device=device), torch.arange(W_resized, device=device), indexing="ij")

   

#     mask_indices = prev_segmentation_mask.squeeze(0).squeeze(0).cpu().numpy() > 0  # Get indices of the segmented region
#     flow_x, flow_y = [], []

#     # mask_all_trues = np.ones((H_resized, W_resized))
#     # masked_flow = flow[:, mask_all_trues]  # Select masked values directly
#     # flow_x_all = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
#     # flow_y_all = masked_flow[1].cpu().numpy() # Extract y-component and convert to list

#     masked_flow = flow[:, mask_indices]  # Select masked values directly
#     flow_x = masked_flow[0].cpu().numpy()  # Extract x-component and convert to list
#     flow_y = masked_flow[1].cpu().numpy() # Extract y-component and convert to list

#     # u = flow_x_all / 
#     # v = flow_y_all

#     rad = np.sqrt(np.square(flow_x) + np.square(flow_y))
#     rad_max = np.max(rad)
#     epsilon = 1e-5
#     u = flow_x / (rad_max + epsilon)
#     v = flow_y / (rad_max + epsilon)

#     import pdb; pdb.set_trace()

#     # u = flow_x  / W_resized
#     # v = flow_y / H_resized



#     x_coords_masked = x_coords[mask_indices].cpu().numpy()
#     y_coords_masked = y_coords[mask_indices].cpu().numpy()

#     x_coords_mapped = (x_coords_masked + u).round()
#     y_coords_mapped = (y_coords_masked + v).round()

#     x_coords_mapped = np.clip(x_coords_mapped, 0, W_resized - 1)
#     y_coords_mapped = np.clip(y_coords_mapped, 0, H_resized - 1)


#     mapped_mask = np.zeros((H_resized, W_resized), dtype=np.uint8)
#     mapped_mask[y_coords_mapped.astype(int), x_coords_mapped.astype(int)] = 1

#     mapped_mask = torch.tensor(mapped_mask)
#     mapped_mask = mapped_mask.unsqueeze(0).unsqueeze(0)

#     mapped_mask = padder.unpad(mapped_mask)
#     # mapped_mask = torch.nn.functional.interpolate(mapped_mask, size=(H_original, W_original), mode='bilinear', align_corners=False)
#     mapped_mask = mapped_mask.squeeze(0).squeeze(0)
#     next_mask = mapped_mask.cpu().numpy()

#     # # # For visualization
#     # # next_mask = prev_segmentation_mask
#     mapped_mask_img = Image.fromarray(next_mask* 255)  # Scale to 0-255 for visibility
#     mapped_mask_img.save("of_" + seq +"/"+ str(frame_idx) + "z_mapped_segmentation_mask.png")

#     return next_mask


# get_mask(None, None, None)

# # shaking,singer3,marathon,hand2,wheel,ants1,zebrafish1,soldier,tennis,wheel,drone1

