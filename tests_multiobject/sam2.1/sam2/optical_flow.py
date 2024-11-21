
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os 
from torchvision.models.optical_flow import raft_large
from PIL import Image

from skimage.transform import resize_local_mean

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_and_save(imgs, save_path, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    cnt = 0
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()



def preprocess(batch, H, W):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=((W//8)*8, (H//8)*8)), # 520, 960
        ]
    )
    batch = transforms(batch)
    return batch


def get_mask(frame_path1, frame2_path, segmentation_mask, frame_idx):
    # base_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color"


    frame1_path = frame_path1# os.path.join(base_path, "00000020.jpg")  # Update with your paths
    frame2_path =frame2_path # os.path.join(base_path, "00000021.jpg")

    frame1 = Image.open(frame1_path).convert("RGB")
    frame2 = Image.open(frame2_path).convert("RGB")

    H_original,W_original=frame1.size
    # print("SASSASsssssss", H,W)

    frame1 = T.ToTensor()(frame1).unsqueeze(0) 
    frame2 = T.ToTensor()(frame2).unsqueeze(0) 

    img1_batch = preprocess(frame1, H_original, W_original).to(device)
    img2_batch = preprocess(frame2, H_original, W_original).to(device)

    H_resized, W_resized = (W_original//8)*8, (H_original//8)*8

    # print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")


    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    # print(f"type = {type(list_of_flows)}")
    # print(f"length = {len(list_of_flows)} = number of iterations of the model")


    predicted_flows = list_of_flows[-1]
    # print(f"dtype = {predicted_flows.dtype}")
    # print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    # print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")


    from torchvision.utils import flow_to_image

    flow_imgs = flow_to_image(predicted_flows)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    # plot_and_save(grid, "output_flow1.jpg")


    # segmentation_mask = np.loadtxt('array.txt', dtype=np.float32)

    # # Convert to a PyTorch tensor and move to the appropriate device
    # segmentation_mask = torch.from_numpy(segmentation_mask).to(device)

    # # Ensure the mask is binary (values 0 or 1)
    # segmentation_mask = (segmentation_mask > 0).float()



    flow = predicted_flows[0]  # Shape: (2, H, W)

    # Create a grid of pixel coordinates
    H, W = segmentation_mask.shape

    segmentation_mask = resize_local_mean(segmentation_mask, (H_resized, W_resized))


    y_coords, x_coords = torch.meshgrid(torch.arange(H_resized, device=device), torch.arange(W_resized, device=device), indexing="ij")






    # Filter coordinates using the segmentation mask
    mask_indices = segmentation_mask > 0  # Get indices of the segmented region
    x_coords_masked = x_coords[mask_indices]
    y_coords_masked = y_coords[mask_indices]



    # second_image = np.asarray(Image.open(frame2_path).convert("RGB"))

    # # Extract x and y coordinates from mapped_coords
    # y_mapped, x_mapped = x_coords_masked.cpu().numpy(), y_coords_masked.cpu().numpy()

    # # Plot the second image with overlaid points
    # plt.figure(figsize=(10, 10))
    # plt.imshow(second_image)
    # plt.scatter(y_mapped, x_mapped, c='red', s=5, label="Mapped Points")  # Red points for mapped coordinates
    # plt.axis('off')
    # plt.legend()
    # plt.title("Mapped Points on Second Image")
    # plt.savefig("mapped_points_overlay1.png")  # Save the visualization
    # plt.show()



    # import pdb; pdb.set_trace()

    flow_x, flow_y = [], []


    # Get flow values for the segmented region
    # import pdb; pdb.set_trace()


    for i in range(H_resized):
        for j in range(W_resized):
            if mask_indices[i, j]:
                flow_x.append(flow[0, i, j].cpu().item())
                flow_y.append(flow[1, i, j].cpu().item())

    flow_x = np.array(flow_x)
    flow_y = np.array(flow_y)


    # flow_x = flow[0, mask_indices]  # Flow in the x-direction
    # flow_y = flow[1, mask_indices]  # Flow in the y-direction
    x_coords_masked = x_coords_masked.cpu().numpy()
    y_coords_masked = y_coords_masked.cpu().numpy()

    # Compute new coordinates
    x_coords_mapped = (x_coords_masked + flow_x).round()
    y_coords_mapped = (y_coords_masked + flow_y).round()


    # Ensure mapped coordinates are within bounds
    x_coords_mapped = np.clip(x_coords_mapped, 0, W_resized - 1)
    y_coords_mapped = np.clip(y_coords_mapped, 0, H_resized - 1)

    # Combine mapped coordinates
    # mapped_coords = torch.stack([y_coords_mapped, x_coords_mapped], dim=1)  # Shape: (num_pixels, 2)


    # import matplotlib.pyplot as plt
    # from PIL import Image
    # import numpy as np

    # Load the second image for visualization
    second_image = np.asarray(Image.open(frame2_path).convert("RGB"))

    # Extract x and y coordinates from mapped_coords
    y_mapped, x_mapped = x_coords_mapped, y_coords_mapped


    # H, W = second_image.shape[:2]  # Assuming `second_image` is already loaded as a numpy array

    # Initialize an empty mask
    mapped_mask = np.zeros((H_resized, W_resized), dtype=np.uint8)



    # Populate the mask using mapped coordinates
    # x_mapped = x_coords_mapped.cpu().numpy()
    # y_mapped = y_coords_mapped.cpu().numpy()

    # Set the corresponding pixels in the mask to 1

    # import pdb; pdb.set_trace()

    for i in range(len(y_mapped)):       
        mapped_mask[int(x_mapped[i]), int(y_mapped[i])] = 1



    mapped_mask = torch.tensor(mapped_mask)

    mapped_mask = mapped_mask.unsqueeze(0).unsqueeze(0)

    # Perform bilinear interpolation, resizing to a larger size (e.g., 6x6)
    mapped_mask = torch.nn.functional.interpolate(mapped_mask, size=(W_original, H_original), mode='bilinear', align_corners=False)

    # Remove the batch and channel dimensions to return to 2D
    mapped_mask = mapped_mask.squeeze(0).squeeze(0)


    mapped_mask = mapped_mask.cpu().numpy()

    # # Save the segmentation mask as an image (optional)
    # from PIL import Image
    mapped_mask_img = Image.fromarray(mapped_mask * 255)  # Scale to 0-255 for visibility
    # mapped_mask_img.save("of_ants1/" + str(frame_idx) + "mapped_segmentation_mask.png")

    return mapped_mask

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
# import torchvision.transforms as T
# import os 
# from torchvision.models.optical_flow import raft_large
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"


# def plot_and_save(imgs, save_path, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     cnt = 0
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             img = F.to_pil_image(img.to("cpu"))
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     plt.tight_layout()
#     plt.savefig(save_path)  # Save the figure to the specified path
#     plt.close()



# def preprocess(batch, H, W):
#     transforms = T.Compose(
#         [
#             T.ConvertImageDtype(torch.float32),
#             T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
#             T.Resize(size=((W//8)*8, (H//8)*8)), # 520, 960
#         ]
#     )
#     batch = transforms(batch)
#     return batch


# def get_mask(frame_path1, frame2_path, segmentation_mask, frame_idx):
#     # base_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color"


#     frame1_path = frame_path1# os.path.join(base_path, "00000020.jpg")  # Update with your paths
#     frame2_path =frame2_path # os.path.join(base_path, "00000021.jpg")

#     frame1 = Image.open(frame1_path).convert("RGB")
#     frame2 = Image.open(frame2_path).convert("RGB")

#     H,W=frame1.size
#     print("SASSASsssssss", H,W)

#     frame1 = T.ToTensor()(frame1).unsqueeze(0) 
#     frame2 = T.ToTensor()(frame2).unsqueeze(0) 

#     img1_batch = preprocess(frame1, H, W).to(device)
#     img2_batch = preprocess(frame2, H, W).to(device)

#     print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")


#     model = raft_large(pretrained=True, progress=False).to(device)
#     model = model.eval()

#     list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
#     print(f"type = {type(list_of_flows)}")
#     print(f"length = {len(list_of_flows)} = number of iterations of the model")


#     predicted_flows = list_of_flows[-1]
#     print(f"dtype = {predicted_flows.dtype}")
#     print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
#     print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")


#     from torchvision.utils import flow_to_image

#     flow_imgs = flow_to_image(predicted_flows)

#     # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
#     img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

#     grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
#     # plot_and_save(grid, "output_flow1.jpg")


#     # segmentation_mask = np.loadtxt('array.txt', dtype=np.float32)

#     # # Convert to a PyTorch tensor and move to the appropriate device
#     # segmentation_mask = torch.from_numpy(segmentation_mask).to(device)

#     # # Ensure the mask is binary (values 0 or 1)
#     # segmentation_mask = (segmentation_mask > 0).float()



#     flow = predicted_flows[0]  # Shape: (2, H, W)

#     # Create a grid of pixel coordinates
#     H, W = segmentation_mask.shape
#     y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")





#     # Filter coordinates using the segmentation mask
#     mask_indices = segmentation_mask > 0  # Get indices of the segmented region
#     x_coords_masked = x_coords[mask_indices]
#     y_coords_masked = y_coords[mask_indices]



#     # second_image = np.asarray(Image.open(frame2_path).convert("RGB"))

#     # # Extract x and y coordinates from mapped_coords
#     # y_mapped, x_mapped = x_coords_masked.cpu().numpy(), y_coords_masked.cpu().numpy()

#     # # Plot the second image with overlaid points
#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(second_image)
#     # plt.scatter(y_mapped, x_mapped, c='red', s=5, label="Mapped Points")  # Red points for mapped coordinates
#     # plt.axis('off')
#     # plt.legend()
#     # plt.title("Mapped Points on Second Image")
#     # plt.savefig("mapped_points_overlay1.png")  # Save the visualization
#     # plt.show()



#     # import pdb; pdb.set_trace()

#     flow_x, flow_y = [], []


#     # Get flow values for the segmented region
#     # import pdb; pdb.set_trace()


#     for i in range(H):
#         for j in range(W):
#             if mask_indices[i, j]:
#                 flow_x.append(flow[0, i, j].cpu().item())
#                 flow_y.append(flow[1, i, j].cpu().item())

#     flow_x = np.array(flow_x)
#     flow_y = np.array(flow_y)


#     # flow_x = flow[0, mask_indices]  # Flow in the x-direction
#     # flow_y = flow[1, mask_indices]  # Flow in the y-direction
#     x_coords_masked = x_coords_masked.cpu().numpy()
#     y_coords_masked = y_coords_masked.cpu().numpy()

#     # Compute new coordinates
#     x_coords_mapped = (x_coords_masked + flow_x).round()
#     y_coords_mapped = (y_coords_masked + flow_y).round()


#     # Ensure mapped coordinates are within bounds
#     x_coords_mapped = np.clip(x_coords_mapped, 0, W - 1)
#     y_coords_mapped = np.clip(y_coords_mapped, 0, H - 1)

#     # Combine mapped coordinates
#     # mapped_coords = torch.stack([y_coords_mapped, x_coords_mapped], dim=1)  # Shape: (num_pixels, 2)


#     # import matplotlib.pyplot as plt
#     # from PIL import Image
#     # import numpy as np

#     # Load the second image for visualization
#     second_image = np.asarray(Image.open(frame2_path).convert("RGB"))

#     # Extract x and y coordinates from mapped_coords
#     y_mapped, x_mapped = x_coords_mapped, y_coords_mapped


#     # H, W = second_image.shape[:2]  # Assuming `second_image` is already loaded as a numpy array

#     # Initialize an empty mask
#     mapped_mask = np.zeros((H, W), dtype=np.uint8)

#     # Populate the mask using mapped coordinates
#     # x_mapped = x_coords_mapped.cpu().numpy()
#     # y_mapped = y_coords_mapped.cpu().numpy()

#     # Set the corresponding pixels in the mask to 1

#     # import pdb; pdb.set_trace()

#     for i in range(len(y_mapped)):       
#         mapped_mask[int(x_mapped[i]), int(y_mapped[i])] = 1

#     # # Save the segmentation mask as an image (optional)
#     # from PIL import Image
#     mapped_mask_img = Image.fromarray(mapped_mask * 255)  # Scale to 0-255 for visibility
#     mapped_mask_img.save("of_ants1/" + str(frame_idx) + "mapped_segmentation_mask.png")

#     return mapped_mask

# # Plot the second image with overlaid points
# plt.figure(figsize=(10, 10))
# plt.imshow(second_image)
# plt.scatter(y_mapped, x_mapped, c='red', s=5, label="Mapped Points")  # Red points for mapped coordinates
# plt.axis('off')
# plt.legend()
# plt.title("Mapped Points on Second Image")
# plt.savefig("mapped_points_overlay.png")  # Save the visualization
# plt.show()



# # import pdb; pdb.set_trace()

# print(f"Mapped coordinates in the second image: {mapped_coords}")









