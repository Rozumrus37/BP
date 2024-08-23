import vot_utils
import sys
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = False #True
    torch.backends.cudnn.allow_tf32 = False #True

from sam2.build_sam import build_sam2_camera_predictor
import time


sam2_checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"



class NCCTracker2(object):

    def __init__(self, image, mask, cnt):
        non_zero_indices = np.nonzero(mask)
        coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
        points = []

        for (y, x) in coordinates:
            points.append([x, y])

        if cnt == 1:
            points = np.array([[1200, 660]], dtype=np.float32)
        else:
            points = np.array([[210, 340]], dtype=np.float32)

        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

        self.predictor.load_first_frame(image)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started

        points = np.argwhere(mask == 1)
        # points = np.array(self.valid_indices(mask))
        # print("LEN", len(points))

        # print(points)

        # # # Randomly select 3 points
        points = points[np.random.choice(points.shape[0], 5, replace=False)]
        points = points[:, ::-1]

        labels = np.array(np.ones(len(points)), np.int32)
        # _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
        #     frame_idx=ann_frame_idx,
        #     obj_id=ann_obj_id,
        #     points=np.array(points, dtype=np.float32),
        #     labels=labels,
        # )

        mask = self.pad_mask_to_image_size(mask, (image.shape[0], image.shape[1]))


        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=mask
        )



        #  _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
        #     frame_idx=ann_frame_idx,
        #     obj_id=ann_obj_id,
        #     mask=np.array(mask, dtype=np.float32),
        # )


    def valid_indices(self, binary_mask):
        # Get the shape of the binary mask
        rows, cols = binary_mask.shape

        # Initialize a list to store the valid indices
        valid_indices = []

        # Iterate over each point in the binary mask
        for i in range(3, rows - 3):
            for j in range(3, cols - 3):
                # Check if the current point is 1 and has no neighboring 0's
                if (binary_mask[i, j] == 1 and
                    binary_mask[i-1, j] == 1 and
                    binary_mask[i+1, j] == 1 and
                    binary_mask[i, j-1] == 1 and
                    binary_mask[i, j+1] == 1 and

                    binary_mask[i-2, j] == 1 and
                    binary_mask[i+2, j] == 1 and
                    binary_mask[i, j-2] == 1 and
                    binary_mask[i, j+2] == 1 and
                     
                    binary_mask[i-3, j] == 1 and
                    binary_mask[i+3, j] == 1 and
                    binary_mask[i, j-3] == 1 and
                    binary_mask[i, j+3] == 1):

                    valid_indices.append([i, j])

        return valid_indices


    def pad_mask_to_image_size(self, mask, image_size):
        # Get the dimensions of the mask and the desired image size
        mask_rows, mask_cols = mask.shape
        image_rows, image_cols = image_size
        
        # Calculate the amount of padding needed for rows and columns
        pad_rows = image_rows - mask_rows
        pad_cols = image_cols - mask_cols

        # Pad the mask with zeros to the right and bottom
        padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        return padded_mask

    def track(self, image,c):
        # if len(image.shape) == 2:  # Grayscale image
        #     image = np.expand_dims(image, axis=-1)  # Add a ch annel dimension
        #     image = torch.from_numpy(image).permute(2, 0, 1).float()


        out_obj_ids, out_mask_logits = self.predictor.track(image)

        # non_zero_indices = np.nonzero(mask)
        # coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
        # print("LIST", len(coordinates), coordinates)


        #print("LENNNN", (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], len((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]))

        # print("YES")
        # print(out_mask_logits.shape, len(out_mask_logits))
        # print((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])

        # print("SUM", np.sum((out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]))

        # print(len((out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8) ))
        # print(out_mask_logits[0] > 0.0)

        # print(out_mask_logits.shape, out_obj_ids)

        # print("asdasdsadasdsadasdas" + str(out_mask_logits))

        # print((out_mask_logits[0] > 0.0).cpu().numpy()[0] )

        #self.vis_segm(image, (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0], "/home.stud/rozumrus/BP/tests_multiobject/out_sam2/" + str(c) + "_sam2_out.png")

        print("SHAPE", (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0].shape, image.shape)

        return (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0] #out_mask_logits

    def _rect_from_mask(self, mask):
        '''
        create an axis-aligned rectangle from a given binary mask
        mask in created as a minimal rectangle containing all non-zero pixels
        '''
        x_ = np.sum(mask, axis=0)
        y_ = np.sum(mask, axis=1)
        x0 = np.min(np.nonzero(x_))
        x1 = np.max(np.nonzero(x_))
        y0 = np.min(np.nonzero(y_))
        y1 = np.max(np.nonzero(y_))
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    def visualize_bbox_on_image(self, image, mask, output_file):


        # Assuming mask is a binary mask (same size as image) where the object is located
        # Create a bounding box (bbox) from the mask
        region = self._rect_from_mask(mask)
        
        # Creating a vot_utils.Rectangle object (assume vot_utils.Rectangle is defined)
        region = vot_utils.Rectangle(region[0], region[1], region[2], region[3])
        
        # Calculate window size based on the bbox
        self.window = max(region.width, region.height) * 2

        # Calculate the bounding box coordinates
        left = max(region.x, 0)
        top = max(region.y, 0)
        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        # Draw the bounding box on the original image
        image_with_bbox = image.copy()
        cv2.rectangle(image_with_bbox, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        # Optionally, crop the template region (not required for visualization)
        self.template = image[int(top):int(bottom), int(left):int(right)]
        
        # Save the image with the bounding box
        cv2.imwrite(output_file, image_with_bbox)

        print(f"Image with bbox saved to {output_file}")

    def vis_segm(self, image, mask, output_file):
        non_zero_indices = np.nonzero(mask)
        coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
        # print("LIST", len(coordinates), coordinates)
        
        # Make a copy of the image to draw on
        image_with_dots = image.copy()
        
        # Draw black dots on the image at each coordinate
        for (y, x) in coordinates:
            cv2.circle(image_with_dots, (x, y), radius=2, color=(0, 0, 0), thickness=-1)
        
        # Save the resulting image
        cv2.imwrite(output_file, image_with_dots)
        print(f"Image with black dots saved to {output_file}")



handle = vot_utils.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()

#image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

trackers = [] #NCCTracker2(image, object) for object in objects]

cnt = 0

for object in objects:
    trackers.append(NCCTracker2(image, object, cnt))
    cnt+=1

# print("LENGTH", len(objects), len(trackers))
# print(objects[0].shape)
# print(objects[1].shape)
# print(objects[2].shape)

c=0

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    out=[]
    for tracker in trackers:
        out.append(tracker.track(image, c))
        c+=1

    c+=1
    handle.report(out) #[tracker.track(image, c) for tracker in trackers])


# ml PyTorch/2.3.0-foss-2023a-CUDA-12.3.0
# ml OpenCV/4.9.0-foss-2023a-CUDA-12.3.0-contrib
# ml  torchvision/0.18.0-foss-2023a-CUDA-12.3.0 //torchvision/0.18.0-foss-2023b-CUDA-12.4.0
# ml Hydra/1.3.2-GCCcore-12.3.0  

