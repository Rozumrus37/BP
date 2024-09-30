import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig('example' + str(i) + '_46.png')

 

checkpoint = "/home.stud/rozumrus/BP/tests_multiobject/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
imagefile = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/book/color/00000001.jpg"


predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

image = Image.open(imagefile)
image = np.array(image.convert("RGB"))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)


def get_logits():
    input_point = np.array([[244, 87], [238, 72], [289, 121], [272, 38]])
    input_label = np.array([1, 1, 0, 0])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )


    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]


    show_masks(image, np.expand_dims(masks[0], axis=0), scores, borders=True)

    # print("Masks, scores, logits: ", masks.shape, scores.shape, logits.shape)
    print("IoU initial: ", scores)

    return logits

logits = get_logits()
    
def get_masks_sam2(logits):
    masks, scores, logits = predictor.predict(
        multimask_output=True,
        mask_input=logits
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    return scores, masks, logits


scores, masks, logits = get_masks_sam2(np.expand_dims(logits[0], axis=0))
print("IoU when the mask was passed is: ", scores)
show_masks(image, np.expand_dims(masks[0], axis=0), scores, borders=True)


# show_masks(image, masks, scores, borders=True)

# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)




# def pad_mask_to_image_size(mask, image_size):
#     # Get the dimensions of the mask and the desired image size
#     mask_rows, mask_cols = mask.shape
#     image_rows, image_cols = image_size
    
#     # Calculate the amount of padding needed for rows and columns
#     pad_rows = image_rows - mask_rows
#     pad_cols = image_cols - mask_cols

#     # Pad the mask with zeros to the right and bottom
#     padded_mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

#     return padded_mask





