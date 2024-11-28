import cv2 
  
import numpy as np 

import matplotlib.pyplot as plt

# read the image 
img = cv2.imread("of_hand2/104z_mapped_segmentation_mask.png", 0) 
  
# # binarize the image 
# binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
  
# define the kernel 
kernel = np.ones((5, 5), np.uint8) 
  
# opening the image 
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1) 
  
# print the output 
# plt.imshow(closing, cmap='gray') 

cv2.imwrite("denoised_bigger_kernel.png", closing)
