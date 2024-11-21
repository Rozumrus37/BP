import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read reference image
img1 = cv2.imread("/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color/00000001.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Read image to be aligned
img2 = cv2.imread("show_saved_dino/00000041_first.jpg", cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

img1_display = cv2.drawKeypoints(img1, keypoints1,
                                 outImage = np.array([]),
                                 color = (255, 0, 0),
                                 flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img2_display = cv2.drawKeypoints(img2, keypoints2,
                                 outImage = np.array([]),
                                 color = (255, 0, 0),
                                 flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.imshow(img1_display)

cv2.imwrite("40.png", img1_display)
cv2.imwrite("41.png", img2_display)




# import cv2
# import numpy as np
# from scipy.optimize import minimize

# # Load the two images
# image1 = cv2.imread('/datagrid/personal/rozumrus/BP_dg/sam2.1_output/hand2/19.png', 0)  # Load as grayscale
# image2 = cv2.imread('/datagrid/personal/rozumrus/BP_dg/sam2.1_output/hand2/0.png', 0)

# # Ensure both images have the same size
# image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))


# def dice_coefficient(img1, img2):
#     intersection = np.sum(img1 & img2)
#     total = np.sum(img1) + np.sum(img2)
#     return 2 * intersection / total if total > 0 else 0

# def transform_image(img, tx, ty, theta, scale):
#     rows, cols = img.shape
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, scale)
#     M[:, 2] += [tx, ty]  # Add translation
#     transformed = cv2.warpAffine(img, M, (cols, rows))
#     return transformed

# # Define the objective function
# def objective_function(params):
#     tx, ty, theta, scale = params
#     transformed = transform_image(image2, tx, ty, theta, scale)
#     return -dice_coefficient(image1, transformed)  # Negative because we want to maximize

# # Initial guess: no transformation
# initial_guess = [0, 0, 0, 1]  # tx, ty, theta, scale

# # Bounds for the parameters
# bounds = [(-50, 50), (-50, 50), (-180, 180), (0.5, 2.0)]  # Adjust as needed

# # Perform optimization
# result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

# # Extract optimized parameters
# optimized_tx, optimized_ty, optimized_theta, optimized_scale = result.x
# print("Optimized Parameters:", result.x)

# aligned_image = transform_image(image2, optimized_tx, optimized_ty, optimized_theta, optimized_scale)

# # Visualize the result
# # cv2.imshow("Aligned Image", aligned_image)
# # cv2.imshow("Original Image", image1)


# # import cv2
# # import numpy as np
# # from scipy.optimize import minimize

# # # Load the two images
# # image1 = cv2.imread('saved_dino/00000041_third.jpg')  # Load as grayscale
# # image2 = cv2.imread('saved_dino/00000040_first.jpg')

# # # Ensure both images have the same size
# # image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))


# # def dice_coefficient(img1, img2):
# #     intersection = np.sum(img1 & img2)
# #     total = np.sum(img1) + np.sum(img2)
# #     return 2 * intersection / total if total > 0 else 0

# # def transform_image(img, tx, ty, theta, scale):
# #     rows, cols = img.shape
# #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, scale)
# #     M[:, 2] += [tx, ty]  # Add translation
# #     transformed = cv2.warpAffine(img, M, (cols, rows))
# #     return transformed

# # # Define the objective function
# # def objective_function(params):
# #     tx, ty, theta, scale = params
# #     transformed = transform_image(image2, tx, ty, theta, scale)
# #     return -dice_coefficient(image1, transformed)  # Negative because we want to maximize

# # # Initial guess: no transformation
# # initial_guess = [0, 0, 0, 1]  # tx, ty, theta, scale

# # # Bounds for the parameters
# # bounds = [(-50, 50), (-50, 50), (-180, 180), (0.5, 2.0)]  # Adjust as needed

# # # Perform optimization
# # result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

# # # Extract optimized parameters
# # optimized_tx, optimized_ty, optimized_theta, optimized_scale = result.x
# # print("Optimized Parameters:", result.x)

# # aligned_image = transform_image(image2, optimized_tx, optimized_ty, optimized_theta, optimized_scale)

# # # # Visualize the result
# # # cv2.imshow("Aligned Image", aligned_image)
# # # cv2.imshow("Original Image", image1)