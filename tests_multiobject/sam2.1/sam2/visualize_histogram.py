import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from compute_iou import *
from utilities_eval import *


thresholds = [0.005, 0.01, 0.05, 0.1, 0.20, 0.5]
base_path = "/datagrid/personal/rozumrus/BP_dg/votstval/sequences"
SEQ = []

for folder in os.listdir(base_path):
    current_path = os.path.join(base_path, folder)

    if os.path.isdir(current_path):
        SEQ.append(folder)

# SEQ = ['agility', 'animal', 'ants1', 'bag', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2', 'bolt1', 'book', 'bubble', 'butterfly', 'car1', 'conduction1', 'crabs1', 'dinosaur', 'diver', 'drone1', 'drone_across', 'fernando', 'fish1', 'fish2', 'flamingo1', 'frisbee', 'girl', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'hand2', 'handball1', 'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'kangaroo', 'lamb', 'leaves', 'marathon', 'matrix', 'monkey', 'motocross1', 'nature', 'polo', 'rabbit', 'rabbit2', 'rowing', 'shaking', 'singer2', 'singer3', 'snake', 'soccer1', 'soccer2', 'soldier', 'surfing', 'tennis', 'tiger', 'wheel', 'wiper', 'zebrafish1']

average_segmentation_areas = []
average_bounding_box_areas = []

for seq in SEQ:
    mask_first_frame = get_nth_mask(seq, 0)

    segmentation_area = np.sum(mask_first_frame > 0) / mask_first_frame.size

    min_row, min_col, max_row, max_col = get_bounding_box(mask_first_frame)
    area = abs(min_row-max_row) * abs(min_col-max_col)
    bbox_area = area * 1.0 / mask_first_frame.size

    # if segmentation_area < 0.2 and segmentation_area > 0.1:
    #     print(f"{seq} and {segmentation_area*100} and bbox area: {bbox_area*100}\n")

    average_segmentation_areas.append(segmentation_area)
    average_bounding_box_areas.append(bbox_area)

segmentation_bins = [0] * len(thresholds)
bounding_box_bins = [0] * len(thresholds)

for area in average_segmentation_areas:
    for i, threshold in enumerate(thresholds):
        if area < threshold:
            segmentation_bins[i] += 1
            break

for area in average_bounding_box_areas:
    for i, threshold in enumerate(thresholds):
        if area < threshold:
            bounding_box_bins[i] += 1
            break

labels = ['0%-0.05%', '0.05%-1%', '1%-5%', '5%-10%', '10%-20%', '20%-50%']
x = np.arange(len(labels))
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, segmentation_bins, width, label='Segmentation', color='lightblue')
rects2 = ax.bar(x + width/2, bounding_box_bins, width, label='Bounding Box', color='lightgreen')

ax.set_xlabel('Percentage of total image area')
ax.set_ylabel('Number of sequences')
ax.set_title('Histogram of segmentation and bounding box areas')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('segm_area_vot24stval.png')


print("Histogram saved as 'segmentation_vs_bounding_box_histogram.png' and bin counts saved as 'segmentation_vs_bounding_box_bins.csv'.")

