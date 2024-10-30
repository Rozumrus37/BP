#!/bin/bash

# Define an array of commands
commands=(
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks --memory_stride 7"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks --memory_stride 7 --no_mask_set_full_image"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64  --no_mask_set_full_image"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100  --no_mask_set_full_image"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 3  --no_mask_set_full_image"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 4  --no_mask_set_full_image"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 7  --no_mask_set_full_image"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 10  --no_mask_set_full_image"

)

# Iterate over each command
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    eval "$cmd"  # Run the command
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1  # Exit if any command fails
    fi
done

echo "All commands executed successfully."