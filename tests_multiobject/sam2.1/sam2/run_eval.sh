#!/bin/bash

# Define an array of commands
commands=(
    # "CUDA_VISIBLE_DEVICES=1, python3 generate_output_for_sav.py --sam2_original"
    # "CUDA_VISIBLE_DEVICES=1, python3 generate_output_for_sav.py --sam2_RR --factor 100 --exclude_empty_masks --memory_stride 7"
    # "CUDA_VISIBLE_DEVICES=2, python3 generate_output_for_sav.py --sam2_RR --factor 100 --exclude_empty_masks --memory_stride 1"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 80"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 81"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 82"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 99"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 100"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 101"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 120"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 122"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --exclude_empty_masks"



    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --oracle"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 5"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 10"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 20"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 30"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 40"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 50"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 144 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox --double_memory_bank --uncroped_mask_for_double_MB"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 400 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox --double_memory_bank --uncroped_mask_for_double_MB"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox --double_memory_bank --uncroped_mask_for_double_MB"
    
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 4"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 9"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 16"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 25"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 36" 
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 49" 
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 64"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 81"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100"  
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 4 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 5 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 6 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 8 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 9 --exclude_empty_masks --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py  --use_prev_box --factor 100 --memory_stride 10 --exclude_empty_masks --no_mask_set_larger_prev_bbox"

    
      
      
      
 




    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 16"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 64"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks --memory_stride 7"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks --memory_stride 7 --no_mask_set_larger_prev_bbox"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --exclude_empty_masks"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 16 --no_mask_set_whole_image"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --no_mask_set_whole_image"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --exclude_empty_masks --memory_stride 7 --no_mask_set_whole_image"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 10 --no_mask_set_whole_image"



    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64  --no_mask_set_full_image"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100  --no_mask_set_full_image"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 3  --no_mask_set_full_image"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 4  --no_mask_set_full_image"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 7  --no_mask_set_full_image"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --use_square_box --factor 10  --no_mask_set_full_image"

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