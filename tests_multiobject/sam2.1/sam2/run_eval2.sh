#!/bin/bash

# Define an array of commands
commands=(
    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --sequences zebrafish1"
    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --sequences zebrafish1"
    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --sequences zebrafish1"
    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --sequences zebrafish1"
    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --sequences zebrafish1"

    "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --sequences zebrafish1"



    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --save_res_path OF0.1.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --save_res_path OF0.15.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --save_res_path OF0.25.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --save_res_path OF0.35.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --save_res_path OF0.5.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --save_res_path OF0.6.csv"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.2 -save_res_path OF0.2.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.3 -save_res_path OF0.3.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.4 -save_res_path OF0.4.csv"


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --direct_comp_to_prev_pred --save_res_path PM0.1.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --direct_comp_to_prev_pred --save_res_path PM0.15.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --direct_comp_to_prev_pred --save_res_path PM0.25.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --direct_comp_to_prev_pred --save_res_path PM0.35.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --direct_comp_to_prev_pred --save_res_path PM0.5.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --direct_comp_to_prev_pred --save_res_path PM0.6.csv"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.2 --direct_comp_to_prev_pred -save_res_path PM0.2.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.3 --direct_comp_to_prev_pred  -save_res_path PM0.3.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.4 --direct_comp_to_prev_pred -save_res_path PM0.4.csv"


    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --oracle"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --oracle --oracle_threshold 5"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --oracle --oracle_threshold 10 --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox "
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --oracle --oracle_threshold 20 --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox "
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --oracle --oracle_threshold 30 --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox "
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --oracle --oracle_threshold 40 --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox "
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --oracle --oracle_threshold 50 --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --no_mask_set_larger_prev_bbox "


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