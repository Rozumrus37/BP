#!/bin/bash

# Define an array of commands
commands=(
    # "CUDA_VISIBLE_DEVICES=1, python3 generate_output_for_sav.py --sam2_original"
    # "CUDA_VISIBLE_DEVICES=1, python3 generate_output_for_sav.py --sam2_RR --factor 100 --exclude_empty_masks --memory_stride 7"
    # "CUDA_VISIBLE_DEVICES=2, python3 generate_output_for_sav.py --sam2_RR --factor 100 --exclude_empty_masks --memory_stride 1"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 80"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 81"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 82"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 99"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 100"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 101"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 120"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --use_prev_box --factor 122"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py  --exclude_empty_masks"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --save_res_path BOF_025.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --interpolation replicate --save_res_path of_repl_af_0.1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --interpolation bilinear --save_res_path of_bilinear_af_0.1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.1.csv"


    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --interpolation replicate --save_res_path of_repl_af_0.15.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --interpolation bilinear --save_res_path of_bilinear_af_0.15.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.15.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.15.csv"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --interpolation replicate --save_res_path of_repl_af_0.25.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --interpolation bilinear --save_res_path of_bilinear_af_0.25.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.25.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.25.csv"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --interpolation replicate --save_res_path of_repl_af_0.35.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --interpolation bilinear --save_res_path of_bilinear_af_0.35.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.35.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.35.csv"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --interpolation replicate --save_res_path of_repl_af_0.5.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --interpolation bilinear --save_res_path of_bilinear_af_0.5.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.5.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.5.csv"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --interpolation replicate --save_res_path of_repl_af_0.6.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --interpolation bilinear --save_res_path of_bilinear_af_0.6.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.6.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.6.csv"


    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation replicate --save_res_path of_repl_af_1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation bilinear --save_res_path of_bilinear_af_1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation replicate --save_res_path bof_repl_af_1.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_1.csv"

    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation replicate --close_trans --open_trans --save_res_path of_repl_af_1_co_trans.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation bilinear --close_trans --open_trans --save_res_path of_bilinear_af_1_co_trans.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation replicate --close_trans --open_trans  --save_res_path bof_repl_af_1_co_trans.csv"
    "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation bilinear --close_trans --open_trans  --save_res_path bof_bilinear_af_1_co_trans.csv"


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --save_res_path BOF_05.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --save_res_path BOF_01.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --save_res_path BOF_015.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --save_res_path BOF_06.csv"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --save_res_path OF_06.csv"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6"

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