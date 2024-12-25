#!/bin/bash

# Define an array of commands
commands=(


    "CUDA_VISIBLE_DEVICES=1, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.2 --thr_Amask_to_Abb 0.1 --save_res_path thrIoU0.2_thrAmask_0.1.csv" 
    "CUDA_VISIBLE_DEVICES=1, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.2 --thr_Amask_to_Abb 0.2 --save_res_path thrIoU0.2_thrAmask_0.2.csv" 
    "CUDA_VISIBLE_DEVICES=1, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.2 --thr_Amask_to_Abb 0.3 --save_res_path thrIoU0.2_thrAmask_0.3.csv" 
    "CUDA_VISIBLE_DEVICES=1, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.2 --thr_Amask_to_Abb 0.5 --save_res_path thrIoU0.2_thrAmask_0.5.csv" 
    "CUDA_VISIBLE_DEVICES=1, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.2 --thr_Amask_to_Abb 0.7 --save_res_path thrIoU0.2_thrAmask_0.7.csv" 


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 4 --min_box_factor 128  --save_res_path with128_factor4_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 9 --min_box_factor 128  --save_res_path with128_factor9_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 16 --min_box_factor 128  --save_res_path with128_factor16_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 25 --min_box_factor 128  --save_res_path with128_factor25_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 36 --min_box_factor 128  --save_res_path with128_factor36_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 49 --min_box_factor 128  --save_res_path with128_factor49_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 64 --min_box_factor 128  --save_res_path with128_factor64_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 81 --min_box_factor 128  --save_res_path with128_factor81_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 121 --min_box_factor 128  --save_res_path with128_factor121_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 144 --min_box_factor 128  --save_res_path with128_factor144_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 169 --min_box_factor 128  --save_res_path with128_factor169_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 400 --min_box_factor 128  --save_res_path with128_factor400_duda.csv"


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 4 --save_res_path with256_factor4.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 9  --save_res_path with256_factor9.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 16  --save_res_path with256_factor16.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 25  --save_res_path with128_factor25_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 36  --save_res_path with128_factor36_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 49  --save_res_path with128_factor49_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 64  --save_res_path with128_factor64_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 81  --save_res_path with128_factor81_duda.csv"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100  --save_res_path with128_factor100_duda.csv"

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --save_res_path with128_factor100_RR7_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 6 --exclude_empty_masks --save_res_path with128_factor100_RR6_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 5 --exclude_empty_masks --save_res_path with128_factor100_RR5_duda.csv"


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 121  --save_res_path with128_factor121_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 144  --save_res_path with128_factor144_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 169  --save_res_path with128_factor169_duda.csv"
    

    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 196  --save_res_path with256_factor196_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 225  --save_res_path with256_factor225_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 256  --save_res_path with256_factor256_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 289  --save_res_path with256_factor289_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 400  --save_res_path with256_factor400_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 625  --save_res_path with256_factor625_duda.csv"


    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 4 --save_res_path with256_factor100_MS4_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 5 --save_res_path with256_factor100_MS5_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 6 --save_res_path with256_factor100_MS6_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 7 --save_res_path with256_factor100_MS7_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 8 --save_res_path with256_factor100_MS8_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 9 --save_res_path with256_factor100_MS9_duda.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 10 --save_res_path with256_factor100_MS10_duda.csv"



    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 1 --save_res_path MS_1.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 2 --save_res_path MS_2.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 3 --save_res_path MS_3.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 4 --save_res_path MS_4.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 5 --save_res_path MS_5.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 6 --save_res_path MS_6.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 7 --save_res_path MS_7.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 8 --save_res_path MS_8.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 9 --save_res_path MS_9.csv"
    # "CUDA_VISIBLE_DEVICES=2, python3 cropped_tracking.py --memory_stride 10 --save_res_path MS_10.csv"


    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --sequences zebrafish1"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --sequences zebrafish1"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --sequences zebrafish1"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --sequences zebrafish1"
    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --sequences zebrafish1"

    # "CUDA_VISIBLE_DEVICES=0, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --sequences zebrafish1"



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