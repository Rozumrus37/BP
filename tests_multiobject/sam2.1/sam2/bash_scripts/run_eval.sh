#!/bin/bash

# Define an array of commands
commands=(
    "CUDA_VISIBLE_DEVICES=0, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.1 --thr_Amask_to_Abb 0.1 --save_res_path thrIoU0.1_thrAmask_0.1.csv" 
    "CUDA_VISIBLE_DEVICES=0, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.1 --thr_Amask_to_Abb 0.2 --save_res_path thrIoU0.1_thrAmask_0.2.csv" 
    "CUDA_VISIBLE_DEVICES=0, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.1 --thr_Amask_to_Abb 0.3 --save_res_path thrIoU0.1_thrAmask_0.3.csv" 
    "CUDA_VISIBLE_DEVICES=0, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.1 --thr_Amask_to_Abb 0.5 --save_res_path thrIoU0.1_thrAmask_0.5.csv" 
    "CUDA_VISIBLE_DEVICES=0, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.1 --thr_Amask_to_Abb 0.7 --save_res_path thrIoU0.1_thrAmask_0.7.csv" 



    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 9 --crop_gt --save_res_path gt_factor9.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 16 --crop_gt --save_res_path gt_factor16.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 25 --crop_gt --save_res_path gt_factor25.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 36 --crop_gt --save_res_path gt_factor36.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 49 --crop_gt --save_res_path gt_factor49_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64 --crop_gt --save_res_path gt_factor64_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 81 --crop_gt --save_res_path gt_factor81_duda.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --crop_gt --save_res_path gt_factor100_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 121 --crop_gt --save_res_path gt_factor121_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 144 --crop_gt --save_res_path gt_factor144_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 169 --crop_gt --save_res_path gt_factor169_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 196 --crop_gt --save_res_path gt_factor196_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 225 --crop_gt --save_res_path gt_factor225_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 256 --crop_gt --save_res_path gt_factor256_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 289 --crop_gt --save_res_path gt_factor289_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 400 --crop_gt --save_res_path gt_factor400_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 625 --crop_gt --save_res_path gt_factor625_duda.csv"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 4 --min_box_factor 512  --save_res_path with512_factor4_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 9 --min_box_factor 512  --save_res_path with512_factor9_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 16 --min_box_factor 512  --save_res_path with512_factor16_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 25 --min_box_factor 512  --save_res_path with512_factor25_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 36 --min_box_factor 512  --save_res_path with512_factor36_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 49 --min_box_factor 512  --save_res_path with512_factor49_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64 --min_box_factor 512  --save_res_path with512_factor64_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 81 --min_box_factor 512  --save_res_path with512_factor81_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 121 --min_box_factor 512  --save_res_path with512_factor121_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 144 --min_box_factor 512  --save_res_path with512_factor144_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 169 --min_box_factor 512  --save_res_path with512_factor169_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 400 --min_box_factor 512  --save_res_path with512_factor400_duda.csv"





    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 36 --min_box_factor 512  --save_res_path with64_factor36_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 49  --save_res_path with128_factor49_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64  --save_res_path with128_factor64_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 81  --save_res_path with128_factor81_duda.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100  --save_res_path with128_factor100_duda.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 7 --exclude_empty_masks --save_res_path with128_factor100_RR7_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 6 --exclude_empty_masks --save_res_path with128_factor100_RR6_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --memory_stride 5 --exclude_empty_masks --save_res_path with128_factor100_RR5_duda.csv"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 121  --save_res_path with128_factor121_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 144  --save_res_path with128_factor144_duda.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 169  --save_res_path with128_factor169_duda.csv"





    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 100 --save_res_path factor100.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 4 --save_res_path factor4.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 9 --save_res_path factor9.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 16 --save_res_path factor16.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 25 --save_res_path factor25.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 36 --save_res_path factor36.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 49 --save_res_path factor49.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 64 --save_res_path factor64.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 81 --save_res_path factor81.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 121 --save_res_path factor121.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 144 --save_res_path factor144.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 169 --save_res_path factor169.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 196 --save_res_path factor196.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 225 --save_res_path factor225.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 256 --save_res_path factor256.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 289 --save_res_path factor289.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 324 --save_res_path factor324.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 361 --save_res_path factor361.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 400 --save_res_path factor400.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_box --factor 625 --save_res_path factor625.csv"




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
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --interpolation replicate --save_res_path of_repl_af_0.1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --interpolation bilinear --save_res_path of_bilinear_af_0.1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.1 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.1.csv"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --interpolation replicate --save_res_path of_repl_af_0.15.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --interpolation bilinear --save_res_path of_bilinear_af_0.15.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.15.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.15 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.15.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --interpolation replicate --save_res_path of_repl_af_0.25.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --interpolation bilinear --save_res_path of_bilinear_af_0.25.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.25.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.25 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.25.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --interpolation replicate --save_res_path of_repl_af_0.35.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --interpolation bilinear --save_res_path of_bilinear_af_0.35.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.35.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.35 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.35.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --interpolation replicate --save_res_path of_repl_af_0.5.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --interpolation bilinear --save_res_path of_bilinear_af_0.5.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.5.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.5 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.5.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --interpolation replicate --save_res_path of_repl_af_0.6.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --interpolation bilinear --save_res_path of_bilinear_af_0.6.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --interpolation replicate --save_res_path bof_repl_af_0.6.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 0.6 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_0.6.csv"


    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation replicate --save_res_path of_repl_af_1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation bilinear --save_res_path of_bilinear_af_1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation replicate --save_res_path bof_repl_af_1.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation bilinear --save_res_path bof_bilinear_af_1.csv"

    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation replicate --close_trans --open_trans --save_res_path of_repl_af_1_co_trans.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --interpolation bilinear --close_trans --open_trans --save_res_path of_bilinear_af_1_co_trans.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation replicate --close_trans --open_trans  --save_res_path bof_repl_af_1_co_trans.csv"
    # "CUDA_VISIBLE_DEVICES=1, python3 cropped_tracking.py --use_prev_mask --alfa_flow 1 --backward_of --interpolation bilinear --close_trans --open_trans  --save_res_path bof_bilinear_af_1_co_trans.csv"


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