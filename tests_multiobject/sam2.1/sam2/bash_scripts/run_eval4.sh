#!/bin/bash

# Define an array of commands
commands=(


    "CUDA_VISIBLE_DEVICES=3, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.5 --thr_Amask_to_Abb 0.1 --save_res_path thrIoU0.5_thrAmask_0.1.csv" 
    "CUDA_VISIBLE_DEVICES=3, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.5 --thr_Amask_to_Abb 0.2 --save_res_path thrIoU0.5_thrAmask_0.2.csv"" 
    "CUDA_VISIBLE_DEVICES=3, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.5 --thr_Amask_to_Abb 0.3 --save_res_path thrIoU0.5_thrAmask_0.3.csv"" 
    "CUDA_VISIBLE_DEVICES=3, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.5 --thr_Amask_to_Abb 0.5 --save_res_path thrIoU0.5_thrAmask_0.5.csv"" 
    "CUDA_VISIBLE_DEVICES=3, python3 parallel_tracking.py  --use_prev_box --thr_IoU_BB1_BBsm 0.5 --thr_Amask_to_Abb 0.7 --save_res_path thrIoU0.5_thrAmask_0.7.csv"" 


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