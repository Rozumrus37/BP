# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
 
from visualizer import Visualizer
# from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--video_path",
    #     default="./assets/apple.mp4",
    #     help="path to a video",
    # )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()


    # if not os.path.isfile(args.video_path):
    #     raise ValueError("Video file does not exist")

    # if args.checkpoint is not None:
    #     model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    # else:
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    # print(model)
    # print("asd", model.macro_block_size)

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        

        queries = torch.tensor([
            [0., 314., 290.],
            [0., 314., 297.],
            [0., 316., 282.]

            # [0., 433., 277.],
            # [0., 433., 285.],
            # [0., 439.,290.],
            # [0., 442., 282.],
            # [0., 434., 271.]


            # [0., 433., 277.] # hand2

            # [0., 356., 122.], zerbafish1
            # # [0., 377., 120.],
            # # [0., 400., 120.]



            # [0., 136., 518.], # ants1
            # [0., 139., 506.],
            # [0., 143., 476.]

            # [0., 723., 241.], conductinon1 
            # [0., 724., 248.],
            # [0., 717., 244.]

            # [0., 539., 201.], #drone1
            # [0., 545., 207.],
            # [0., 555., 214.]


            # [0., 628., 232.]
            # [0., 625.,218.],
            # [0., 611.,216.],
            # [0., 595.,212.]

        ], device='cuda')



        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],
            # grid_size=grid_size,
            # grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True

    video_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/marathon/color"

    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    print("Len frames: ", len(frame_names))


    cnt = 0
    print("Model step: ", model.step)

    model.step = 8

    for i, frame in enumerate(frame_names):
        # print(model.step)
        if i % model.step == 0 and i != 0:
            # print(window_frames, is_first_step)
            # import pdb; pdb.set_trace();

            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=10,#args.grid_size,
                grid_query_frame=0,#args.grid_query_frame,
            )



            is_first_step = False
        window_frames.append(iio.imread(os.path.join(video_path, frame)))
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )

    print("Tracks are computed")

    # save a video with predicted tracks
    
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]

    # import pdb; pdb.set_trace();


    vis = Visualizer(save_dir="saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )