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
            [0., 539., 209.],
            [0., 553.,208.],
            [0., 536.,218.],
            [0., 547.,214.]

            # [0., 721., 245.],
            # [0., 728., 246.],
            # [0., 717., 251.]

            # [0., 437., 287.],
            # [0., 436., 275.],
            # [0., 429., 279.],

            # [0.,566.,282.],
            # [0.,555.,266.],
            # [0.,568.,291.],
            # [0.,557.,291.],
            # [0.,560.,274.],

            # [0,245,58],
            # [0,246,85],
            # [0,269,90],
            # [0,260,102],
            # [0,241,101],
            # [0,265,75],
            # [0,224,74],
            # [0,226,93],
            # [0,278,103],
            # [0., 233.0, 69.0],
            # [0., 250.0, 87.0],  # point tracked from the first frame
            # [10., 600., 500.], # frame number 10
            # [20., 750., 600.], # ...
            # [30., 900., 200.]
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

    video_path = "/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/rabbit/color"

    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = [i for i in frame_names if i[0] != '.']
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    print("Len frames: ", len(frame_names))


    cnt = 0
    print("Model step: ", model.step)

    for i, frame in enumerate(frame_names):
        # print(model.step)
        if i % model.step == 0 and i != 0:
            # print(window_frames, is_first_step)
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
    seq_name = "rabbit"
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]

    # import pdb; pdb.set_trace();


    vis = Visualizer(save_dir="saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )