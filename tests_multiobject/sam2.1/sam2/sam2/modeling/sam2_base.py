# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from compute_iou import * 
from utilities_eval import *
import numpy as np 

from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import torch.nn as nn
from optical_flow import get_mask
import random
# from ptl_flow import get_mask_ptl_flow

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0
# seq="rabbit"

class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        self.min_obj_score_logits = 0# -1

        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def compute_center_of_gravity(self, mask):
        """
        Compute the center of gravity of a binary segmentation mask.
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return 0
            # raise ValueError("Empty segmentation mask")
        center_of_gravity = coords.mean(axis=0)
        return center_of_gravity

    def translate_mask(self, mask, translation):
        """
        Translate a binary segmentation mask by a given vector.
        """
        translated_mask = np.zeros_like(mask)
        coords = np.argwhere(mask > 0)
        for coord in coords:
            new_coord = (coord + translation).astype(int)
            if all(0 <= new_coord[i] < mask.shape[i] for i in range(len(mask.shape))):
                translated_mask[tuple(new_coord)] = 1
        return translated_mask


    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        frame_idx=None,
        video_H=None,
        video_W=None,
        seq=None,
        oracle_threshold=None,
        bbox=None,
        prev_mask=None,
        H_original=None,
        W_original=None,
        alfa_flow=None,
        direct_comp_to_prev_pred=False,
        backward_of=False,
        interpolation='bilinear', 
        kernel_size=3, 
        close_trans=False, 
        open_trans=False,
        oracle=False,
        forward_of=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:

            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
    
        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None


        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.min_obj_score_logits #0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]

        miss = 0
        best_low_res_multimask_according_to_sam2 = None

        if multimask_output:
            if oracle: #seq != None and False:
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, 0].unsqueeze(1)

                _, first_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                low_res_masks = low_res_multimasks[batch_inds, 1].unsqueeze(1)

                _, second_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                low_res_masks = low_res_multimasks[batch_inds, 2].unsqueeze(1)

                _, third_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                # import pdb; pdb.set_trace();

                H, W = get_nth_mask(seq, frame_idx).shape

                first_mask = get_full_size_mask(first_mask, bbox, H, W)
                second_mask = get_full_size_mask(second_mask, bbox, H, W)
                third_mask = get_full_size_mask(third_mask, bbox, H, W)

                IoU_with_prev_for_mask1 = obatin_iou(first_mask, get_nth_mask(seq, frame_idx))
                IoU_with_prev_for_mask2 = obatin_iou(second_mask, get_nth_mask(seq, frame_idx))
                IoU_with_prev_for_mask3 = obatin_iou(third_mask, get_nth_mask(seq, frame_idx))

                max_oracle_iou = max(IoU_with_prev_for_mask1, max(IoU_with_prev_for_mask2, IoU_with_prev_for_mask3))

                if IoU_with_prev_for_mask1 == max_oracle_iou:
                    best_iou_inds = 0
                elif IoU_with_prev_for_mask2 == max_oracle_iou:
                    best_iou_inds = 1
                else:
                    best_iou_inds = 2

                best_idx_sam2 = torch.argmax(ious, dim=-1)
                best_low_res_multimask_according_to_sam2 = low_res_multimasks[batch_inds, best_idx_sam2].unsqueeze(1)

                if best_idx_sam2 == 0:
                    oracle_score_best_sam2_mask = IoU_with_prev_for_mask1
                elif best_idx_sam2 == 1:
                    oracle_score_best_sam2_mask = IoU_with_prev_for_mask2
                else:
                    oracle_score_best_sam2_mask = IoU_with_prev_for_mask3

                # print(best_iou_inds,  torch.argmax(ious, dim=-1) )
                if (max_oracle_iou - oracle_score_best_sam2_mask)*100 > oracle_threshold and best_iou_inds != best_idx_sam2:
                    miss = 1
                
                # print(f"For frame idx {frame_idx} the error is: {(max_oracle_iou - oracle_score_best_sam2_mask)*100: .2f}% and oracle_IoU {oracle_score_best_sam2_mask} and max {max_oracle_iou}")                
            elif forward_of or backward_of or direct_comp_to_prev_pred: #not (prev_mask is None):

                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, 0].unsqueeze(1)

                _, first_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                low_res_masks = low_res_multimasks[batch_inds, 1].unsqueeze(1)

                _, second_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                low_res_masks = low_res_multimasks[batch_inds, 2].unsqueeze(1)

                _, third_mask = self._get_orig_video_res_output_RR(
                    device, video_H, video_W, low_res_masks
                )

                H, W = H_original, W_original

                first_mask = get_full_size_mask(first_mask, bbox, H, W)
                second_mask = get_full_size_mask(second_mask, bbox, H, W)
                third_mask = get_full_size_mask(third_mask, bbox, H, W)

                if np.any(prev_mask[0] > 0) and np.any(first_mask > 0) and np.any(second_mask > 0) and np.any(third_mask > 0):

                    if backward_of:
                        of_mask1 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", 
                            f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", first_mask, frame_idx+1, seq=seq, 
                            interpolation=interpolation, kernel_size=kernel_size, close_trans=close_trans, open_trans=open_trans)
                        of_mask2 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", 
                            f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", second_mask, frame_idx+1, seq=seq,
                            interpolation=interpolation, kernel_size=kernel_size, close_trans=close_trans, open_trans=open_trans)
                        of_mask3 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", 
                            f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", third_mask, frame_idx+1, seq=seq, 
                            interpolation=interpolation, kernel_size=kernel_size, close_trans=close_trans, open_trans=open_trans)

                        iou1 = alfa_flow * obatin_iou(prev_mask[0], of_mask1) + (1 - alfa_flow) * ious[0][0].item()
                        iou2 = alfa_flow * obatin_iou(prev_mask[0], of_mask2) + (1 - alfa_flow) * ious[0][1].item()
                        iou3 = alfa_flow * obatin_iou(prev_mask[0], of_mask3) + (1 - alfa_flow) * ious[0][2].item()

                        if iou1 > iou2 and iou1 > iou3:
                            best_iou_inds = 0
                        elif iou2 > iou1 and iou2 > iou3:
                            best_iou_inds = 1
                        elif iou3 > iou1 and iou3 > iou2:
                            best_iou_inds = 2

                        # random_number = random.choice([0, 1, 2])
                        # best_iou_inds = random_number

                    elif forward_of:
                        of_mask = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", 
                            f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", 
                            prev_mask[0], frame_idx+1, seq=seq, 
                            interpolation=interpolation, kernel_size=kernel_size, close_trans=close_trans, open_trans=open_trans)
                    
                        iou1 = alfa_flow * obatin_iou(first_mask, of_mask) + (1 - alfa_flow) * ious[0][0].item()
                        iou2 = alfa_flow * obatin_iou(second_mask, of_mask) + (1 - alfa_flow) * ious[0][1].item()
                        iou3 = alfa_flow * obatin_iou(third_mask, of_mask) + (1 - alfa_flow) * ious[0][2].item()

                        if iou1 > iou2 and iou1 > iou3:
                            best_iou_inds = 0
                        elif iou2 > iou1 and iou2 > iou3:
                            best_iou_inds = 1
                        elif iou3 > iou1 and iou3 > iou2:
                            best_iou_inds = 2
                    elif direct_comp_to_prev_pred:
                        iou1 = alfa_flow * obatin_iou(first_mask, prev_mask[0]) + (1 - alfa_flow) * ious[0][0].item()
                        iou2 = alfa_flow * obatin_iou(second_mask, prev_mask[0]) + (1 - alfa_flow) * ious[0][1].item()
                        iou3 = alfa_flow * obatin_iou(third_mask, prev_mask[0]) + (1 - alfa_flow) * ious[0][2].item()

                        if iou1 > iou2 and iou1 > iou3:
                            best_iou_inds = 0
                        elif iou2 > iou1 and iou2 > iou3:
                            best_iou_inds = 1
                        elif iou3 > iou1 and iou3 > iou2:
                            best_iou_inds = 2


            else:
                best_iou_inds = torch.argmax(ious, dim=-1) 

            # take the best mask prediction (with the highest IoU estimation)
            #best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)


            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks



        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            miss,
            best_low_res_multimask_according_to_sam2,
        )


    def _get_orig_video_res_output_RR(self, device, video_H, video_W, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
       
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks


    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _, _, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            0,
            None,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False, # tracking in reverse time order (for demo usage)
        use_log_memory_stride=False,
    ):

        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )

            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

     
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.

            if not use_log_memory_stride:
                stride = 1 if self.training else self.memory_temporal_stride_for_eval
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)
                            prev_frame_idx = ((frame_idx - 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                        # frames, we still attend to it as if it's a non-conditioning frame.
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out))
            else:
                to_subtract = [0, 32, 16, 8, 4, 2, 1]

                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame

                    prev_frame_idx = frame_idx - to_subtract[t_pos]

                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                        # frames, we still attend to it as if it's a non-conditioning frame.
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out))




            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # print(len(to_cat_memory_pos_embed))


            # print(len(to_cat_memory_pos_embed), frame_idx)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = self.max_obj_ptrs_in_encoder #min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0: #or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]

                    # print("TO CAT MEMOYR:", len(to_cat_memory), to_cat_memory[0].shape, obj_ptrs.shape, self.max_obj_ptrs_in_encoder)
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem


            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        # print("MEMMEM FINAL: ", memory.shape)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        # import pdb; pdb.set_trace()

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        video_H=None, 
        video_W=None,
        seq=None,
        oracle_threshold=None,
        bbox=None,
        prev_mask=None,
        H_original=None,
        W_original=None,
        alfa_flow=None,
        direct_comp_to_prev_pred=False,
        backward_of=False,
        interpolation='bilinear', 
        kernel_size=3, 
        close_trans=False, 
        open_trans=False,
        use_log_memory_stride=False,
        forward_of=False,
        oracle=False,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_log_memory_stride=use_log_memory_stride,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                frame_idx=frame_idx,
                video_H=video_H,
                video_W=video_W,
                seq=seq,
                oracle_threshold=oracle_threshold,
                bbox=bbox,
                prev_mask=prev_mask,
                H_original=H_original,
                W_original=W_original,
                alfa_flow=alfa_flow,
                direct_comp_to_prev_pred=direct_comp_to_prev_pred,
                backward_of=backward_of,
                forward_of=forward_of,
                oracle=oracle,
                interpolation=interpolation, 
                kernel_size=kernel_size, 
                close_trans=close_trans, 
                open_trans=open_trans,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        memory_stride=1,
        video_H=None,
        video_W=None,
        seq=None,
        oracle_threshold=None,
        bbox=None,
        prev_mask=None,
        original_H=None,
        original_W=None,
        alfa_flow=None,
        direct_comp_to_prev_pred=False,
        backward_of=False,
        interpolation='bilinear', 
        kernel_size=3, 
        close_trans=False, 
        open_trans=False,
        use_log_memory_stride=False,
        oracle=False,
        forward_of=False,
    ):  
        self.memory_temporal_stride_for_eval = memory_stride

        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            prev_sam_mask_logits=prev_sam_mask_logits,
            video_H=video_H,
            video_W=video_W,
            seq=seq,
            oracle_threshold=oracle_threshold,
            bbox=bbox,
            prev_mask=prev_mask,
            H_original=original_H,
            W_original=original_W,
            alfa_flow=alfa_flow,
            direct_comp_to_prev_pred=direct_comp_to_prev_pred,
            backward_of=backward_of,
            forward_of=forward_of,
            oracle=oracle,
            interpolation=interpolation, 
            kernel_size=kernel_size, 
            close_trans=close_trans, 
            open_trans=open_trans,
            use_log_memory_stride=use_log_memory_stride,

        )


        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            miss,
            second_best_low_res_masks,
        ) = sam_outputs

        # print(frame_idx, object_score_logits)

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out, miss, second_best_low_res_masks

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks





# IoU_with_prev_for_mask1 = obatin_iou(first_mask, prev_mask)
# IoU_with_prev_for_mask2 = obatin_iou(second_mask, prev_mask)
# IoU_with_prev_for_mask3 = obatin_iou(third_mask, prev_mask)



# cog1 = self.compute_center_of_gravity(prev_mask)
# cog2 = self.compute_center_of_gravity(first_mask)

# # Align mask2 to mask1
# translation = cog2 - cog1
# aligned_mask2 = self.translate_mask(prev_mask, translation)

# IoU_with_prev_for_mask1 = obatin_iou(aligned_mask2, first_mask)

# cog1 = self.compute_center_of_gravity(prev_mask)
# cog2 = self.compute_center_of_gravity(second_mask)

# # Align mask2 to mask1
# translation = cog2 - cog1
# aligned_mask2 = self.translate_mask(prev_mask, translation)

# IoU_with_prev_for_mask2 = obatin_iou(aligned_mask2, second_mask)

# cog1 = self.compute_center_of_gravity(prev_mask)
# cog2 = self.compute_center_of_gravity(third_mask)

# # Align mask2 to mask1
# translation = cog2 - cog1
# aligned_mask2 = self.translate_mask(prev_mask, translation)

# IoU_with_prev_for_mask3 = obatin_iou(aligned_mask2, third_mask)



# print(IoU_with_prev_for_mask1, IoU_with_prev_for_mask2, IoU_with_prev_for_mask3)
# print(ious)
# print(obatin_iou(first_mask, prev_mask), obatin_iou(second_mask, prev_mask), obatin_iou(third_mask, prev_mask))



# best_idx_sam2 = torch.argmax(ious, dim=-1)

# if best_idx_sam2 == 0:
#     best_sam2_mask = IoU_with_prev_for_mask1
# elif best_idx_sam2 == 1:
#     best_sam2_mask = IoU_with_prev_for_mask2
# else:
#     best_sam2_mask = IoU_with_prev_for_mask3

# #print(obatin_iou(first_mask, prev_mask), obatin_iou(second_mask, prev_mask), obatin_iou(third_mask, prev_mask))

# # if best_sam2_mask > 0.5:
# #     best_iou_inds= best_iou_inds
# # else:
# if IoU_with_prev_for_mask1 > IoU_with_prev_for_mask2+0.05 and IoU_with_prev_for_mask1 > IoU_with_prev_for_mask3+0.05 and obatin_iou(first_mask, prev_mask) > 0.02 and ious[0][0] > 0.1:
#     best_iou_inds = 0
# elif IoU_with_prev_for_mask2 > IoU_with_prev_for_mask1+0.05 and IoU_with_prev_for_mask2 > IoU_with_prev_for_mask3+0.05 and obatin_iou(second_mask, prev_mask) > 0.02 and ious[0][1] > 0.1:
#     best_iou_inds = 1
# elif IoU_with_prev_for_mask3 > IoU_with_prev_for_mask1+0.05 and IoU_with_prev_for_mask3 > IoU_with_prev_for_mask2+0.05 and obatin_iou(third_mask, prev_mask) > 0.02 and ious[0][2] > 0.1:
#     best_iou_inds = 2

# print("picked ", best_iou_inds)


# import pdb; pdb.set_trace()

# best_idx_sam2 = torch.argmax(ious, dim=-1)

# if best_idx_sam2 == 0:
#     best_sam2_mask = IoU_with_prev_for_mask1
# elif best_idx_sam2 == 1:
#     best_sam2_mask = IoU_with_prev_for_mask2
# else:
#     best_sam2_mask = IoU_with_prev_for_mask3

# if best_sam2_mask > 0.1:
#     best_iou_inds = best_idx_sam2
# elif IoU_with_prev_for_mask1> 0.1:
#     best_iou_inds = 0
# elif IoU_with_prev_for_mask2 > 0.1:
#     best_iou_inds = 1
# elif IoU_with_prev_for_mask3 > 0.1: 
#     best_iou_inds = 2
# else:
#     best_iou_inds = best_idx_sam2





# seq="drone1"
# of_mask1 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", first_mask, frame_idx+1)
# of_mask2 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", second_mask, frame_idx+1)
# of_mask3 = get_mask(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg", f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx:0{8}d}.jpg", third_mask, frame_idx+1)


# # print(of_mask.shape, first_mask.shape)
# iou1 = obatin_iou(prev_mask[0], of_mask1)
# iou2 =obatin_iou(prev_mask[0], of_mask2)
# iou3 = obatin_iou(prev_mask[0], of_mask3)

# if iou1 > iou2 and iou1 > iou3:
#     index_maxx = 0
# elif iou2 > iou1 and iou2 > iou3:
#     index_maxx = 1
# elif iou3 > iou1 and iou3 > iou2:
#     index_maxx = 2




# min_row, min_col, max_row, max_col = get_bounding_box(first_mask)

# img_pil = Image.open(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg")

# first_mask = np.repeat(first_mask[:, :, np.newaxis], 3, axis=2)

# image_array = np.array(img_pil)
# masked_image_array = image_array * first_mask

# # Convert back to a PIL image
# masked_image = Image.fromarray(masked_image_array)

# first_img = masked_image.crop((min_row, min_col, max_row, max_col))

 
# min_row, min_col, max_row, max_col = get_bounding_box(second_mask)

# img_pil = Image.open(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg")

# second_mask = np.repeat(second_mask[:, :, np.newaxis], 3, axis=2)

# image_array = np.array(img_pil)
# masked_image_array = image_array * second_mask

# # Convert back to a PIL image
# masked_image = Image.fromarray(masked_image_array)

# second_img = masked_image.crop((min_row, min_col, max_row, max_col))



# min_row, min_col, max_row, max_col = get_bounding_box(third_mask)

# img_pil = Image.open(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx+1:0{8}d}.jpg")

# third_mask = np.repeat(third_mask[:, :, np.newaxis], 3, axis=2)

# image_array = np.array(img_pil)
# masked_image_array = image_array * third_mask

# # Convert back to a PIL image
# masked_image = Image.fromarray(masked_image_array)

# third_img = masked_image.crop((min_row, min_col, max_row, max_col))


# prev_imgs = []
# cnt=0

# for mask_i in prev_mask:
#     # print(cnt, len(prev_mask), mask_i.shape)
#     min_row, min_col, max_row, max_col = get_bounding_box(mask_i)

#     img_pil = Image.open(f"/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/{seq}/color/" + f"{frame_idx-cnt:0{8}d}.jpg")

#     image_array = np.array(img_pil)

#     mask_i = np.repeat(mask_i[:, :, np.newaxis], 3, axis=2)

#     masked_image_array = image_array * mask_i

#     # print("masked image array: ", masked_image_array.shape)

#     # Convert back to a PIL image
#     masked_image = Image.fromarray(masked_image_array.astype(np.uint8))

#     source_img = masked_image.crop((min_row, min_col, max_row, max_col))

#     prev_imgs.append(source_img)

#     cnt+=1


# first_img.save(f"saved_dino/{frame_idx+1:0{8}d}_first.jpg")
# second_img.save(f"saved_dino/{frame_idx+1:0{8}d}_second.jpg")
# third_img.save(f"saved_dino/{frame_idx+1:0{8}d}_third.jpg")



# if frame_idx+1 == 32:
#     first_img.save('32_first_img.png')
#     second_img.save('32_second_img.png')
#     third_img.save('32_third_img.png')

# if frame_idx+1 == 33:
#     first_img.save('33_first_img.png')
#     second_img.save('33_second_img.png')
#     third_img.save('33_third_img.png')

# if frame_idx+1 == 34:
#     first_img.save('34_first_img.png')
#     second_img.save('34_second_img.png')
#     third_img.save('34_third_img.png')



# first_img = (first_mask * 255).astype(np.uint8)
# first_img = Image.fromarray(first_img)

# second_img = (second_mask * 255).astype(np.uint8)
# second_img = Image.fromarray(second_img)

# third_img = (third_mask * 255).astype(np.uint8)
# third_img = Image.fromarray(third_img)

# source_img = (prev_mask * 255).astype(np.uint8)
# source_img = Image.fromarray(source_img)


# images = [first_img, second_img, third_img]


# maxx, cnt, index_maxx = -1, 0, -1

# for img in images:

#     summ = 0
#     cnt_inner = 0
#     for source_img in prev_imgs:  
#         with torch.no_grad():
#             inputs1 = processor_dino(images=img, return_tensors="pt").to(device)
#             outputs1 = model_dino(**inputs1)
#             image_features1 = outputs1.last_hidden_state
#             image_features1 = image_features1.mean(dim=1)

#         # import pdb; pdb.set_trace()


#         with torch.no_grad():
#             inputs2 = processor_dino(images=source_img, return_tensors="pt").to(device)
#             outputs2 = model_dino(**inputs2)
#             image_features2 = outputs2.last_hidden_state
#             image_features2 = image_features2.mean(dim=1)

#         cos = nn.CosineSimilarity(dim=0)
#         sim = cos(image_features1[0],image_features2[0]).item()
#         sim = (sim+1)/2

#         if cnt_inner == 0 or cnt_inner == len(prev_imgs)-1:
#             sim *= 2 

#         summ += sim
#         cnt_inner += 1

#     summ /= len(prev_imgs)

#     if summ > maxx:
#         maxx = summ
#         index_maxx = cnt 

#     print('Similarity:', summ)


#     cnt+=1

# print(f"chosen for {frame_idx+1} is {index_maxx}")




# import pdb; pdb.set_trace()

# import pdb; pdb.
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# index_dino = faiss.IndexFlatL2(768)

# for img in images:
#     # print(image_path)
#     img = img.convert('RGB')
#     # clip_features = extract_features_clip(img)
#     # add_vector_to_index(clip_features,index_clip)
#     dino_features = extract_features_dino(img)
#     add_vector_to_index(dino_features,index_dino)

# faiss.write_index(index_dino,"dino.index")


# image = source_img

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')

# with torch.no_grad():
# # print(processor_dino)
#     inputs_dino = processor_dino(images=image, return_tensors="pt").to('cuda')
#     outputs_dino = model_dino(**inputs_dino)
#     image_features_dino = outputs_dino.last_hidden_state

#     image_features_dino = image_features_dino.mean(dim=1)

# image_features_dino = normalizeL2(image_features_dino)
# # image_features_clip = normalizeL2(image_features_clip)

# #Search the top 5 images
# # index_clip = faiss.read_index("clip.index")
# index_dino = faiss.read_index("dino.index")

# #Get distance and indexes of images associated
# d_dino,i_dino = index_dino.search(image_features_dino,3)

# print(i_dino, d_dino)

# import pdb; pdb.set_trace()

# seq="hand2"

# IoU_with_prev_for_mask1 = obatin_iou(first_mask, get_nth_mask(seq, frame_idx))
# IoU_with_prev_for_mask2 = obatin_iou(second_mask, get_nth_mask(seq, frame_idx))
# IoU_with_prev_for_mask3 = obatin_iou(third_mask, get_nth_mask(seq, frame_idx))

# if IoU_with_prev_for_mask1 > IoU_with_prev_for_mask2 and IoU_with_prev_for_mask1 > IoU_with_prev_for_mask3:
#     print(0)
# elif IoU_with_prev_for_mask2 > IoU_with_prev_for_mask1 and IoU_with_prev_for_mask2 > IoU_with_prev_for_mask3:
#     print(1)
# else:
#     print(2)

# print(IoU_with_prev_for_mask1, IoU_with_prev_for_mask2, IoU_with_prev_for_mask3, frame_idx+1)