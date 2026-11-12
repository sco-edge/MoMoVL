# Copyright 2023 VLMEvalKit Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0-VLMEvalKit.txt).
# Modified from VLMEvalKit (https://github.com/open-compass/VLMEvalKit)
# by Dongkeun Lee for MoMoVL, 2026.

import copy as cp
import os
from functools import partial

import vlmeval.vlm as vlm

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = "Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. "

# Use MoE-based VLMs (i.e., deepseekvl2 and MolmoE) for evaluation
deepseekvl2_series = {
    "deepseek_vl2_tiny": partial(
        vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-tiny"
    ),
    "deepseek_vl2_small": partial(
        vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-small"
    ),
    "deepseek_vl2": partial(
        vlm.DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2"),
}

molmo_series = {
    "molmoE-1B-0924": partial(vlm.molmo, model_path="allenai/MolmoE-1B-0924"),
    # "molmo-7B-D-0924": partial(vlm.molmo, model_path="allenai/Molmo-7B-D-0924"),
    # "molmo-7B-O-0924": partial(vlm.molmo, model_path="allenai/Molmo-7B-O-0924"),
    # "molmo-72B-0924": partial(vlm.molmo, model_path="allenai/Molmo-72B-0924"),
}

supported_VLM = {}

model_groups = [deepseekvl2_series, molmo_series]

for grp in model_groups:
    supported_VLM.update(grp)
