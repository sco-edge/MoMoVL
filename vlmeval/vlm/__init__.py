# Copyright 2023 VLMEvalKit Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0-VLMEvalKit.txt).
# Modified from VLMEvalKit (https://github.com/open-compass/VLMEvalKit)
# by Dongkeun Lee for MoMoVL, 2026.

import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .deepseek_vl2 import DeepSeekVL2
from .molmo import molmo