import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .deepseek_vl2 import DeepSeekVL2
from .molmo import molmo