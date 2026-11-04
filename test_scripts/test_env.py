import os, sys
PROJECT_PATH = os.getenv("PROJECT_PATH")
sys.path.insert(0, PROJECT_PATH)

import torch, gc
from vlmeval.config import supported_VLM

target_models = ['deepseek_vl2_tiny', 'molmoE-1B-0924']

for model_name in target_models:
	print(f"[+] Testing {model_name}")
	model = supported_VLM[model_name]()

	# Forward Single Image
	ret = model.generate([f'{PROJECT_PATH}/test_scripts/test_images/giraffe.jpg', 'Can you tell me what is in this image?'])
	print(ret) 

	# Clear GPU memory
	del model
	gc.collect()
	torch.cuda.empty_cache()
