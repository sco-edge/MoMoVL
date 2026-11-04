import os, sys
PROJECT_PATH = os.getenv("PROJECT_PATH")
sys.path.insert(0, PROJECT_PATH)

import logging
import string

import pandas as pd
import gc, torch
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import listinstr
from .base import BaseModel

from models.molmoe import MolmoImageProcessor, MolmoProcessor, MolmoForCausalLM
from transformers import GPT2TokenizerFast

TYPE_PROMPTS = {
    'Y/N': 'vqa2:',
    'VQA': 'vqa2:',
    'MCQ': 'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST': 'ai2_diagram:',
    'AI2D_TEST_NO_MASK': 'ai2_diagram:',
    'COCO_VAL': 'coco_captioning:',
    'ChartQA_TEST': 'chart_qa:',
    'ChartQA_VAL': 'chart_qa:',
    'DocVQA_VAL': 'doc_qa:',
    'DocVQA_TEST': 'doc_qa:',
    'InfoVQA_TEST': 'info_qa:',
    'InfoVQA_VAL': 'info_qa:',
    'OCRVQA_TEST': 'ocr_vqa:',
    'OCRVQA_TESTCORE': 'ocr_vqa:',
    'ScienceQA_VAL': 'science_qa:',
    'ScienceQA_TEST': 'science_qa:',
    'TableVQABench': 'tabwmp_da:',
    'TextVQA_VAL': 'text_vqa:'
}

class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False
    NUM_EXPERTS = 64
    
    def __init__(self, model_path="allenai/MolmoE-1B-0924", **kwargs):
        # try:
        #     import einops  # noqa: F401
        #     from transformers import (AutoModelForCausalLM, AutoProcessor,  # noqa: F401
        #                               GenerationConfig)
        # except Exception as e:
        #     logging.critical('Please install transformer and einops before using molmo.')
        #     raise e

        self.model = MolmoForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        ).eval()
        
        image_processor = MolmoImageProcessor.from_pretrained(f'{PROJECT_PATH}/models/molmoe')
        tokenizer = GPT2TokenizerFast.from_pretrained(f'{PROJECT_PATH}/models/molmoe')
        self.processor = MolmoProcessor(image_processor=image_processor, tokenizer=tokenizer, torch_dtype=torch.bfloat16)
        self.kwargs = kwargs
        self.model_name = model_path
        # set default maximum number of crops to 36
        self.max_crops = kwargs.get('max_crops', 36)

        # [DEV] Quantize model if specific quantization method is given
        self.quant_method = kwargs.get('quant_method')
        if self.quant_method is not None:
            # print(f'[+] self.quant_method = {self.quant_method}')
            self.quantize_model(self.quant_method)

        # [DEV] Quantize experts if quantization config is provided
        self.quant_config = kwargs.get('quant_config')
        if self.quant_config is not None:
            # print(f'[+] self.quant_config = {self.quant_config}')
            self.quantize_experts(self.quant_config)

        # Clean up memory space occupied by original weights
        if self.quant_method is not None or self.quant_config is not None:
            gc.collect()
            torch.cuda.empty_cache()

    # [DEV] Implemented methods: HQQ_4/8, MoE-APEX, Fate
    def quantize_model(self, quant_method):
        if 'HQQ' in quant_method:
            from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
            try:
                bitwidth = int(quant_method.split("_")[1])
            except Exception as e:
                print(f"[!] Error parsing HQQ bitwidth: {e}")
                sys.exit(1)
            
            if bitwidth == 4:
                quant_config = BaseQuantizeConfig(nbits=4, group_size=64)
            elif bitwidth == 8:
                quant_config = BaseQuantizeConfig(nbits=8, group_size=64)
            else:
                raise NotImplementedError(f"[!] HQQ bitwidth should be either 4 or 8")
            
            # Iterate through linear layers in backbone LLM and quantize with HQQ
            for block in self.model.model.transformer.blocks:
                block.attn.att_proj = HQQLinear(
                    linear_layer=block.attn.att_proj, 
                    quant_config=quant_config,
                    compute_dtype=torch.bfloat16,
                    device='cuda',
                    initialize=True,
                    del_orig=True)

                block.attn.attn_out = HQQLinear(
                    linear_layer=block.attn.attn_out, 
                    quant_config=quant_config,
                    compute_dtype=torch.bfloat16,
                    device='cuda',
                    initialize=True,
                    del_orig=True)
                
                for expert in block.mlp.experts:
                    for proj in ['gate_proj', 'up_proj', 'down_proj']:
                        quant_proj = HQQLinear(
                            linear_layer=getattr(expert, proj), 
                            quant_config=quant_config,
                            compute_dtype=torch.bfloat16,
                            device='cuda',
                            initialize=True,
                            del_orig=True)
                        setattr(expert, proj, quant_proj)

        elif quant_method == 'APEX' or quant_method == 'Fate':
            from models.molmoe import MolmoForCausalLM_RT
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.model = MolmoForCausalLM_RT.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            ).eval()
            
            # Expert quantization after weight load
            print(f"[+] [{quant_method}] Quantizing experts after model initialization.")
            for MoE_block in self.model.model.transformer.blocks:
                MoE_block.mlp.quant_method = quant_method
                MoE_block.mlp.quantize_experts()
    
        else:
            raise NotImplementedError(f"[!] {quant_method} is not supported.")

    # [DEV] Replace expert's linear layer with HQQ
    def quantize_experts(self, quant_config):
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
        q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
        q2_config = BaseQuantizeConfig(nbits=2, group_size=32)

        for layer_idx, layer in enumerate(self.model.model.transformer.blocks):
            int4_experts = quant_config[str(layer_idx)]
            expert_config = [q4_config if i in int4_experts else q2_config for i in range(self.NUM_EXPERTS)]

            for expert_idx, expert in enumerate(layer.mlp.experts):
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    quant_proj = HQQLinear(
                        linear_layer=getattr(expert, proj), 
                        quant_config=expert_config[expert_idx],
                        compute_dtype=torch.bfloat16,
                        device='cuda',
                        initialize=True,
                        del_orig=True)
                    setattr(expert, proj, quant_proj)


    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        prefix = None
        if dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt = self.build_prompt_mcq_vqa(line)
        elif dataset in ['MathVista_MINI']:
            prompt = self.build_prompt_mathvista(line)
        elif dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            prompt = self.build_prompt_ai2d(line)
        elif dataset is not None and listinstr(list(DATASET_PROMPTS.keys()), dataset):
            prefix = DATASET_PROMPTS[dataset]  # rest of supervised datasets are in VQA format
            prompt = self.build_prompt_vqa(line, prefix)
        elif dataset is not None and listinstr(['MCQ'], DATASET_TYPE(dataset)):
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        # interleave dataset
        if dataset.startswith('MMMU_'):
            # from .. import MMMUDataset
            from vlmeval.dataset import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
        return message

    def build_prompt_mathvista(self, line):
        if line['question_type'] == 'multi_choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_ai2d(self, line):
        def option_is_abc(line):
            for cand in string.ascii_uppercase:
                if cand in line and not pd.isna(line[cand]):
                    # check if option is single letter
                    if not line[cand].strip().isalpha() or len(line[cand].strip()) > 1:
                        return False
            return True

        if line['abcLabel'] and option_is_abc(line):
            prompt = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                prompt += f'\n{item}'
            prompt = f"ai2_diagram_no_letter: {prompt}"
            # prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram_no_letter:')
        else:
            prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram:')
        return prompt

    def build_prompt_mcq_vqa(self, line):
        if line['question_type'] == 'multiple-choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_multiple_choice(self, line, prefix=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}: {item}'
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['MCQ']} {question}"
        else:
            prompt = f"{prefix} {question}"

        return prompt

    def build_prompt_vqa(self, line, prefix=None):
        question = line['question']
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['VQA']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def generate_inner(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # process the image and text
        max_crops = self.max_crops
        inputs = self.processor.process(
            images=[image],
            text=prompt,
            images_kwargs={
                "max_crops": max_crops
            }
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                # tokenizer=self.processor.tokenizer,
                use_cache=False
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D: map direct answer to letter option
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            # 'ai2_diagram_no_letter: Which of the following is the magma chamber?\nK\nB\nC\nH'
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                # [DEV] Added fallback
                try:
                    answer = options.index(generated_text)
                    generated_text = chr(answer + ord('A'))
                except Exception as e:
                    generated_text = f'Error: {str(e)}'

        # print(dataset, prompt, generated_text, inputs['images'].size()) # uncomment to debug

        return generated_text