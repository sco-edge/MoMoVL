import os, sys
PROJECT_PATH = os.getenv("PROJECT_PATH")
sys.path.insert(0, PROJECT_PATH)

import logging
import warnings

import gc, torch
from PIL import Image
from models.deepseek_vl2 import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from utils.deepseek_vl2.io import load_pil_images

from .base import BaseModel


class DeepSeekVL2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True
    NUM_EXPERTS = 64    # deepseek-vl2-tiny/small

    # def check_install(self):
    #     try:
    #         import deepseek_vl2  # noqa: F401
    #     except Exception as e:
    #         logging.critical(
    #             'Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
    #         raise e

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-tiny', **kwargs):
        # self.check_install()
        assert model_path is not None
        self.model_path = model_path

        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        ).eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=2048, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        self.kwargs['quant_method'] = None    # Prevent ValueError due to unused 'model_kwargs'
        self.kwargs['quant_config'] = None
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

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
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.hf.base import AutoHQQHFModel
            try:
                bitwidth = int(quant_method.split("_")[1])
            except Exception as e:
                print(f'[!] Error parsing HQQ bitwidth: {e}')
                sys.exit(1)
            
            if bitwidth == 4:
                quant_config = BaseQuantizeConfig(nbits=4, group_size=64) 
            elif bitwidth == 8:
                quant_config = BaseQuantizeConfig(nbits=8, group_size=64) 
            else:
                raise NotImplementedError(f'[!] HQQ bitwidth should be either 4 or 8')
            
            AutoHQQHFModel.quantize_model(self.model.language, quant_config=quant_config, compute_dtype=torch.bfloat16, device='cuda')

        elif quant_method == 'APEX' or quant_method == 'Fate':
            from models.deepseek_vl2 import DeepseekVLV2ForCausalLM_RT
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.model = DeepseekVLV2ForCausalLM_RT.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            ).eval()
            
            # Expert quantization after weight load
            print(f"[+] [{quant_method}] Quantizing experts after model initialization.")
            for MoE_block in self.model.language.model.layers[1:]:
                MoE_block.mlp.quant_method = quant_method
                MoE_block.mlp.quantize_experts()
    
        else:
            raise NotImplementedError(f"[!] {quant_method} is not supported.")

    # [DEV] Replace expert's linear layer with HQQ
    def quantize_experts(self, quant_config):
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
        q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
        q2_config = BaseQuantizeConfig(nbits=2, group_size=32)

        # "first_k_dense_replace": 1
        for layer_idx, layer in enumerate(self.model.language.model.layers[1:]):
            int4_experts = quant_config[str(layer_idx)]
            expert_config = [q4_config if i in int4_experts else q2_config for i in range(self.NUM_EXPERTS)]
            # Quantize routed_experts
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
            
            # Quantize shared_experts (INT4)
            expert = layer.mlp.shared_experts
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                quant_proj = HQQLinear(
                    linear_layer=getattr(expert, proj), 
                    quant_config=q4_config,
                    compute_dtype=torch.bfloat16,
                    device='cuda',
                    initialize=True,
                    del_orig=True)
                setattr(expert, proj, quant_proj)


    def prepare_inputs(self, message, dataset=None):

        if dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = '', []
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                # content = '<image>' * (image_idx-1) + '\n' + content
                content = '<image>' * (image_idx - 1) + '\n' + content
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    'Please select the correct answer from the options above.',
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                )
                content = content.replace('Question:', "")
                content = content.replace('Options:\n', "")
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace(
                        'Please select the correct answer from the options above.',
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                    )
                    content = content.replace('Question:', "")
                    content = content.replace('Options:\n', "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        else:

            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        pil_images = load_pil_images(conversation)

        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512
        )

        # run the model to get the response
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)