# Part of MoMoVL, licensed under the MIT License (see LICENSE).
#
# The 'Fate' branch of get_bitwidth_assignment() is adapted from Fate_open
# (https://github.com/FFFzy/Fate_open), Copyright (c) 2026 Zhiyuan Fang,
# released under the MIT License (see LICENSES/MIT-Fate.txt).

import torch
from collections import Counter

def get_bitwidth_assignment(topk_ids, topk_weight, num_experts=64, quant_method=None):
    # configured parameters
    int2_rate = 0.5
    T1 = 0.6; T2 = 0.9

    # Assign bit-width to experts based on topk_weight [MoE-APEX]
    if quant_method == 'APEX':
        # norm gate to sum 1
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
        output = torch.zeros(num_experts, device=topk_weight.device, dtype=topk_weight.dtype)
        output.scatter_add_(0, topk_ids.view(-1), topk_weight.view(-1))
        sorted_weight, sorted_ids = output.sort(descending=True)

        # if s_i <= T1, consider expert important and use INT4 version
        # else if s_i <= T2, use INT2 version, else skip
        # where s_0 = 0, s_i = weight_0 + ... + weight_(i-1)
        int4_treshold = topk_weight.sum().item() * T1
        int2_treshold = topk_weight.sum().item() * T2
        int4_experts = set()
        int2_experts = set()
        current_sum = 0
        for expert_id, weight in zip(sorted_ids.tolist(), sorted_weight.tolist()):
            if current_sum <= int4_treshold:
                int4_experts.add(expert_id)
            elif current_sum <= int2_treshold:
                int2_experts.add(expert_id)
            else:
                break
            current_sum += weight
        
        bitwidth_assignment = [
            '4' if e in int4_experts else '2'
            # '2' if e in int2_experts else
            # '0' 
            for e in range(num_experts)
        ]
    
    # Assign bit-width to experts based on token-to-expert assignment [Fate]
    # Adapted from https://github.com/FFFzy/Fate_open/blob/main/models/DeepSeekMoE/modeling_deepseek.py#L746-L764
    elif quant_method == 'Fate':
        if topk_ids.shape[0] == 1:
            bitwidth_assignment = ['4'] * num_experts
        else:
            activated_expert_list = topk_ids.view(-1).tolist()
            activated_expert_list = Counter(activated_expert_list)
            most_common_items = activated_expert_list.most_common()

            # determine which experts to int2
            activated_expert_dict = dict(most_common_items)
            value_sum = sum(activated_expert_dict.values())
            target_sum = value_sum * int2_rate
            current_sum = 0
            int2_experts = set()
            for key, value in reversed(activated_expert_dict.items()):
                if current_sum + value > target_sum:
                    break
                current_sum += value
                int2_experts.add(key)
            
            bitwidth_assignment = [
                '2' if e in int2_experts else 
                '4' 
                for e in range(num_experts)
            ]
    else:
        raise NotImplementedError(f"[!] {quant_method} is not supported.")
    
    # cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    # num_activated = cnts.scatter_(1, topk_ids, 1).sum(dim=0).nonzero().shape[0]
    # print(f'[+] [{quant_method}] topk_ids.shape: {topk_ids.shape}, num_activated: {num_activated}, assignment: {Counter(bitwidth_assignment)}')
    return bitwidth_assignment