import os
PROJECT_PATH = os.getenv("PROJECT_PATH")

import math, json
import pandas as pd
import numpy as np

# 64 experts both for DeepSeek-VL2 and MolmoE
N_ROUTED_EXPERTS = 64
MODEL_NAME = {
    'molmoe': 'MolmoE-1B-0924',
    'deepseek': 'deepseek-vl2-tiny'
}

# Threshold to determine experts quantized to INT4
INT4_RATE = {
    'molmoe': {
        'weight': 0.6,  # T1 in MoE-APEX
        'index': 0.5    # (1 - int2_rate) in Fate
    },
    'deepseek': {
        'weight': 0.6,
        'index': 0.5
    }
}
# Scaling factor for visual/text-centric metrics
MODALITY_SCALE_FACTOR = None

def get_modal_weights(calib_name):
    modality_scale_factor = {}
    for target_model in ['molmoe', 'deepseek']:
        w_vis_list = []
        w_txt_list = []
        model_name = MODEL_NAME[target_model]
        data_path = f'{PROJECT_PATH}/data/modal_statistics/{target_model}/profile_modality_sensitivity_{calib_name}_{model_name}.json'
        with open(data_path, 'r') as f:
            per_layer = json.load(f)['per_layer']
            for stat in per_layer:
                print(f"[{target_model}] {stat['layer']}: {stat['sum_visual_grad']:.10f} / {stat['sum_text_grad']:.10f}")
                w_vis = stat['sum_visual_grad'] / (stat['sum_visual_grad'] + stat['sum_text_grad'])
                w_vis_list.append(w_vis)
                w_txt_list.append(1-w_vis)
        
        modality_scale_factor[target_model] = {'visual': w_vis_list, 'text': w_txt_list}
    
    return modality_scale_factor

def get_activation_data(target_model, calib_name):
    model_name = MODEL_NAME[target_model]
    csv_path = f'{PROJECT_PATH}/data/modal_statistics/{target_model}/profile_experts_{calib_name}_{model_name}.csv'

    calib_df = pd.read_csv(csv_path)
    calib_vis = calib_df[calib_df['is_visual_token']]
    calib_txt = calib_df[calib_df['is_visual_token'] == False]

    return calib_vis, calib_txt

def parse_topk_idx(data):
    topk_idx_per_layer = []
    topk_idx_columns = data.columns[data.columns.str.contains('topk_idx')]
    for c in topk_idx_columns:
        topk_idx_per_layer.append(np.array([[int(x) for x in topk_idx.split('-')] for topk_idx in list(data[c])]))
    return np.stack(topk_idx_per_layer, axis=0)

def parse_topk_weight(data):
    topk_weight_per_layer = []
    topk_weight_columns = data.columns[data.columns.str.contains('topk_weight')]
    for c in topk_weight_columns:
        topk_weight_per_layer.append(np.array([[float(x) for x in topk_weight.split('-')] for topk_weight in list(data[c])], dtype=np.float32))
    return np.stack(topk_weight_per_layer, axis=0)

# compute sum of normalized gating weights of each expert (per-token / per-layer)
def compute_experts_gating_weights(topk_idx, topk_weight):
    n_layers, n_tokens, n_experts_per_tok = topk_idx.shape
    expert_weights_sum = np.zeros((n_layers, N_ROUTED_EXPERTS), dtype=float)

    for layer_idx in range(n_layers):
        # norm gate to sum 1
        denominator = topk_weight[layer_idx].sum(axis=1, keepdims=True) + 1e-20
        norm_weight = topk_weight[layer_idx] / denominator
        output = np.zeros(N_ROUTED_EXPERTS, dtype=norm_weight.dtype)
        np.add.at(output, topk_idx[layer_idx].ravel(), norm_weight.ravel())
        expert_weights_sum[layer_idx] = output / n_tokens      # normalize by group size

    return expert_weights_sum

# compute activation frequency of each expert (per-token / per-layer)
def compute_activation_freq(topk_idx):
    n_layers, n_tokens, n_experts_per_tok = topk_idx.shape
    freq = np.zeros((n_layers, N_ROUTED_EXPERTS), dtype=float)

    for layer_idx in range(n_layers):
        counts = np.bincount(topk_idx[layer_idx].ravel(), minlength=N_ROUTED_EXPERTS)
        freq[layer_idx] = counts / n_tokens           # normalize by group size
    
    return freq

# Select hot experts based on cumulative sum of given metrics
def select_hot_experts(metrics, int4_rate):
    n_layers = metrics.shape[0]
    int4_treshold = metrics[0].sum() * int4_rate
    quant_config = {}
    for layer_idx in range(n_layers):
        current_sum = 0
        int4_experts = []
        for expert_id in metrics[layer_idx].argsort()[::-1].tolist():
            if current_sum <= int4_treshold:
                int4_experts.append(expert_id)
            else:
                break
            current_sum += metrics[layer_idx][expert_id]
        quant_config[layer_idx] = sorted(int4_experts)

    return quant_config

def generate_quant_config(target_model, calib_name, config_type, vis_idx, txt_idx, vis_weight, txt_weight):
    int4_rate = INT4_RATE[target_model][config_type]
    vis_scale = np.array(MODALITY_SCALE_FACTOR[target_model]['visual'])
    txt_scale = np.array(MODALITY_SCALE_FACTOR[target_model]['text'])

    # Select INT4 experts based on weighted routing weights
    if config_type == 'weight':
        vis_expert_weight = compute_experts_gating_weights(vis_idx, vis_weight)
        txt_expert_weight = compute_experts_gating_weights(txt_idx, txt_weight)
        # Combine visual/text-centric activation profile
        metrics = (vis_expert_weight * vis_scale[:, None]) + (txt_expert_weight * txt_scale[:, None])

        quant_config = select_hot_experts(metrics, int4_rate)

    # Select INT4 experts based on weighted activation frequency
    elif config_type == 'index':
        vis_freq = compute_activation_freq(vis_idx)  
        txt_freq = compute_activation_freq(txt_idx)  
        # Combine visual/text-centric activation profile
        metrics = (vis_freq * vis_scale[:, None]) + (txt_freq * txt_scale[:, None])

        quant_config = select_hot_experts(metrics, int4_rate)

    # Save quant_config to file
    with open(f'{PROJECT_PATH}/data/quant_config/{target_model}/quant_modal_scale_sum_grad_{config_type}_{calib_name}.json', 'w') as f:
        json.dump(quant_config, f, indent=4)


if __name__ == '__main__':
    
    calib_name = 'COCO128'
    MODALITY_SCALE_FACTOR = get_modal_weights(calib_name)
    
    for target_model in ['molmoe', 'deepseek']:
        # Read activation data
        df_vis, df_txt = get_activation_data(target_model, calib_name)
        vis_idx = parse_topk_idx(df_vis)
        txt_idx = parse_topk_idx(df_txt)
        vis_weight = parse_topk_weight(df_vis)
        txt_weight = parse_topk_weight(df_txt)
        
        for config_type in ['weight', 'index']:
            # Generate quantization config
            print(f'[+] generate_quant_config: {target_model} - {config_type}')
            generate_quant_config(target_model, calib_name, config_type, vis_idx, txt_idx, vis_weight, txt_weight)
