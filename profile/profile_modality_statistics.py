import os, sys
PROJECT_PATH = os.getenv("PROJECT_PATH")
sys.path.insert(0, PROJECT_PATH)

# --- stdlib ---
import argparse
import csv
import gc
import json
import random
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- third-party ---
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# --- local (need sys.path above) ---
from models.deepseek_vl2 import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from models.molmoe import MolmoImageProcessor, MolmoProcessor, MolmoForCausalLM

# explicitly enable autograd 
torch.set_grad_enabled(True)


@dataclass
class Sample:
    sample_id: str     # dataset row id, written to the "id" column of the stats CSV
    image: Image.Image
    prompt: str        # instructions and question 
    response: str       # the "gpt" turn, used as the teacher-forced target

@dataclass
class ProbeInputs:
    model_kwargs: Any               # forward()/prepare_*() kwargs, WITHOUT labels
    labels: torch.Tensor            # (1, seq_len); -100 except response positions
    visual_mask: torch.Tensor       # (seq_len,) bool; True at visual-token positions


def load_samples_COCO(num_samples: int=128) -> List[Sample]:
    # COCO subset of ShareGPT4V
    SHAREGPT4V_REPO = "Lin-Chen/ShareGPT4V"
    COCO_IMAGE_BASE = "https://images.cocodataset.org"
    SEED = 42

    ds = load_dataset(SHAREGPT4V_REPO, "ShareGPT4V", split="train")
    coco_idx = [i for i, path in enumerate(ds["image"]) if str(path).startswith("coco/")]
    random.Random(SEED).shuffle(coco_idx)
    chosen = coco_idx[:num_samples]

    cache_dir = os.path.join(PROJECT_PATH, "data", "coco_cache")
    os.makedirs(cache_dir, exist_ok=True)

    samples = []
    for idx in chosen:
        row = ds[idx]
        rel = str(row["image"])[len("coco/"):]          # train2017/000000000009.jpg
        local = os.path.join(cache_dir, rel.replace("/", "_"))
        if not os.path.exists(local):
            resp = requests.get(f"{COCO_IMAGE_BASE}/{rel}", verify=False, timeout=30)
            resp.raise_for_status()
            with open(local, "wb") as f:
                f.write(resp.content)
        turns = row["conversations"]
        prompt = next((t["value"] for t in turns if t.get("from") == "human"), "")
        response = next((t["value"] for t in turns if t.get("from") == "gpt"), "")
        # the human turn carries an "<image>" placeholder at either end; each
        # model inserts its own image tokens, so drop it here
        prompt = prompt.replace("<image>", "").strip()
        if prompt and response.strip():
            samples.append(Sample(sample_id='coco_'+str(row["id"]),
                                  image=Image.open(local).convert("RGB"),
                                  prompt=prompt, response=response.strip()))
    return samples


class BaseModalityProfiler(ABC):
    # routing constants known per model
    TOP_K: int = 0
    N_EXPERTS: int = 0
    USE_AUTOCAST: bool = False

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.topk = self.TOP_K
        self.n_experts = self.N_EXPERTS
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self._load_model_and_processor(model_path)

        self._captured: Dict[int, torch.Tensor] = {}
        self._captured_routing: Dict[int, Any] = {}

        # DeepSeek-VL2 sets first_k_dense_replace=1, so its layer 0 is not MoE
        layers = self._decoder_layers()
        self.expert_layer_ids = [i for i, l in enumerate(layers) if hasattr(l.mlp, "experts")]
        for i in self.expert_layer_ids:
            mlp = layers[i].mlp
            assert len(mlp.experts) == self.n_experts, (len(mlp.experts), self.n_experts)
            mlp.register_forward_hook(self._make_activation_hook(i))
            mlp.gate.register_forward_hook(self._make_router_hook(i))
        print(f"[{type(self).__name__}] hooked {len(self.expert_layer_ids)} MoE mlp(s) at layers {self.expert_layer_ids}; ")

    # -- model-specific ------------------------------------------------

    @abstractmethod
    def _load_model_and_processor(self, model_path: str) -> None:
        """Must set self.model (+ processor/tokenizer)."""

    @abstractmethod
    def _decoder_layers(self) -> torch.nn.Module:
        """The ModuleList of decoder layers."""

    @abstractmethod
    def build_inputs(self, sample: Sample) -> ProbeInputs:
        """One (image, response) Sample -> ProbeInputs."""

    @abstractmethod
    def _routing_from_gate(self, raw: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """MoE Gate output -> (topk_idx, topk_weight), each (n_tokens, self.topk)."""

    @abstractmethod
    def _forward_and_compute_loss(self, probe_inputs: ProbeInputs) -> torch.Tensor:
        """Run the model and return the SFT loss (fp32, response-only)."""

    # -- hooks ---------------------------------------------------------

    def _make_activation_hook(self, idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden.retain_grad()
            self._captured[idx] = hidden
        return hook

    def _make_router_hook(self, idx: int):
        def hook(module, inputs, output):
            self._captured_routing[idx] = output
        return hook

    # -- measurement ---------------------------------------------------

    @staticmethod
    def format_tensor_row(row, is_float=False):
        if is_float:
            return "-".join([f"{x:.6f}" for x in row])
        else:
            return "-".join([str(int(x)) for x in row])

    @staticmethod
    def _mean_abs(x: torch.Tensor) -> torch.Tensor:
        """(1, seq_len, d) -> (seq_len,) per-token mean absolute value."""
        out = x.float().abs().mean(dim=-1)
        return out[0] if out.dim() == 2 else out

    def compute_layer_gradients(self, sample: Sample
                                ) -> Tuple[Dict[int, Dict[str, float]],
                                           Dict[int, np.ndarray],
                                           Optional[pd.DataFrame]]:
        """One forward+backward. Returns
        (mean_stats, sum_stats, token_counts, expert_importance, expert_stats):
          mean_stats        {layer: {visual|text|prompt|response: avg |grad|}}
          sum_stats         {layer: {visual|text: total |grad| over that group}}
          token_counts      {n_vis, n_text} token counts for this sample
          expert_stats      one row per token: id, is_visual_token, and
                            layer_{L}_topk_idx / layer_{L}_topk_weight
        """
        self.model.zero_grad(set_to_none=True)
        self._captured.clear()
        self._captured_routing.clear()

        probe_inputs = self.build_inputs(sample)
        with torch.autocast(device_type="cuda", enabled=self.USE_AUTOCAST, dtype=self.dtype):
            loss = self._forward_and_compute_loss(probe_inputs)
        loss.backward()

        visual_mask = probe_inputs.visual_mask.to(self.device)
        text_mask = ~visual_mask
        response_mask = (probe_inputs.labels[0] != -100).to(self.device)
        masks = {"visual": visual_mask, "text": text_mask,
                 "prompt": text_mask & ~response_mask, "response": response_mask}

        mean_stats: Dict[int, Dict[str, float]] = {}
        sum_stats: Dict[int, Dict[str, float]] = {}
        topk_results: Dict[str, List[str]] = {}

        for layer_id in self.expert_layer_ids:
            hidden = self._captured.get(layer_id)
            if hidden is None or hidden.grad is None:
                mean_stats[layer_id] = {k: float("nan") for k in masks}
                sum_stats[layer_id] = {k: float("nan") for k in ("visual", "text")}
                continue

            per_token = self._mean_abs(hidden.grad)
            mean_stats[layer_id] = {name: (per_token[m].mean().item() if m.any() else float("nan"))
                            for name, m in masks.items()}
            sum_stats[layer_id] = {name: per_token[masks[name]].sum().item()
                                   for name in ("visual", "text")}

            topk_idx, topk_w = self._routing_from_gate(self._captured_routing[layer_id])
            topk_idx = topk_idx.detach().reshape(-1, topk_idx.shape[-1]).long()
            topk_w = topk_w.detach().reshape(-1, topk_w.shape[-1]).float()

            topk_results[f'layer_{layer_id}_topk_idx'] = [
                self.format_tensor_row(r) for r in topk_idx.tolist()]
            topk_results[f'layer_{layer_id}_topk_weight'] = [
                self.format_tensor_row(r, is_float=True) for r in topk_w.tolist()]

        expert_stats = None
        if topk_results:
            expert_stats = pd.DataFrame(topk_results)
            expert_stats.insert(0, "id", sample.sample_id)
            expert_stats.insert(1, "is_visual_token", probe_inputs.visual_mask.cpu().tolist())

        # Release activations, grads, and GPU memory
        self._captured.clear()
        self._captured_routing.clear()
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        token_counts = {"n_vis": int(visual_mask.sum()), "n_text": int(text_mask.sum())}

        return mean_stats, sum_stats, token_counts, expert_stats

    def run(self, samples: List[Sample], out_path: str) -> Dict[str, Any]:
        GROUPS = ("visual", "text", "prompt", "response")
        SUM_GROUPS = ("visual", "text")
        acc = {g: {i: [] for i in self.expert_layer_ids} for g in GROUPS}
        sum_acc = {g: {i: [] for i in self.expert_layer_ids} for g in SUM_GROUPS}
        n_vis_list, n_text_list = [], []
        expert_stats_list: List[pd.DataFrame] = []

        for sample_id, sample in enumerate(tqdm(samples, desc="profiling samples", unit="sample")):
            try:
                layer_grads, layer_sums, token_counts, expert_stats = \
                    self.compute_layer_gradients(sample)
            except Exception as e:
                tqdm.write(f"[sample {sample_id}] failed: {e}")
                continue
            for idx, vals in layer_grads.items():
                if any(vals[g] != vals[g] for g in GROUPS):   # NaN
                    continue
                for g in GROUPS:
                    acc[g][idx].append(vals[g])
                for g in SUM_GROUPS:
                    sum_acc[g][idx].append(layer_sums[idx][g])
            n_vis_list.append(token_counts["n_vis"])
            n_text_list.append(token_counts["n_text"])
            if expert_stats is not None:
                expert_stats_list.append(expert_stats)

        per_layer, rows = [], []
        for idx in self.expert_layer_ids:
            if not acc["visual"][idx]:
                continue
            mean_grad = {g: sum(acc[g][idx]) / len(acc[g][idx]) for g in GROUPS}
            sum_grad = {g: sum(sum_acc[g][idx]) / len(sum_acc[g][idx]) for g in SUM_GROUPS}
            mean_ratio = mean_grad["text"] / mean_grad["visual"] if mean_grad["visual"] > 0 else float("inf")
            sum_ratio = sum_grad["text"] / sum_grad["visual"] if sum_grad["visual"] > 0 else float("inf")
            per_layer.append({"layer": idx, "mean_visual_grad": mean_grad["visual"],
                              "mean_text_grad": mean_grad["text"],
                              "mean_prompt_grad": mean_grad["prompt"],
                              "mean_response_grad": mean_grad["response"],
                              "sum_visual_grad": sum_grad["visual"],
                              "sum_text_grad": sum_grad["text"],
                              "mean_ratio": mean_ratio, "sum_ratio": sum_ratio})
            rows.append((idx, mean_grad, sum_grad, mean_ratio, sum_ratio))

        if not per_layer:
            raise RuntimeError("No samples succeeded.")

        n_vis = sum(n_vis_list) / len(n_vis_list) if n_vis_list else float("nan")
        n_text = sum(n_text_list) / len(n_text_list) if n_text_list else float("nan")

        # The last layer has a structurally zero visual gradient gained from loss 0 (no label)
        print("\n=== Modality sensitivity ===")
        print(f"  tokens/sample: visual={n_vis:.0f} text={n_text:.0f} ({n_vis/n_text:.2f}:1)")
        print(f"{'layer':>5} | {'mean_vis':>10} {'mean_txt':>10} {'mean_ratio':>11} | {'sum_vis':>10} {'sum_txt':>10} {'sum_ratio':>11}")
        for idx, mean_grad, sum_grad, mean_ratio, sum_ratio in rows:
            print(f"{idx:>5} | {mean_grad['visual']:>10.3e} {mean_grad['text']:>10.3e} {mean_ratio:>11.2f} "
                  f"| {sum_grad['visual']:>10.3e} {sum_grad['text']:>10.3e} {sum_ratio:>11.2f} ")
        
        finite = [p for p in per_layer if abs(p["mean_ratio"]) != float("inf")
                  and p["mean_ratio"] == p["mean_ratio"]]
        degenerate = [p["layer"] for p in per_layer if p not in finite]
        mean_overall = sum(p["mean_ratio"] for p in finite) / len(finite) if finite else float("nan")
        sum_overall = sum(p["sum_ratio"] for p in finite) / len(finite) if finite else float("nan")
        print(f"\n[Cross-layer ratio (text / visual)] mean: {mean_overall:.2f}x | sum: {sum_overall:.2f}x"
              f"[{len(finite)}/{len(self.expert_layer_ids)} hooked layers]")

        results = {"model_path": self.model_path,
                   "num_samples": len(acc["visual"][self.expert_layer_ids[0]]),
                   "n_vis": n_vis, "n_text": n_text,
                   "hooked_layers": self.expert_layer_ids, "per_layer": per_layer,
                   "overall_mean_ratio": mean_overall, "overall_sum_ratio": sum_overall,
                   "degenerate_layers": degenerate}

        if expert_stats_list:
            results["expert_stats"] = self._save_expert_stats(expert_stats_list, out_path)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")
        
        return results

    def _save_expert_stats(self, expert_stats_list, out_path) -> Dict[str, Any]:
        aggregated = pd.concat(expert_stats_list, axis=0, ignore_index=True)
        n_visual = int(aggregated["is_visual_token"].sum())
        print("\n=== Per-token expert routing stats ===")
        print(f"  rows (tokens): {len(aggregated)} (visual {n_visual}, text {len(aggregated) - n_visual})")

        csv_path = out_path.replace("profile_modality_sensitivity", "profile_experts").replace(".json", ".csv")
        aggregated.to_csv(csv_path, index=False)

        return {"n_rows": len(aggregated), "n_visual_tokens": n_visual,
                "n_samples": len(expert_stats_list), "stats_csv": csv_path}


class DeepSeekModalityProfiler(BaseModalityProfiler):
    TOP_K = 6
    N_EXPERTS = 64
    USE_AUTOCAST = False

    def _load_model_and_processor(self, model_path: str) -> None:
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cuda').eval()
        self._make_moe_differentiable()

    def _decoder_layers(self) -> torch.nn.Module:
        return self.model.language.model.layers

    def _make_moe_differentiable(self) -> None:
        # verbatim copy of 'moe_infer' without @torch.no_grad() and 'ep_size > 1' branches to enable gradient computation
        def moe_infer_grad(self, x, topk_ids, topk_weight):
            cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
            cnts.scatter_(1, topk_ids, 1)
            tokens_per_expert = cnts.sum(dim=0)
            idxs = topk_ids.view(-1).argsort()
            sorted_tokens = x[idxs // topk_ids.shape[1]]
            tokens_per_expert = tokens_per_expert.cpu().numpy()

            outputs = []
            start_idx = 0
            for i, num_tokens in enumerate(tokens_per_expert):
                end_idx = start_idx + num_tokens
                if num_tokens == 0:
                    continue
                expert = self.experts[i + self.ep_rank * self.experts_per_rank]
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
                expert_out = expert(tokens_for_this_expert)
                outputs.append(expert_out)
                start_idx = end_idx

            outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

            new_x = torch.empty_like(outs)
            new_x[idxs] = outs
            final_out = (
                new_x.view(*topk_ids.shape, -1)
                .type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1))
                .sum(dim=1)
                .type(new_x.dtype)
            )
            return final_out

        n = 0
        for module in self.model.modules():
            if type(module).__name__ == "DeepseekV2MoE":
                assert module.ep_size == 1, f"ep_size={module.ep_size}"
                module.moe_infer = types.MethodType(moe_infer_grad, module)
                n += 1
        print(f"[DeepSeek] rebound moe_infer on {n} MoE block(s) (dropped @torch.no_grad)")

    def _routing_from_gate(self, raw):
        # MoEGate returns (topk_idx, topk_weight, aux_loss) cf. modeling_deepseek.py:424-528.
        return raw[0], raw[1]

    def _build_conversation(self, sample: Sample, assistant_text: str):
        return [
            {"role": "<|User|>", "content": f"<image>\n{sample.prompt}",
             "images": [sample.image]},
            {"role": "<|Assistant|>", "content": assistant_text},
        ]

    def build_inputs(self, sample: Sample) -> ProbeInputs:
        # Prefix pass with an EMPTY assistant turn gives the exact token length up through "<|Assistant|>:"
        prefix = self.processor(conversations=self._build_conversation(sample, ""),
                                images=[sample.image], force_batchify=True,
                                system_prompt="")
        response_start = prefix["input_ids"].shape[1]

        full = self.processor(conversations=self._build_conversation(sample, sample.response),
                              images=[sample.image], force_batchify=True,
                              system_prompt="").to(self.model.device)
        labels = full["input_ids"].clone()
        labels[:, :response_start] = -100
        return ProbeInputs(model_kwargs=full, labels=labels, visual_mask=full.images_seq_mask[0])

    def _forward_and_compute_loss(self, probe_inputs: ProbeInputs) -> torch.Tensor:
        inputs_embeds = self.model.prepare_inputs_embeds(**probe_inputs.model_kwargs)
        return self.model.language(
            inputs_embeds=inputs_embeds,
            attention_mask=probe_inputs.model_kwargs["attention_mask"],
            labels=probe_inputs.labels,
        ).loss


class MolmoEModalityProfiler(BaseModalityProfiler):
    TOP_K = 8
    N_EXPERTS = 64
    # Autocast required: the image processor emits fp32 image tensors against bf16
    USE_AUTOCAST = True

    # processor defaults, preprocessing_molmo.py:76-77
    MESSAGE_FORMAT = "role"
    ALWAYS_START_WITH_SPACE = True

    def _load_model_and_processor(self, model_path: str) -> None:
        image_processor = MolmoImageProcessor.from_pretrained(f'{PROJECT_PATH}/models/molmoe')
        self.tokenizer = GPT2TokenizerFast.from_pretrained(f'{PROJECT_PATH}/models/molmoe')
        self.processor = MolmoProcessor(image_processor=image_processor,
                                        tokenizer=self.tokenizer)
        self.model = MolmoForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='cuda').eval()

    def _decoder_layers(self) -> torch.nn.Module:
        return self.model.model.transformer.blocks

    def _routing_from_gate(self, raw):
        # mlp.gate: nn.Linear -> router_logits (n_tokens, n_experts) cf. modeling_molmoe.py:827-833
        probs = F.softmax(raw.float(), dim=-1)
        w, idx = torch.topk(probs, self.topk, dim=-1)
        return idx, w.to(self.dtype)

    def build_inputs(self, sample: Sample) -> ProbeInputs:
        prompt_tokens = self.processor.get_tokens_input(
            sample.prompt, self.MESSAGE_FORMAT, self.ALWAYS_START_WITH_SPACE)
        response_tokens = self.tokenizer.encode(sample.response, add_special_tokens=False)

        prompt_out = self.processor.process(images=[sample.image], tokens=prompt_tokens)
        full_out = self.processor.process(images=[sample.image], tokens=prompt_tokens + response_tokens)
        prompt_len = prompt_out["input_ids"].shape[0]

        model_kwargs = {k: v.to(self.model.device).unsqueeze(0) for k, v in full_out.items()}
        input_ids = model_kwargs["input_ids"]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        seq_len = input_ids.shape[1]
        # cf. modeling_molmoe.py:1904-1922
        image_input_idx = model_kwargs["image_input_idx"]
        valid = image_input_idx >= 0
        visual_positions = image_input_idx[valid].reshape(-1).long()
        assert int(visual_positions.max()) < seq_len, "image position out of range"

        visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        visual_mask[visual_positions] = True
        return ProbeInputs(model_kwargs=model_kwargs, labels=labels, visual_mask=visual_mask)

    def _forward_and_compute_loss(self, probe_inputs: ProbeInputs) -> torch.Tensor:
        # Run the model without `labels` and compute the SFT loss here 
        logits = self.model(**probe_inputs.model_kwargs).logits
        return F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)).float(),
            probe_inputs.labels[..., 1:].reshape(-1),
            ignore_index=-100,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default='deepseek', choices=['deepseek', 'molmoe'])
    parser.add_argument("--num-samples",  type=int, default=128)
    args = parser.parse_args()

    MODEL_MAP = {'deepseek': 'deepseek-ai/deepseek-vl2-tiny',
                 'molmoe': 'allenai/MolmoE-1B-0924'}
    PROFILER_MAP = {'deepseek': DeepSeekModalityProfiler,
                    'molmoe': MolmoEModalityProfiler}

    profiler = PROFILER_MAP[args.model](model_path=MODEL_MAP[args.model])
    samples = load_samples_COCO(num_samples=args.num_samples)

    model_name = MODEL_MAP[args.model].split('/')[-1]
    out_path = (f'{PROJECT_PATH}/data/modal_statistics/{args.model}/profile_modality_sensitivity_COCO{args.num_samples}_{model_name}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    profiler.run(samples, out_path=out_path)
