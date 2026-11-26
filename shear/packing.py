import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText


class PackedExperts(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, widths, act_fn):
        super().__init__()
        self.widths = widths
        self.num_experts = len(widths)
        self.register_buffer("offsets", torch.tensor([0, *widths]).cumsum(0).int())
        self.gate_proj = nn.Parameter(gate_proj, requires_grad=False)
        self.up_proj = nn.Parameter(up_proj, requires_grad=False)
        self.down_proj = nn.Parameter(down_proj, requires_grad=False)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        flat_experts = top_k_index.flatten()
        order = flat_experts.argsort()
        tokens = order // top_k_index.shape[1]
        weights = top_k_weights.flatten()[order]
        start, offset = 0, 0
        for count, width in zip(flat_experts.bincount(minlength=self.num_experts).tolist(), self.widths):
            if count and width:
                rows = slice(offset, offset + width)
                selected = tokens[start : start + count]
                states = hidden_states[selected]
                hidden = self.act_fn(F.linear(states, self.gate_proj[rows])) * F.linear(states, self.up_proj[rows])
                output.index_add_(0, selected, (hidden @ self.down_proj[rows]) * weights[start : start + count, None])
            start += count
            offset += width
        return output


def pack_experts(model, masks):
    for layer, mask in zip(model.model.language_model.layers, masks.to(model.device)):
        experts = layer.mlp.experts
        expert, channel = mask.nonzero(as_tuple=True)
        layer.mlp.experts = PackedExperts(
            experts.gate_up_proj[expert, channel],
            experts.gate_up_proj[expert, channel + experts.intermediate_dim],
            experts.down_proj[expert, :, channel],
            mask.sum(1).tolist(),
            experts.act_fn,
        )


def save_packed(model, tokenizer, directory, shard_bytes=5 * 1024**3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shards = [{}]
    for name, tensor in model.state_dict().items():
        if sum(t.nbytes for t in shards[-1].values()) + tensor.nbytes > shard_bytes:
            shards.append({})
        shards[-1][name] = tensor
    for index, shard in enumerate(shards):
        save_file({name: tensor.contiguous().cpu() for name, tensor in shard.items()}, directory / f"model-{index:05d}.safetensors")
    widths = [layer.mlp.experts.widths for layer in model.model.language_model.layers]
    (directory / "expert_widths.json").write_text(json.dumps(widths))
    model.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def load_packed(directory, device="cuda"):
    directory = Path(directory)
    config = AutoConfig.from_pretrained(directory)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    widths = json.loads((directory / "expert_widths.json").read_text())
    for layer, layer_widths in zip(model.model.language_model.layers, widths):
        experts = layer.mlp.experts
        gate, up, down = (torch.empty(sum(layer_widths), experts.hidden_dim, device="meta") for _ in range(3))
        layer.mlp.experts = PackedExperts(gate, up, down, layer_widths, experts.act_fn)
    for shard in sorted(directory.glob("model-*.safetensors")):
        model.load_state_dict(load_file(shard, device=str(device)), strict=False, assign=True)
    return model.to(device).eval()
