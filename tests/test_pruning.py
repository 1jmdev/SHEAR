import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

from shear.kernels import accumulate_segmented_gram, swiglu_weight
from shear.packing import load_packed, pack_experts, save_packed
from shear.pruning import prune, refit, select_channels


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(
        text_config=dict(
            hidden_size=64,
            num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )
    return Qwen3_5MoeForConditionalGeneration(config).to("cuda", torch.bfloat16).eval().requires_grad_(False)


def test_kernels_match_reference():
    torch.manual_seed(0)
    gate_up = torch.randn(300, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(300, device="cuda", dtype=torch.bfloat16)
    gate, up = gate_up.float().chunk(2, -1)
    activation = swiglu_weight(gate_up, weights)
    assert torch.allclose(activation.float(), weights.float()[:, None] * F.silu(gate) * up, atol=2e-2, rtol=2e-2)
    offsets = torch.tensor([0, 0, 120, 300], device="cuda", dtype=torch.int32)
    gram = torch.ones(3, 64, 64, device="cuda")
    accumulate_segmented_gram(activation, activation, offsets, gram)
    reference = activation.float()[120:].T @ activation.float()[120:]
    assert (gram[0] == 1).all()
    assert torch.allclose(gram[2] - 1, reference, rtol=1e-3, atol=1e-2)


def test_selection_meets_budget_and_refit_reconstructs():
    torch.manual_seed(0)
    features = torch.randn(4, 400, 32, device="cuda") @ torch.randn(4, 32, 32, device="cuda")
    hessian = features.transpose(1, 2) @ features / 400
    down = torch.randn(4, 16, 32, device="cuda") * torch.tensor([0.1, 1.0, 1.0, 3.0], device="cuda")[:, None, None]
    mask = select_channels(hessian, down, 96, 0.01, 32)
    assert int(mask.sum()) == 96
    assert int(mask[0].sum()) < int(mask[3].sum())
    refitted = refit(down, hessian, hessian, mask, 0.01)
    error = lambda weights: ((features @ (weights - down).transpose(1, 2)).square().mean())
    assert error(refitted) < error(down * mask[:, None, :])


def test_prune_and_pack_round_trip(tmp_path):
    model = tiny_model()
    windows = [torch.randint(0, 128, (4, 64)), torch.randint(0, 128, (3, 64))]
    masks, summary = prune(model, windows, 0.25, batch_size=2)
    assert masks.shape == (4, 8, 64)
    assert summary["budget_exact"]
    masked = model(input_ids=windows[0].cuda()).logits.float()
    assert torch.isfinite(masked).all()
    pack_experts(model, masks)
    assert torch.allclose(model(input_ids=windows[0].cuda()).logits.float(), masked, atol=5e-2)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"a": 0}, unk_token="a")))
    save_packed(model, tokenizer, tmp_path)
    reloaded = load_packed(tmp_path, "cuda")
    assert torch.allclose(reloaded(input_ids=windows[0].cuda()).logits.float(), masked, atol=5e-2)


def test_healing_reduces_error_and_keeps_pruned_channels_zero():
    windows = torch.randint(0, 128, (8, 64))
    _, plain = prune(tiny_model(), [windows], 0.5, batch_size=4)
    healed_model = tiny_model()
    masks, healed = prune(healed_model, [windows], 0.5, batch_size=4, heal_steps=100, heal_learning_rate=1e-4, heal_tokens=512)
    down_projection = healed_model.model.language_model.layers[0].mlp.experts.down_proj.transpose(1, 2)
    assert (down_projection[~masks[0].cuda()] == 0).all()
    assert sum(healed["relative_error_by_layer"]) < sum(plain["relative_error_by_layer"])
