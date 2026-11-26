import random

import torch
import torch.nn.functional as F

from shear.kernels import accumulate_segmented_gram, swiglu_weight


class CaptureComplete(Exception):
    pass


def capture_layer_inputs(model, batches):
    captured = []

    def intercept(module, args, kwargs):
        captured.append((args[0], kwargs))
        raise CaptureComplete

    handle = model.model.language_model.layers[0].register_forward_pre_hook(intercept, with_kwargs=True)
    for batch in batches:
        try:
            model(input_ids=batch.to(model.device), use_cache=False)
        except CaptureComplete:
            pass
    handle.remove()
    return captured


def route(top_k_index, top_k_weights, experts):
    flat_experts = top_k_index.flatten()
    order = flat_experts.argsort()
    counts = torch.zeros(experts, dtype=torch.int32, device=flat_experts.device)
    counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int32))
    ends = counts.cumsum(0, dtype=torch.int32)
    return order // top_k_index.shape[1], top_k_weights.flatten()[order], ends, counts


def activations(experts, states, ends, weights):
    gate_up = torch._grouped_mm(states, experts.gate_up_proj.transpose(1, 2), offs=ends)
    return swiglu_weight(gate_up, weights)


def scaled_down_projection(text_model, index, down_projection):
    following = text_model.layers[index + 1].input_layernorm if index + 1 < len(text_model.layers) else text_model.norm
    return (1 + following.weight.float())[None, :, None] * down_projection.float()


def select_channels(hessian, scaled_weights, budget, damping, rounds, minimum_width=0):
    experts, channels, _ = hessian.shape
    regularization = damping * hessian.diagonal(dim1=1, dim2=2).mean().expand(experts) + 1e-10
    identity = torch.eye(channels, device=hessian.device)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian + regularization[:, None, None] * identity))
    weights = scaled_weights.transpose(1, 2).contiguous()
    alive = torch.ones(experts, channels, dtype=torch.bool, device=hessian.device)
    remaining = torch.tensor(experts * channels - budget, device=hessian.device)
    span = torch.arange(experts, device=hessian.device)
    for _ in range(rounds):
        diagonal = inverse.diagonal(dim1=1, dim2=2).clamp_min(1e-20)
        score = (weights.square().sum(2) / diagonal).masked_fill(~alive | (alive.sum(1, keepdim=True) <= minimum_width), torch.inf)
        best, channel = score.min(1)
        threshold = score.flatten().sort().values[(remaining - 1).clamp_min(0)]
        take = (best <= threshold) & (best.argsort().argsort() < remaining) & best.isfinite()
        column = inverse[span, :, channel]
        pivot = diagonal[span, channel]
        factor = take / pivot
        weights -= factor[:, None, None] * column[:, :, None] * weights[span, channel][:, None, :]
        inverse -= factor[:, None, None] * column[:, :, None] * column[:, None, :]
        alive[span, channel] &= ~take
        remaining -= take.sum()
    return alive


def refit(down_projection, hessian, cross, mask, damping):
    kept = mask.float()
    regularization = damping * hessian.diagonal(dim1=1, dim2=2).mean() + 1e-10
    system = hessian * kept[:, :, None] * kept[:, None, :] + torch.diag_embed(1 - kept + regularization * kept)
    target = (cross * kept[:, :, None]) @ down_projection.float().transpose(1, 2)
    return (torch.linalg.solve(system, target) * kept[:, :, None]).transpose(1, 2).to(down_projection.dtype)


def heal(layer, residuals, dense, mask, steps, learning_rate, tokens):
    keep = mask[:, None, :].float()
    with torch.inference_mode(False), torch.enable_grad():
        mlp = layer.mlp.float().requires_grad_(True)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
        for _ in range(steps):
            batch = random.randrange(len(residuals))
            residual = residuals[batch].flatten(0, 1)
            rows = torch.randint(len(residual), (tokens,), device=residual.device)
            states = layer.post_attention_layernorm(residual[rows]).float()
            target = (dense[batch].flatten(0, 1)[rows] - residual[rows]).float()
            loss = (mlp(states[None])[0] - target).square().mean() / target.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mlp.experts.down_proj.data.mul_(keep)
        mlp.to(torch.bfloat16).requires_grad_(False)


@torch.inference_mode()
def prune(
    model,
    domain_windows,
    ratio,
    damping=0.01,
    dense_target=1.0,
    domain_weights=None,
    batch_size=64,
    heal_steps=0,
    heal_learning_rate=2e-5,
    heal_tokens=16384,
    minimum_width=0,
):
    text_model = model.model.language_model
    layers = text_model.layers
    domains = len(domain_windows)
    weights = torch.tensor(domain_weights or [1 / domains] * domains, device=model.device)
    batches = [(domain, batch) for domain, windows in enumerate(domain_windows) for batch in windows.split(batch_size)]
    tokens = torch.tensor([float(windows.numel()) for windows in domain_windows], device=model.device)
    captured = capture_layer_inputs(model, [batch for _, batch in batches])
    student = [hidden for hidden, _ in captured]
    dense = [hidden.clone() for hidden in student]
    arguments = [kwargs for _, kwargs in captured]
    masks, errors = [], []

    for index, layer in enumerate(layers):
        experts = layer.mlp.experts
        count, channels = experts.num_experts, experts.intermediate_dim
        hessian = torch.zeros(domains, count, channels, channels, device=model.device)
        cross = torch.zeros_like(hessian)
        captured_dense, residuals, current = {}, [], [0]

        def capture_dense(module, args, output):
            captured_dense["states"] = args[0]

        def capture_residual(module, args):
            residuals.append(args[0])

        def accumulate(module, args, output):
            token_index, routing, ends, _ = route(args[1], args[2], count)
            student_activation = activations(module, args[0][token_index], ends, routing)
            dense_activation = activations(module, captured_dense["states"][token_index], ends, routing)
            offsets = F.pad(ends, (1, 0))
            accumulate_segmented_gram(student_activation, student_activation, offsets, hessian[current[0]])
            accumulate_segmented_gram(student_activation, dense_activation, offsets, cross[current[0]])

        for batch, (domain, _) in enumerate(batches):
            handle = experts.register_forward_hook(capture_dense)
            dense[batch] = layer(dense[batch], **arguments[batch])
            handle.remove()
            current[0] = domain
            handles = [
                experts.register_forward_hook(accumulate),
                layer.post_attention_layernorm.register_forward_pre_hook(capture_residual),
            ]
            layer(student[batch], **arguments[batch])
            for handle in handles:
                handle.remove()

        scale = (weights / tokens)[:, None, None, None]
        pooled_hessian = (scale * hessian).sum(0)
        pooled_cross = (scale * (dense_target * cross + (1 - dense_target) * hessian)).sum(0)
        del hessian, cross
        scaled = scaled_down_projection(text_model, index, experts.down_proj)
        budget = round(count * channels * (1 - ratio))
        mask = select_channels(pooled_hessian, scaled, budget, damping, channels, minimum_width)
        experts.down_proj.data.copy_(refit(experts.down_proj, pooled_hessian, pooled_cross, mask, damping))
        if heal_steps:
            heal(layer, residuals, dense, mask, heal_steps, heal_learning_rate, heal_tokens)
        for batch, residual in enumerate(residuals):
            student[batch] = residual + layer.mlp(layer.post_attention_layernorm(residual))
        errors.append(sum((s - d).float().square().sum() for s, d in zip(student, dense)) / sum(d.float().square().sum() for d in dense))
        masks.append(mask)

    masks = torch.stack(masks).cpu()
    experts = layers[0].mlp.experts
    budget = round(experts.num_experts * experts.intermediate_dim * (1 - ratio))
    return masks, {
        "budget_exact": bool((masks.sum((1, 2)) == budget).all()),
        "widths_min": masks.sum(2).amin(1).tolist(),
        "widths_max": masks.sum(2).amax(1).tolist(),
        "relative_error_by_layer": torch.stack(errors).tolist(),
        "routed_parameters_removed": int((~masks).sum()) * 3 * experts.hidden_dim,
        "routed_parameters_total": masks.numel() * 3 * experts.hidden_dim,
    }
