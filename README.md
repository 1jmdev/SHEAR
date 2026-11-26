# SHEAR

**S**tructured **H**ealing and **E**xpert-width **A**daptive **R**eduction

SHEAR makes Mixture-of-Experts language models smaller and cheaper to run by narrowing their experts instead of removing them. Every expert, the router and the number of experts stay the same. Each expert loses the hidden channels that contribute least to the layer's output, and the weights that remain are corrected so they make up for what was removed.

The method is designed to be fast. It needs no large-scale training: one pass over calibration data and a short healing step per layer. On a single H200, pruning Qwen3.5-35B-A3B takes under 10 minutes.

## What it does

A routed expert is a SwiGLU feed-forward block with `gate`, `up` and `down` projections. One hidden channel is a row of `gate`, a row of `up` and a column of `down`. SHEAR removes whole channels, so pruned experts become genuinely smaller dense matrices:

- **Physically smaller model.** Pruned channels are deleted, not zeroed. Each layer stores its experts packed contiguously, at uneven widths, with an offset table.
- **Same architecture.** No experts are dropped and the router is unchanged. The model loads and runs like the original.
- **Exact budget.** Every layer removes exactly the requested share of routed-expert parameters.

## How it works

SHEAR processes the model one layer at a time, shallow to deep, so every layer is calibrated on the outputs of the layers already pruned before it.

1. **Statistics.** Calibration windows from several domains run through the layer. Triton kernels accumulate, per expert and in fp32, the Gram matrix of the routing-weighted SwiGLU activations, plus their cross-correlation with the dense model's activations. Each domain is normalised by its token count and weighted separately, so no single domain dominates the selection.
2. **Adaptive width allocation.** A greedy Optimal-Brain-Surgeon selection runs over all experts of the layer jointly. At every step it removes the channel whose removal, after optimal compensation, increases the output error the least, and it updates the inverse Hessian of the affected expert. Experts that matter end up wide and redundant ones narrow, subject to a minimum width that keeps rarely used experts alive.
3. **Least-squares refit.** Each expert's remaining down-projection is solved in closed form so the kept channels reproduce the dense model's expert output.
4. **Layer healing.** A short Adam run (64 steps by default) on the layer's kept MLP weights matches the dense layer's output on calibration tokens. Pruned channels are held at zero throughout.
5. **Packing.** The kept channels are gathered into contiguous per-layer buffers, checked against the masked model, and saved as safetensors with the expert widths.

The whole loop runs on the GPU without host synchronisation: routing, statistics, selection, refit and healing all stay on the device.

### Calibration data

The default `extended` set uses training splits only, so it never overlaps the evaluation data:

| Domain | Source |
|---|---|
| General | WikiText-2 |
| Reasoning | GSM8K, MathQA |
| Mathematics | MATH |
| Code | MBPP |
| Science | SciQ |
| Dialogue | UltraChat 200k |
| Multilingual | Aya |
| Knowledge | MMLU auxiliary training set |

The default domain weights are 0.4 for general text, 0.15 each for multilingual and knowledge, and 0.06 for each of the other domains.

## Results

Qwen3.5-35B-A3B, 25% of routed-expert parameters removed (8.05B of 32.21B). The figures are relative changes against the unpruned model: perplexity on held-out test splits (higher is worse) and accuracy on a 1,710-question MMLU subset of 30 questions per subject (lower is worse).

| Metric | Change |
|---|---|
| Perplexity, general | +3.5% |
| Perplexity, reasoning | +3.3% |
| Perplexity, mathematics | +2.0% |
| Perplexity, code | +2.4% |
| Perplexity, science | +2.2% |
| Perplexity, dialogue | +1.4% |
| Perplexity, multilingual | +6.7% |
| MMLU accuracy | −2.1% |

Pruning plus this evaluation took about 9.3 minutes on one H200.

### Speed against quality

SHEAR deliberately trades some quality for a short pruning time. Methods that come close to the original model at this compression ratio, such as SlimMoE, recover quality by distilling the pruned experts over hundreds of millions to billions of tokens, which takes hours to days of GPU time. SHEAR uses only closed-form corrections and a few dozen local optimisation steps per layer, so it finishes in minutes, but it leaves a gap of a few percent. Multilingual text loses the most, because 25% narrower experts lack the capacity to cover many languages without retraining.

Lower ratios keep the model much closer to the original: at 10%, everything except multilingual perplexity stays within about 1%.

## Usage

The pipeline runs on [Modal](https://modal.com). Everything is cached on the `shear-cache` volume: the model, tokenised calibration and evaluation windows, benchmark datasets and baseline scores, so GPU time is spent only on pruning and evaluation.

```sh
uv run modal run modal_app.py::prefetch_assets    # CPU only: download the model and build all data once
uv run modal run modal_app.py::smoke_test         # short end-to-end check on an H200
uv run modal run modal_app.py::prune_model        # full prune, lm-eval benchmarks, packed export
```

`prune_model` options:

| Option | Default | Meaning |
|---|---|---|
| `--ratio` | `0.25` | Share of routed-expert channels removed in every layer |
| `--domain-set` | `extended` | Calibration set: `paper`, `broad` or `extended` |
| `--domain-weights` | `0.4,0.06,0.06,0.06,0.06,0.06,0.15,0.15` | Per-domain weights, in calibration-set order |
| `--minimum-width` | `128` | Smallest number of channels any expert keeps |
| `--heal-steps` | `64` | Adam steps per layer; `0` disables healing |
| `--heal-learning-rate` | `2e-5` | Healing learning rate |
| `--damping` | `0.01` | Hessian damping for selection and refit |
| `--export` | `true` | Save the packed model |

Each run writes `masks.pt`, `report.json` and the packed model to `/cache/runs/<name>/`. A packed model is loaded with `shear.packing.load_packed`.

For quick comparisons, `explore` runs one configuration and reports per-domain perplexity plus the MMLU subset; `sweep` runs a list of configurations in parallel:

```sh
uv run modal run modal_app.py::sweep --configurations '[{"domain_set": "extended", "ratio": 0.2}]'
```

Local tests run on a tiny random Qwen3.5-MoE model on a CUDA GPU in a few seconds:

```sh
uv run --group test pytest -q
```

## Project layout

| Path | Contents |
|---|---|
| `shear/pruning.py` | Layer-wise statistics, greedy selection, refit and healing |
| `shear/kernels.py` | Triton kernels for weighted SwiGLU and segmented Gram accumulation |
| `shear/packing.py` | Packed variable-width experts, save and load |
| `shear/calibration.py` | Calibration and evaluation window construction |
| `shear/evaluation.py` | Perplexity, lm-eval benchmarks and the MMLU subset |
| `modal_app.py` | Modal entry points |

## Related work

SHEAR builds on ideas from these papers:

- **UC-RAMP** ([2610.25717](https://arxiv.org/abs/2610.25717)) separates how wide each expert should be from which channels it keeps; SHEAR's adaptive widths follow this split.
- **CAMERA** ([2508.02322](https://arxiv.org/abs/2508.02322)) treats expert channels as micro-experts spanning the gate, up and down projections.
- **Optimal Brain Surgeon** and its structured successors motivate the greedy, error-compensated channel selection.
- **SlimMoE** ([2506.18349](https://arxiv.org/abs/2506.18349)) shows how far expert slimming goes when combined with large-scale distillation.
- **Router KD** ([2603.02217](https://arxiv.org/abs/2603.02217)) studies post-compression recovery in fine-grained MoE models.

## Limitations

- About 2–7% quality loss at 25% compression; reaching the original model's quality needs a training-based recovery step.
- The packed experts currently run through a per-expert reference loop. Faster inference needs a grouped GEMM kernel for variable widths.
- Results are measured on Qwen3.5-35B-A3B only.
