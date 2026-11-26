import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
CACHE = Path("/cache")
SEED = 1234
SMOKE_TASK_LIMITS = {"arc_easy": 20, "gsm8k": 4}

app = modal.App("shear")
volume = modal.Volume.from_name("shear-cache", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "lm-eval[hf]",
        "flash-linear-attention",
        "typing-extensions",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)
cpu_image = base_image.add_local_python_source("shear")
gpu_image = base_image.env({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}).add_local_python_source("shear")


@app.function(image=cpu_image, volumes={"/cache": volume}, cpu=8, memory=32768, timeout=7200)
def prefetch_assets():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from shear.calibration import DOMAIN_SETS, build_windows
    from shear.evaluation import TASK_LIMITS, build_choice_benchmark, prefetch_tasks

    snapshot_download(MODEL_ID)
    volume.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    data = CACHE / "data"
    data.mkdir(exist_ok=True)
    for name, domains in DOMAIN_SETS.items():
        if not (data / f"calibration_{name}.pt").exists():
            windows = build_windows(tokenizer, domains, "train", 1.0, SEED)
            torch.save(windows, data / f"calibration_{name}.pt")
            print(name, {domain: tuple(tensor.shape) for domain, tensor in windows.items()})
    if not (data / "evaluation.pt").exists():
        torch.save(build_windows(tokenizer, DOMAIN_SETS["broad"], "eval", 0.125, SEED), data / "evaluation.pt")
    if not (data / "choice_benchmark.pt").exists():
        torch.save(build_choice_benchmark(tokenizer, 30), data / "choice_benchmark.pt")
    prefetch_tasks(TASK_LIMITS)
    volume.commit()


def execute(name, calibration, evaluation, task_limits, settings, export):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from shear.evaluation import evaluate
    from shear.packing import pack_experts, save_packed
    from shear.pruning import prune
    started = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm"
    )
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"model loaded in {time.time() - started:.0f}s")

    versions = {package: version(package) for package in ("torch", "transformers", "lm_eval", "datasets")}
    identity = json.dumps(
        {"tasks": task_limits, "windows": {k: len(v) for k, v in evaluation.items()}, "versions": versions}, sort_keys=True
    )
    baseline_path = CACHE / "results" / f"baseline_{hashlib.sha256(identity.encode()).hexdigest()[:12]}.json"
    if not baseline_path.exists():
        started = time.time()
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(evaluate(model, tokenizer, evaluation, task_limits), indent=2))
        volume.commit()
        print(f"baseline evaluated in {time.time() - started:.0f}s")
    baseline = json.loads(baseline_path.read_text())

    started = time.time()
    masks, summary = prune(model, list(calibration.values()), **settings)
    summary["seconds"] = time.time() - started
    print(f"pruned in {summary['seconds']:.0f}s")
    started = time.time()
    pruned = evaluate(model, tokenizer, evaluation, task_limits)
    print(f"pruned model evaluated in {time.time() - started:.0f}s")

    run = CACHE / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    torch.save(masks, run / "masks.pt")
    sample = next(iter(evaluation.values()))[:1].to(model.device)
    with torch.inference_mode():
        masked_logits = model(input_ids=sample, use_cache=False).logits.float()
        pack_experts(model, masks)
        torch.cuda.empty_cache()
        packed_logits = model(input_ids=sample, use_cache=False).logits.float()
    summary["packed_logit_max_difference"] = (packed_logits - masked_logits).abs().max().item()
    summary["packed_top1_agreement"] = (packed_logits.argmax(-1) == masked_logits.argmax(-1)).float().mean().item()
    summary["packed_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    print(f"packed model: {summary['packed_parameters'] / 1e9:.2f}B parameters, top-1 agreement {summary['packed_top1_agreement']:.4f}")
    if export:
        started = time.time()
        save_packed(model, tokenizer, run / "model")
        print(f"packed model saved to {run / 'model'} in {time.time() - started:.0f}s")
    report = {"settings": settings, "domains": list(calibration), "pruning": summary, "baseline": baseline, "pruned": pruned}
    (run / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()

    print(f"{'metric':32s} {'baseline':>10s} {'pruned':>10s} {'change':>9s}")
    for metric, value in pruned.items():
        print(f"{metric:32s} {baseline[metric]:10.4f} {value:10.4f} {100 * (value / baseline[metric] - 1):+8.2f}%")
    print(f"removed {summary['routed_parameters_removed'] / 1e9:.2f}B of {summary['routed_parameters_total'] / 1e9:.2f}B routed parameters")
    print(f"report saved to {run / 'report.json'}")


@app.function(image=gpu_image, gpu="H200", volumes={"/cache": volume}, memory=65536, timeout=3600)
def prune_model(
    ratio: float = 0.25,
    domain_set: str = "extended",
    damping: float = 0.01,
    dense_target: float = 1.0,
    batch_size: int = 64,
    heal_steps: int = 64,
    heal_learning_rate: float = 2e-5,
    minimum_width: int = 128,
    domain_weights: str = "0.4,0.06,0.06,0.06,0.06,0.06,0.15,0.15",
    export: bool = True,
):
    import torch

    from shear.evaluation import TASK_LIMITS

    settings = dict(
        ratio=ratio,
        damping=damping,
        dense_target=dense_target,
        batch_size=batch_size,
        heal_steps=heal_steps,
        heal_learning_rate=heal_learning_rate,
        minimum_width=minimum_width,
        domain_weights=[float(weight) for weight in domain_weights.split(",")],
    )
    execute(
        f"{domain_set}_ratio{ratio}_width{minimum_width}_heal{heal_steps}",
        torch.load(CACHE / "data" / f"calibration_{domain_set}.pt"),
        torch.load(CACHE / "data" / "evaluation.pt"),
        TASK_LIMITS,
        settings,
        export,
    )


@app.function(image=gpu_image, gpu="H200", volumes={"/cache": volume}, memory=65536, timeout=600)
def smoke_test():
    import torch

    calibration = torch.load(CACHE / "data" / "calibration_paper.pt")
    evaluation = torch.load(CACHE / "data" / "evaluation.pt")
    execute(
        "smoke",
        {domain: tensor[:2] for domain, tensor in calibration.items()},
        {domain: tensor[:1] for domain, tensor in evaluation.items()},
        SMOKE_TASK_LIMITS,
        dict(ratio=0.25, batch_size=64),
        False,
    )


@app.function(image=gpu_image, gpu="H200", volumes={"/cache": volume}, memory=65536, timeout=1200)
def explore(settings: dict):
    import torch
    from transformers import AutoModelForImageTextToText

    from shear.evaluation import proxy_scores
    from shear.pruning import prune

    settings = dict(settings)
    domain_set = settings.pop("domain_set", "broad")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", experts_implementation="grouped_mm"
    )
    model.eval().requires_grad_(False)
    evaluation = torch.load(CACHE / "data" / "evaluation.pt")
    benchmark = torch.load(CACHE / "data" / "choice_benchmark.pt")
    versions = {package: version(package) for package in ("torch", "transformers")}
    baseline_path = CACHE / "results" / f"proxy_baseline_{hashlib.sha256(json.dumps(versions).encode()).hexdigest()[:12]}.json"
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(proxy_scores(model, evaluation, benchmark), indent=2))
        volume.commit()
    baseline = json.loads(baseline_path.read_text())
    started = time.time()
    masks, summary = prune(model, list(torch.load(CACHE / "data" / f"calibration_{domain_set}.pt").values()), **settings)
    pruned = proxy_scores(model, evaluation, benchmark)
    changes = {metric: 100 * (value / baseline[metric] - 1) for metric, value in pruned.items()}
    result = {"domain_set": domain_set, **settings, "seconds": time.time() - started, "changes": changes}
    widths = masks.sum(2).float()
    print("relative error by layer", [round(value, 4) for value in summary["relative_error_by_layer"]])
    print("expert widths: min", int(widths.min()), "zero-width experts", int((widths == 0).sum()), "quantiles", widths.quantile(torch.tensor([0.05, 0.5, 0.95])).tolist())
    name = hashlib.sha256(json.dumps(result | {"seconds": 0}, sort_keys=True).encode()).hexdigest()[:12]
    (CACHE / "experiments").mkdir(exist_ok=True)
    (CACHE / "experiments" / f"{name}.json").write_text(json.dumps(result | {"summary": summary, "baseline": baseline, "pruned": pruned}, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def sweep(configurations: str):
    for result in explore.map(json.loads(configurations)):
        changes = result.pop("changes")
        print(json.dumps(result))
        print("   " + "  ".join(f"{metric.split('/')[-1]}={value:+.2f}%" for metric, value in changes.items()))
