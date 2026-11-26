import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict

SEED = 1234
TASK_LIMITS = {
    "arc_challenge": 500,
    "arc_easy": 500,
    "piqa": 500,
    "hellaswag": 500,
    "winogrande": 500,
    "mmlu": 20,
    "gsm8k": 250,
}
PRIMARY_METRICS = ("acc_norm,none", "acc,none", "exact_match,flexible-extract")


def prefetch_tasks(task_limits):
    get_task_dict(list(task_limits), TaskManager())


def build_choice_benchmark(tokenizer, per_subject):
    from datasets import get_dataset_config_names, load_dataset

    prompts, answers = [], []
    for subject in sorted(set(get_dataset_config_names("cais/mmlu")) - {"all", "auxiliary_train"}):
        for row in load_dataset("cais/mmlu", subject, split="test").select(range(per_subject)):
            options = "".join(f"\n{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
            header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            prompts.append(torch.tensor(tokenizer(f"{header}{row['question'].strip()}{options}\nAnswer:").input_ids))
            answers.append(row["answer"])
    choices = torch.tensor([tokenizer(f" {letter}").input_ids[-1] for letter in "ABCD"])
    return {"prompts": prompts, "answers": torch.tensor(answers), "choices": choices}


@torch.inference_mode()
def choice_accuracy(model, benchmark, batch_size=32):
    predictions = []
    for start in range(0, len(benchmark["prompts"]), batch_size):
        prompts = benchmark["prompts"][start : start + batch_size]
        padded = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True).to(model.device)
        hidden = model.model(input_ids=padded, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(prompts)), torch.tensor([len(p) - 1 for p in prompts])]
        predictions.append(model.lm_head(last)[:, benchmark["choices"].to(model.device)].argmax(-1).cpu())
    return (torch.cat(predictions) == benchmark["answers"]).float().mean().item()


@torch.inference_mode()
def proxy_scores(model, windows, benchmark):
    scores = {f"perplexity/{name}": perplexity(model, tensor) for name, tensor in windows.items()}
    scores["accuracy/mmlu_proxy"] = choice_accuracy(model, benchmark)
    return scores


def perplexity(model, windows, batch_size=2):
    losses = [model(input_ids=batch, labels=batch, use_cache=False).loss for batch in windows.to(model.device).split(batch_size)]
    return torch.stack(losses).mean().exp().item()


@torch.inference_mode()
def evaluate(model, tokenizer, windows, task_limits, batch_size=32):
    scores = {f"perplexity/{name}": perplexity(model, tensor) for name, tensor in windows.items()}
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size, max_length=4096)
    for task, limit in task_limits.items():
        results = simple_evaluate(
            model=lm,
            tasks=[task],
            limit=limit,
            bootstrap_iters=0,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
        )["results"][task]
        scores[f"accuracy/{task}"] = next(results[metric] for metric in PRIMARY_METRICS if metric in results)
    scores["accuracy/mean"] = sum(value for key, value in scores.items() if key.startswith("accuracy/")) / len(task_limits)
    return scores
