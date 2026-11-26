import torch
from datasets import load_dataset

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
PAPER_DOMAINS = {
    "general": [("Salesforce/wikitext", ("wikitext-2-raw-v1",), "train", "test", "{text}", 128)],
    "reasoning": [
        ("openai/gsm8k", ("main",), "train", "test", "Question: {question}\nAnswer: {answer}", 64),
        ("MU-NLPC/Calc-math_qa", ("original-splits",), "train", "test", "Question: {question}\nRationale: {rationale}", 64),
    ],
    "mathematics": [("EleutherAI/hendrycks_math", MATH_SUBJECTS, "train", "test", "Problem: {problem}\nSolution: {solution}", 128)],
    "code": [("google-research-datasets/mbpp", ("full",), "train", "test", "{text}\n{code}", 64)],
}
DOMAIN_SETS = {
    "paper": PAPER_DOMAINS,
    "broad": PAPER_DOMAINS
    | {
        "science": [
            ("allenai/sciq", ("default",), "train", "test", "{support}\nQuestion: {question}\nAnswer: {correct_answer}", 64)
        ],
        "dialogue": [("HuggingFaceH4/ultrachat_200k", ("default",), "train_sft", "test_sft", None, 64)],
        "multilingual": [("CohereLabs/aya_dataset", ("default",), "train", "test", "{inputs}\n{targets}", 64)],
    },
}
DOMAIN_SETS["extended"] = DOMAIN_SETS["broad"] | {
    "multilingual": [("CohereLabs/aya_dataset", ("default",), "train", "test", "{inputs}\n{targets}", 128)],
    "knowledge": [
        (
            "cais/mmlu",
            ("auxiliary_train",),
            "train",
            "train",
            "{train[question]}\nA. {train[choices][0]}\nB. {train[choices][1]}\nC. {train[choices][2]}\nD. {train[choices][3]}\nAnswer:",
            128,
        )
    ],
}


def source_windows(tokenizer, source, split, count, length, seed):
    path, configs, train_split, eval_split, template, _ = source
    ids = []
    for config in configs:
        budget = len(ids) + count * length * 8 // len(configs)
        for row in load_dataset(path, config, split=train_split if split == "train" else eval_split):
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False) if template is None else template.format(**row)
            ids += tokenizer(text + "\n\n").input_ids
            if len(ids) >= budget:
                break
    windows = torch.tensor(ids[: len(ids) // length * length]).view(-1, length)
    windows = windows[torch.randperm(len(windows), generator=torch.Generator().manual_seed(seed))]
    return windows.repeat(-(-count // len(windows)), 1)[:count]


def build_windows(tokenizer, domains, split, scale, seed, length=2048):
    return {
        name: torch.cat(
            [source_windows(tokenizer, source, split, max(1, round(source[5] * scale)), length, seed) for source in sources]
        )
        for name, sources in domains.items()
    }
