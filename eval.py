import argparse
import os
from dataclasses import dataclass

import torch as t
from transformer_lens import HookedTransformer
from tqdm import tqdm

from utils import (
    load_math_problems, load_completed, load_jsonl, append_result,
    extract_gold_answer, score_problem,
    green, cyan, gray, bold, endc,
)

WEAK_MODEL = "qwen3-1.7b"
STRONG_MODEL = "qwen3-14b"

MODEL_MAP = {
    "weak": WEAK_MODEL,
    "strong": STRONG_MODEL,
}

PROMPT_TEMPLATE = "Solve the following math problem. Show your reasoning, then give your final answer in \\boxed{{}}.\n\n{problem}"


@dataclass
class EvalConfig:
    model_name: str
    model_label: str
    samples_per_problem: int = 16
    target_plausible: int = 30
    accuracy_low: float = 0.25
    accuracy_high: float = 0.75
    temperature: float = 0.7
    max_new_tokens: int = 16384
    data_dir: str = "data"


def load_model(model_name: str) -> HookedTransformer:
    print(f"{gray}Loading {model_name}...{endc}")
    model = HookedTransformer.from_pretrained(model_name, dtype=t.bfloat16, device="cuda")
    print(f"{green}Loaded {model_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params){endc}")
    return model


def format_prompt(model: HookedTransformer, problem: str) -> str:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}]
    return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_samples(model: HookedTransformer, cfg: EvalConfig, prompt_str: str) -> list[str]:
    responses = []
    for _ in range(cfg.samples_per_problem):
        output = model.generate(
            prompt_str,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            verbose=True,
        )
        # model.generate returns the full sequence including the prompt; strip it
        response = output[len(prompt_str):]
        responses.append(response)
    return responses


def evaluate_model(cfg: EvalConfig) -> list[dict]:
    model = load_model(cfg.model_name)

    problems = load_math_problems()
    raw_path = os.path.join(cfg.data_dir, f"raw_results_{cfg.model_label}.jsonl")
    plausible_path = os.path.join(cfg.data_dir, f"plausible_{cfg.model_label}.jsonl")

    completed = load_completed(raw_path)
    plausible = load_jsonl(plausible_path)

    n_attempted = len(completed)
    pbar = tqdm(total=cfg.target_plausible, desc=f"{cfg.model_label}", unit="pl", ncols=120)
    pbar.set_postfix(tried=n_attempted, last_acc="—")
    pbar.update(len(plausible))

    for prob in problems:
        if len(plausible) >= cfg.target_plausible:
            break
        if prob["idx"] in completed:
            continue

        gold = extract_gold_answer(prob["solution"])
        prompt_str = format_prompt(model, prob["problem"])
        responses = generate_samples(model, cfg, prompt_str)
        result = score_problem(responses, gold)
        result["idx"] = prob["idx"]
        result["problem"] = prob["problem"]
        result["gold_answer"] = gold
        result["level"] = prob["level"]
        result["type"] = prob["type"]

        append_result(raw_path, result)
        n_attempted += 1
        pbar.set_postfix(tried=n_attempted, last_acc=f"{result['accuracy']:.2f}")

        if cfg.accuracy_low <= result["accuracy"] <= cfg.accuracy_high:
            plausible.append(result)
            append_result(plausible_path, result)
            pbar.update(1)
            print(f"{green}[PLAUSIBLE]{endc} idx={prob['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {prob['level']}")
        else:
            print(f"{gray}[SKIP]{endc} idx={prob['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {prob['level']}")

    pbar.close()
    return plausible


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation for CoT swapping")
    parser.add_argument("--model", choices=["weak", "strong"], required=True)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    cfg = EvalConfig(model_name=MODEL_MAP[args.model], model_label=args.model)
    print(f"{bold}{cyan}=== Evaluating {args.model} model: {cfg.model_name} ==={endc}")
    plausible = evaluate_model(cfg)
    print(f"{green}Found {len(plausible)} plausible problems for {args.model} model{endc}")
