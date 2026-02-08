import asyncio
import json
import os
import random

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from utils import extract_gold_answer, score_problem

green = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
gray = '\x1b[38;2;127;127;127m'
endc = '\033[0m'

DIFFICULTY_ORDER = ["Level 3", "Level 4", "Level 2", "Level 5", "Level 1"]


def load_math_problems() -> list[dict]:
    """Load MATH train split, ordered so medium-difficulty problems come first."""
    ds = load_dataset("hendrycks/competition_math", split="train")
    problems = [{"idx": i, **row} for i, row in enumerate(ds)]
    rng = random.Random(42)
    groups = {level: [] for level in DIFFICULTY_ORDER}
    for p in problems:
        groups[p["level"]].append(p)
    ordered = []
    for level in DIFFICULTY_ORDER:
        rng.shuffle(groups[level])
        ordered.extend(groups[level])
    return ordered


def load_completed(filepath: str) -> set[int]:
    """Load indices of already-evaluated problems from JSONL."""
    if not os.path.exists(filepath):
        return set()
    completed = set()
    with open(filepath) as f:
        for line in f:
            completed.add(json.loads(line)["idx"])
    return completed


def load_jsonl(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def append_result(filepath: str, result: dict) -> None:
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")


def build_prompt(problem: str) -> list[dict]:
    return [{"role": "user", "content": f"Solve the following math problem. Show your reasoning, then give your final answer in \\boxed{{}}.\n\n{problem}"}]


async def generate_one(client: AsyncOpenAI, cfg, problem: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model=cfg.model_id,
            messages=build_prompt(problem),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return response.choices[0].message.content


async def generate_samples(client: AsyncOpenAI, cfg, problem: str, semaphore: asyncio.Semaphore) -> list[str]:
    tasks = [generate_one(client, cfg, problem, semaphore) for _ in range(cfg.samples_per_problem)]
    return await asyncio.gather(*tasks)


async def evaluate_model(cfg) -> list[dict]:
    client = AsyncOpenAI(
        base_url=cfg.openrouter_base_url,
        api_key=cfg.openrouter_api_key,
        max_retries=5,
    )
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    problems = load_math_problems()
    raw_path = os.path.join(cfg.data_dir, f"raw_results_{cfg.model_label}.jsonl")
    plausible_path = os.path.join(cfg.data_dir, f"plausible_{cfg.model_label}.jsonl")

    completed = load_completed(raw_path)
    plausible = load_jsonl(plausible_path)

    pbar = tqdm(total=cfg.target_plausible, desc=f"{cfg.model_label} plausible", unit="prob")
    pbar.update(len(plausible))

    for prob in problems:
        if len(plausible) >= cfg.target_plausible:
            break
        if prob["idx"] in completed:
            continue

        gold = extract_gold_answer(prob["solution"])
        responses = await generate_samples(client, cfg, prob["problem"], semaphore)
        result = score_problem(responses, gold)
        result["idx"] = prob["idx"]
        result["problem"] = prob["problem"]
        result["gold_answer"] = gold
        result["level"] = prob["level"]
        result["type"] = prob["type"]

        append_result(raw_path, result)

        if cfg.accuracy_low <= result["accuracy"] <= cfg.accuracy_high:
            plausible.append(result)
            append_result(plausible_path, result)
            pbar.update(1)
            print(f"{green}[PLAUSIBLE]{endc} idx={prob['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {prob['level']}")
        else:
            print(f"{gray}[SKIP]{endc} idx={prob['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {prob['level']}")

    pbar.close()
    return plausible
