import json
import os
import random

from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify

DIFFICULTY_ORDER = ["Level 3", "Level 4", "Level 2", "Level 5", "Level 1"]

# ANSI colors
green = '\x1b[38;2;0;255;0m'
cyan = '\x1b[38;2;0;255;255m'
gray = '\x1b[38;2;127;127;127m'
bold = '\033[1m'
endc = '\033[0m'


# JSONL I/O

def load_jsonl(filepath: str) -> list[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def load_completed(filepath: str) -> set[int]:
    """Load indices of already-evaluated problems from JSONL."""
    if not os.path.exists(filepath):
        return set()
    completed = set()
    with open(filepath) as f:
        for line in f:
            completed.add(json.loads(line)["idx"])
    return completed


def append_result(filepath: str, result: dict) -> None:
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")


# Dataset loading

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def load_math_problems() -> list[dict]:
    """Load MATH train split, ordered so medium-difficulty problems come first."""
    ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in MATH_SUBJECTS])
    problems = [{"idx": i, **row} for i, row in enumerate(ds)]
    rng = random.Random(42)
    groups: dict[str, list[dict]] = {level: [] for level in DIFFICULTY_ORDER}
    for p in problems:
        groups.setdefault(p["level"], []).append(p)
    ordered = []
    for level in DIFFICULTY_ORDER:
        rng.shuffle(groups[level])
        ordered.extend(groups[level])
    # Append any unknown levels (e.g. "Level ?") at the end
    for level, probs in groups.items():
        if level not in DIFFICULTY_ORDER:
            rng.shuffle(probs)
            ordered.extend(probs)
    return ordered


# Answer extraction & comparison

def _extract_last_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{} in text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def extract_gold_answer(solution: str) -> str:
    """Extract the answer from a MATH solution's \\boxed{}."""
    answer = _extract_last_boxed(solution)
    assert answer is not None, f"No \\boxed found in solution: {solution[:200]}"
    return answer


def extract_model_answer(response: str) -> str | None:
    """Extract \\boxed{} answer from model output, looking after </think> first."""
    think_end = response.rfind("</think>")
    search_text = response[think_end:] if think_end != -1 else response
    return _extract_last_boxed(search_text)


def check_answer(model_answer: str, gold_answer: str) -> bool:
    """Compare model answer to gold answer using math-verify."""
    gold_parsed = parse(f"${gold_answer}$")
    try:
        model_parsed = parse(f"${model_answer}$")
        return verify(gold_parsed, model_parsed)
    except Exception:
        return False


def score_problem(responses: list[str], gold_answer: str) -> dict:
    """Score all responses for a single problem. Returns accuracy stats + per-sample details."""
    per_sample = []
    num_correct = 0
    for response in responses:
        extracted = extract_model_answer(response)
        correct = check_answer(extracted, gold_answer) if extracted else False
        num_correct += correct
        per_sample.append({
            "extracted_answer": extracted,
            "correct": correct,
            "raw_response": response,
        })
    return {
        "num_correct": num_correct,
        "num_total": len(responses),
        "accuracy": num_correct / len(responses),
        "per_sample": per_sample,
    }
