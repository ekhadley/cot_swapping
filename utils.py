import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify

# ANSI colors
green = '\x1b[38;2;0;255;0m'
cyan = '\x1b[38;2;0;255;255m'
gray = '\x1b[38;2;127;127;127m'
bold = '\033[1m'
endc = '\033[0m'


def load_env():
    """Load .env file into os.environ."""
    env_path = Path(__file__).parent / ".env"
    assert env_path.exists(), f"Missing .env file at {env_path}"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


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


def load_aime_problems() -> list[dict]:
    """Load AIME 1983-2024 problems, shuffled with fixed seed."""
    ds = load_dataset("qq8933/AIME_1983_2024", split="train")
    problems = [
        {
            "idx": i,
            "problem": row["Question"],
            "gold_answer": str(row["Answer"]),
            "level": "AIME",
            "type": f"AIME_{row['Year']}",
            "year": row["Year"],
            "problem_number": row["Problem Number"],
        }
        for i, row in enumerate(ds)
    ]
    random.Random(42).shuffle(problems)
    return problems


# Answer extraction & comparison

def _extract_last_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{} or \\boxed X in text."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    after = idx + len("\\boxed")
    if after >= len(text):
        return None
    if text[after] == '{':
        start = after + 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i]
                depth -= 1
        return None
    # \boxed X (no braces, single LaTeX token)
    start = after
    while start < len(text) and text[start] == ' ':
        start += 1
    if start >= len(text):
        return None
    end = start + 1
    while end < len(text) and text[end] not in ' \t\n.,;)$':
        end += 1
    return text[start:end]


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


def write_jsonl(filepath: str, data: list[dict]) -> None:
    with open(filepath, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def filter_divergent(plausible_path: str, cross_path: str, n: int, base_label: str, cross_label: str) -> list[dict]:
    """Find N problems with largest accuracy divergence between base and cross evaluations."""
    base = {d["idx"]: d for d in load_jsonl(plausible_path)}
    cross = {d["idx"]: d for d in load_jsonl(cross_path)}
    common = set(base) & set(cross)

    merged = []
    for idx in common:
        b, c = base[idx], cross[idx]
        merged.append({
            "idx": idx,
            "problem": b["problem"],
            "gold_answer": b["gold_answer"],
            "level": b["level"],
            "type": b["type"],
            "divergence": abs(b["accuracy"] - c["accuracy"]),
            f"{base_label}_accuracy": b["accuracy"],
            f"{base_label}_num_correct": b["num_correct"],
            f"{base_label}_num_total": b["num_total"],
            f"{base_label}_per_sample": b["per_sample"],
            f"{cross_label}_accuracy": c["accuracy"],
            f"{cross_label}_num_correct": c["num_correct"],
            f"{cross_label}_num_total": c["num_total"],
            f"{cross_label}_per_sample": c["per_sample"],
        })

    merged.sort(key=lambda x: -x["divergence"])
    return merged[:n]


def score_problem(responses: list[str], gold_answer: str, timestamps: list[int] | None = None) -> dict:
    """Score all responses for a single problem. Returns accuracy stats + per-sample details."""
    per_sample = []
    num_correct = 0
    for i, response in enumerate(responses):
        extracted = extract_model_answer(response)
        correct = check_answer(extracted, gold_answer) if extracted else False
        num_correct += correct
        sample = {
            "extracted_answer": extracted,
            "correct": correct,
            "raw_response": response,
        }
        if timestamps is not None:
            sample["created"] = timestamps[i]
        per_sample.append(sample)
    return {
        "num_correct": num_correct,
        "num_total": len(responses),
        "accuracy": num_correct / len(responses),
        "per_sample": per_sample,
    }
