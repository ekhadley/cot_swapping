import random

from datasets import load_dataset
from math_verify import parse, verify

# ANSI colors
green = '\x1b[38;2;0;255;0m'
cyan = '\x1b[38;2;0;255;255m'
gray = '\x1b[38;2;127;127;127m'
bold = '\033[1m'
endc = '\033[0m'


def load_aime_problems() -> list[dict]:
    ds = load_dataset("qq8933/AIME_1983_2024", split="train")
    problems = [
        {"idx": i, "problem": row["Question"], "answer": str(row["Answer"]),
         "year": row["Year"], "number": row["Problem Number"]}
        for i, row in enumerate(ds)
    ]
    random.Random(42).shuffle(problems)
    return problems


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    after = idx + len("\\boxed")
    if after >= len(text) or text[after] != '{':
        return None
    start, depth = after + 1, 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            if depth == 0: return text[start:i]
            depth -= 1
    return None


def extract_answer(response: str) -> str | None:
    think_end = response.rfind("</think>")
    search = response[think_end:] if think_end != -1 else response
    return extract_boxed(search)


def check_answer(model_answer: str, gold_answer: str) -> bool:
    return verify(parse(f"${gold_answer}$"), parse(f"${model_answer}$"))
