from math_verify import parse, verify


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
