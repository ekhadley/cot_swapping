#!./.venv/bin/python
import argparse
import asyncio
import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from tqdm import tqdm

from utils import (
    load_env, load_aime_problems, load_completed, load_jsonl, append_result,
    write_jsonl, filter_divergent,
    build_prompt, extract_model_answer, score_problem,
    green, cyan, gray, bold, endc,
)

load_env()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

WEAK_MODEL = "qwen/qwen3-8b"
STRONG_MODEL = "qwen/qwen3-32b"

MODEL_MAP = {
    "weak": WEAK_MODEL,
    "strong": STRONG_MODEL,
}

OTHER_LABEL = {"weak": "strong", "strong": "weak"}

@dataclass
class EvalConfig:
    model_id: str
    model_label: str
    base_url: str = OPENROUTER_BASE_URL
    api_key: str = OPENROUTER_API_KEY
    samples_per_problem: int = 16
    target_plausible: int = 30
    accuracy_low: float = 0.25
    accuracy_high: float = 0.75
    temperature: float = 0.7
    max_tokens: int = 16384
    max_concurrent: int = 256
    data_dir: str = "data"


MAX_SAMPLE_RETRIES = 3


async def sample_once(client, semaphore, cfg: EvalConfig, problem_text: str) -> str:
    """Get one completion, retrying if response is empty or missing \\boxed{}."""
    content = ""
    for _ in range(MAX_SAMPLE_RETRIES):
        async with semaphore:
            r = await client.chat.completions.create(
                model=cfg.model_id, messages=build_prompt(problem_text),
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            )
            content = r.choices[0].message.content or ""
        if content and extract_model_answer(content) is not None:
            return content
    return content


async def evaluate_model(cfg: EvalConfig) -> list[dict]:
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, max_retries=5)
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    problems = load_aime_problems()
    raw_path = os.path.join(cfg.data_dir, f"raw_results_{cfg.model_label}.jsonl")
    plausible_path = os.path.join(cfg.data_dir, f"plausible_{cfg.model_label}.jsonl")

    completed = load_completed(raw_path)
    plausible = load_jsonl(plausible_path)
    todo = [p for p in problems if p["idx"] not in completed]

    async def process_problem(prob: dict) -> dict:
        responses = await asyncio.gather(*[sample_once(client, semaphore, cfg, prob["problem"]) for _ in range(cfg.samples_per_problem)])
        result = score_problem(responses, prob["gold_answer"])
        result.update(idx=prob["idx"], problem=prob["problem"], gold_answer=prob["gold_answer"], level=prob["level"], type=prob["type"])
        return result

    last_acc = "—"

    def desc():
        return f"{cfg.model_label} | pl={len(plausible)} last={last_acc}"

    pbar = tqdm(desc=desc(), unit="prob", ncols=180)

    active = {asyncio.create_task(process_problem(p)): p for p in todo}

    while active:
        done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            prob = active.pop(task)
            if task.exception():
                pbar.write(f"{gray}[ERROR] idx={prob['idx']}: {task.exception()}{endc}")
                pbar.update(1)
                continue
            result = task.result()

            append_result(raw_path, result)
            last_acc = f"{result['accuracy']:.2f}"
            pbar.set_description(desc())
            pbar.update(1)

            if cfg.accuracy_low <= result["accuracy"] <= cfg.accuracy_high:
                plausible.append(result)
                append_result(plausible_path, result)
                pbar.set_description(desc())
                pbar.write(f"{green}[PLAUSIBLE]{endc} idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}")
            else:
                pbar.write(f"{gray}[SKIP]{endc} idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}")

            if len(plausible) >= cfg.target_plausible:
                for remaining in active:
                    remaining.cancel()
                active.clear()
                break

    pbar.close()
    return plausible


async def evaluate_from_saved(cfg: EvalConfig, n_hardest: int | None = None) -> list[dict]:
    """Evaluate model on problems saved as plausible for the other model."""
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, max_retries=5)
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    other = OTHER_LABEL[cfg.model_label]
    saved_path = os.path.join(cfg.data_dir, f"plausible_{other}.jsonl")
    problems = load_jsonl(saved_path)
    assert problems, f"No plausible problems found at {saved_path}"
    if n_hardest is not None:
        problems = sorted(problems, key=lambda p: p["accuracy"])[:n_hardest]

    out_path = os.path.join(cfg.data_dir, f"cross_{cfg.model_label}_on_{other}.jsonl")
    completed = load_completed(out_path)
    todo = [p for p in problems if p["idx"] not in completed]

    async def process_problem(prob: dict) -> dict:
        responses = await asyncio.gather(*[sample_once(client, semaphore, cfg, prob["problem"]) for _ in range(cfg.samples_per_problem)])
        result = score_problem(responses, prob["gold_answer"])
        result.update(idx=prob["idx"], problem=prob["problem"], gold_answer=prob["gold_answer"], level=prob["level"], type=prob["type"])
        return result

    last_acc = "—"

    def desc():
        return f"{cfg.model_label} on {other} | last={last_acc}"

    pbar = tqdm(total=len(problems), initial=len(problems) - len(todo), desc=desc(), unit="prob", ncols=180)

    active = {asyncio.create_task(process_problem(p)): p for p in todo}
    results = load_jsonl(out_path)

    while active:
        done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            prob = active.pop(task)
            if task.exception():
                pbar.write(f"{gray}[ERROR] idx={prob['idx']}: {task.exception()}{endc}")
                pbar.update(1)
                continue
            result = task.result()
            results.append(result)
            append_result(out_path, result)
            last_acc = f"{result['accuracy']:.2f}"
            pbar.set_description(desc())
            pbar.update(1)
            pbar.write(f"{cyan}idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}{endc}")

    pbar.close()
    avg_acc = sum(r["accuracy"] for r in results) / len(results)
    print(f"\n{bold}{green}Average accuracy: {avg_acc:.3f} over {len(results)} problems{endc}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation for CoT swapping")
    parser.add_argument("--model", choices=["weak", "strong"], required=True)
    parser.add_argument("--from-saved", action="store_true")
    parser.add_argument("--target", type=int, default=30, help="number of plausible problems to find")
    parser.add_argument("--concurrent", type=int, default=256, help="max concurrent API requests")
    parser.add_argument("--n-hardest", type=int, default=None, help="with --from-saved, only use the N lowest-accuracy problems")
    parser.add_argument("--filter-divergent", type=str, default=None, metavar="CROSS_FILE", help="cross-eval JSONL to compute divergence against model's plausible set")
    parser.add_argument("--num", type=int, default=15, help="number of top divergent problems to save (with --filter-divergent)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    cfg = EvalConfig(model_id=MODEL_MAP[args.model], model_label=args.model, target_plausible=args.target, max_concurrent=args.concurrent)

    if args.filter_divergent:
        other = OTHER_LABEL[args.model]
        plausible_path = os.path.join("data", f"plausible_{args.model}.jsonl")
        results = filter_divergent(plausible_path, args.filter_divergent, args.num, args.model, other)
        out_path = os.path.join("data", "target_problems.jsonl")
        write_jsonl(out_path, results)
        print(f"{bold}{green}Saved {len(results)} most divergent problems to {out_path}{endc}")
        for r in results:
            print(f"  idx={r['idx']:>5}  {args.model}={r[f'{args.model}_accuracy']:.3f}  {other}={r[f'{other}_accuracy']:.3f}  div={r['divergence']:.3f}  {r['type']}")
    elif args.from_saved:
        other = OTHER_LABEL[args.model]
        print(f"{bold}{cyan}=== Evaluating {args.model} ({cfg.model_id}) on {other}'s plausible problems ==={endc}")
        asyncio.run(evaluate_from_saved(cfg, n_hardest=args.n_hardest))
    else:
        print(f"{bold}{cyan}=== Evaluating {args.model} model: {cfg.model_id} ==={endc}")
        plausible = asyncio.run(evaluate_model(cfg))
        print(f"{green}Found {len(plausible)} plausible problems for {args.model} model{endc}")
