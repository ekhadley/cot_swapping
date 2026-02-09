#!./.venv/bin/python
import argparse
import asyncio
import os

import aiohttp
from tqdm import tqdm

from api import call_openrouter
from tracker import Tracker
from utils import (
    load_env, load_aime_problems, load_completed, load_jsonl, append_result,
    write_jsonl, filter_divergent, extract_model_answer, check_answer,
    green, cyan, gray, bold, endc,
)

load_env()

MODEL_MAP = {
    "weak": "qwen/qwen3-8b",
    "strong": "qwen/qwen3-32b",
}

OTHER_LABEL = {"weak": "strong", "strong": "weak"}

PROMPT_TEMPLATE = "Solve the following math problem. Show your reasoning, then give your final answer in \\boxed{{}}.\n\n{problem}"

MIN_ACCURACY = 0.1
MAX_ACCURACY = 0.8
MAX_RETRY_WAVES = 5


async def process_sample(session, semaphore, tracker, row_id, model_id, prompt, gold_answer, temperature=0.7):
    """Process a single API request: call OpenRouter, extract answer, score, update tracker."""
    async with semaphore:
        tracker.mark_in_progress(row_id)
        try:
            result = await call_openrouter(session, model_id, prompt, temperature=temperature)
            extracted = extract_model_answer(result.full_text)
            if extracted is None:
                tracker.mark_complete(row_id, result.full_text, None, False, "no_answer")
                return
            correct = check_answer(extracted, gold_answer)
            status = "correct" if correct else "incorrect"
            tracker.mark_complete(row_id, result.full_text, extracted, correct, status)
        except Exception as e:
            tracker.mark_error(row_id, str(e))


async def run_pipeline(tracker, run_id, model_label, problems, n_samples, concurrency, temperature=0.7, pbar=None):
    """Run the full async pipeline, retrying until each problem has n_samples valid results."""
    model_id = MODEL_MAP[model_label]
    semaphore = asyncio.Semaphore(concurrency)

    # Register problems and cache prompts
    prompts = {}
    for prob in problems:
        tracker.register_problem(run_id, {**prob, "model_label": model_label})
        prompts[prob["idx"]] = PROMPT_TEMPLATE.format(problem=prob["problem"])

    next_sample_num = {prob["idx"]: 0 for prob in problems}
    gold_answers = {prob["idx"]: prob["gold_answer"] for prob in problems}

    async def run_one(session, row_id, prompt, gold_answer):
        await process_sample(session, semaphore, tracker, row_id, model_id, prompt, gold_answer, temperature=temperature)
        if pbar:
            pbar.update(1)

    async with aiohttp.ClientSession() as session:
        # Initial wave
        coros = []
        for prob in problems:
            for s in range(n_samples):
                row_id = tracker.enqueue(run_id, prob["idx"], model_label, s, prob["gold_answer"])
                coros.append(run_one(session, row_id, prompts[prob["idx"]], prob["gold_answer"]))
            next_sample_num[prob["idx"]] = n_samples
        await asyncio.gather(*coros)

        # Retry waves: replace no_answer and error samples
        for _ in range(MAX_RETRY_WAVES):
            retry_coros = []
            for prob in problems:
                samples = tracker.get_problem_samples(run_id, prob["idx"])
                valid = sum(1 for s in samples if s["status"] in ("correct", "incorrect"))
                deficit = n_samples - valid
                if deficit <= 0:
                    continue
                sn = next_sample_num[prob["idx"]]
                for _ in range(deficit):
                    row_id = tracker.enqueue(run_id, prob["idx"], model_label, sn, gold_answers[prob["idx"]])
                    retry_coros.append(run_one(session, row_id, prompts[prob["idx"]], gold_answers[prob["idx"]]))
                    sn += 1
                next_sample_num[prob["idx"]] = sn
            if not retry_coros:
                break
            if pbar:
                pbar.total += len(retry_coros)
                pbar.refresh()
            await asyncio.gather(*retry_coros)


def build_results_from_tracker(tracker, run_id, problems):
    """Build backward-compatible JSONL results from tracker data. Only counts valid (correct/incorrect) samples."""
    results = []
    for prob in problems:
        samples = tracker.get_problem_samples(run_id, prob["idx"])
        valid = [s for s in samples if s["status"] in ("correct", "incorrect")]
        if not valid:
            continue
        num_correct = sum(1 for s in valid if s["status"] == "correct")
        per_sample = [{
            "extracted_answer": s["extracted_answer"],
            "correct": bool(s["correct"]) if s["correct"] is not None else False,
            "raw_response": s["response_text"] or "",
        } for s in valid]
        results.append({
            "idx": prob["idx"],
            "problem": prob["problem"],
            "gold_answer": prob["gold_answer"],
            "level": prob.get("level", ""),
            "type": prob.get("type", ""),
            "num_correct": num_correct,
            "num_total": len(valid),
            "accuracy": num_correct / len(valid),
            "per_sample": per_sample,
        })
    return results


def evaluate_model(model_label, target_plausible=30, n_samples=16, concurrency=32, temperature=0.7):
    tracker = Tracker()
    problems = load_aime_problems()

    raw_path = f"data/raw_results_{model_label}.jsonl"
    plausible_path = f"data/plausible_{model_label}.jsonl"

    completed = load_completed(raw_path)
    plausible = load_jsonl(plausible_path)
    todo = [p for p in problems if p["idx"] not in completed]

    batch_size = max(1, concurrency // n_samples)
    pbar_prob = tqdm(total=len(todo), desc=f"{model_label} | pl={len(plausible)}", unit="prob", ncols=140)

    for i in range(0, len(todo), batch_size):
        if len(plausible) >= target_plausible:
            break

        batch = todo[i:i + batch_size]
        run_id = tracker.new_run()
        pbar_samp = tqdm(total=len(batch) * n_samples, desc=f"  batch {i // batch_size + 1}", unit="samp", leave=False, ncols=100)
        asyncio.run(run_pipeline(tracker, run_id, model_label, batch, n_samples, concurrency, temperature=temperature, pbar=pbar_samp))
        pbar_samp.close()

        for result in build_results_from_tracker(tracker, run_id, batch):
            append_result(raw_path, result)
            pbar_prob.update(1)
            pbar_prob.set_description(f"{model_label} | pl={len(plausible)} last={result['accuracy']:.2f}")

            if MIN_ACCURACY <= result["accuracy"] <= MAX_ACCURACY:
                plausible.append(result)
                append_result(plausible_path, result)
                pbar_prob.write(f"{green}[PLAUSIBLE]{endc} idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}")
            else:
                pbar_prob.write(f"{gray}[SKIP]{endc} idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}")

    pbar_prob.close()
    return plausible


def evaluate_from_saved(model_label, n_samples=16, n_hardest=None, concurrency=32, temperature=0.7):
    """Evaluate model on problems saved as plausible for the other model."""
    tracker = Tracker()
    run_id = tracker.new_run()
    other = OTHER_LABEL[model_label]

    saved_path = f"data/plausible_{other}.jsonl"
    problems = load_jsonl(saved_path)
    assert problems, f"No plausible problems found at {saved_path}"
    if n_hardest is not None:
        problems = sorted(problems, key=lambda p: p["accuracy"])[:n_hardest]

    out_path = f"data/cross_{model_label}_on_{other}.jsonl"
    completed_idxs = load_completed(out_path)
    todo = [p for p in problems if p["idx"] not in completed_idxs]
    existing_results = load_jsonl(out_path)

    pbar = tqdm(total=len(todo) * n_samples, desc=f"{model_label} on {other}", unit="samp", ncols=140)
    asyncio.run(run_pipeline(tracker, run_id, model_label, todo, n_samples, concurrency, temperature=temperature, pbar=pbar))
    pbar.close()

    for result in build_results_from_tracker(tracker, run_id, todo):
        existing_results.append(result)
        append_result(out_path, result)
        print(f"{cyan}idx={result['idx']} acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['level']}{endc}")

    avg_acc = sum(r["accuracy"] for r in existing_results) / len(existing_results) if existing_results else 0
    print(f"\n{bold}{green}Average accuracy: {avg_acc:.3f} over {len(existing_results)} problems{endc}")
    return existing_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation for CoT swapping")
    parser.add_argument("--model", choices=["weak", "strong"], required=True)
    parser.add_argument("--from-saved", action="store_true")
    parser.add_argument("--target", type=int, default=30, help="number of plausible problems to find")
    parser.add_argument("--n-samples", type=int, default=16, help="samples per problem")
    parser.add_argument("--n-hardest", type=int, default=None, help="with --from-saved, only use the N lowest-accuracy problems")
    parser.add_argument("--concurrency", type=int, default=32, help="max concurrent API requests")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (default 0.7)")
    parser.add_argument("--filter-divergent", type=str, default=None, metavar="CROSS_FILE", help="cross-eval JSONL to compute divergence against model's plausible set")
    parser.add_argument("--num", type=int, default=15, help="number of top divergent problems to save (with --filter-divergent)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.filter_divergent:
        other = OTHER_LABEL[args.model]
        plausible_path = f"data/plausible_{args.model}.jsonl"
        results = filter_divergent(plausible_path, args.filter_divergent, args.num, args.model, other)
        out_path = "data/target_problems.jsonl"
        write_jsonl(out_path, results)
        print(f"{bold}{green}Saved {len(results)} most divergent problems to {out_path}{endc}")
        for r in results:
            print(f"  idx={r['idx']:>5}  {args.model}={r[f'{args.model}_accuracy']:.3f}  {other}={r[f'{other}_accuracy']:.3f}  div={r['divergence']:.3f}  {r['type']}")
    elif args.from_saved:
        other = OTHER_LABEL[args.model]
        print(f"{bold}{cyan}=== Evaluating {args.model} ({MODEL_MAP[args.model]}) on {other}'s plausible problems ==={endc}")
        evaluate_from_saved(args.model, n_samples=args.n_samples, n_hardest=args.n_hardest, concurrency=args.concurrency, temperature=args.temperature)
    else:
        print(f"{bold}{cyan}=== Evaluating {args.model} model: {MODEL_MAP[args.model]} ==={endc}")
        plausible = evaluate_model(args.model, target_plausible=args.target, n_samples=args.n_samples, concurrency=args.concurrency, temperature=args.temperature)
        print(f"{green}Found {len(plausible)} plausible problems for {args.model} model{endc}")
