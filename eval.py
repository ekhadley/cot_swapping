#!/.venv/bin/python
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    load_aime_problems, load_completed, load_jsonl, append_result,
    write_jsonl, filter_divergent, score_problem,
    green, cyan, gray, bold, endc,
)

MODEL_MAP = {
    "weak": "Qwen/Qwen3-1.7B",
    "strong": "Qwen/Qwen3-14B",
}

OTHER_LABEL = {"weak": "strong", "strong": "weak"}

PROMPT_TEMPLATE = "Solve the following math problem. Show your reasoning, then give your final answer in \\boxed{{}}.\n\n{problem}"

MIN_ACCURACY = 0.1
MAX_ACCURACY = 0.8
MAX_RETRY_WAVES = 5


def load_model(model_id):
    print(f"{bold}Loading {model_id}...{endc}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.eval()
    print(f"{green}Model loaded on {model.device}{endc}")
    return model, tokenizer


def generate_samples(model, tokenizer, prompt, n_samples, max_tokens, batch_size, temperature):
    """Generate n_samples completions, retrying truncated ones. Returns list of response strings."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    input_len = input_ids.shape[1]

    all_responses = []
    remaining = n_samples

    for wave in range(MAX_RETRY_WAVES + 1):
        if remaining <= 0:
            break

        for batch_start in range(0, remaining, batch_size):
            bs = min(batch_size, remaining - batch_start)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids.expand(bs, -1),
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=1,  # one per input copy
                )
            for seq in outputs:
                gen_len = seq.shape[0] - input_len
                if gen_len >= max_tokens:
                    continue  # truncated, discard
                response = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                all_responses.append(response)

        remaining = n_samples - len(all_responses)
        if remaining <= 0:
            break
        if wave < MAX_RETRY_WAVES:
            print(f"    retrying {remaining} truncated samples (wave {wave + 1})...")

    return all_responses[:n_samples]


def evaluate_model(model, tokenizer, model_label, target_plausible=30, n_samples=16, max_tokens=16384, batch_size=None, temperature=0.7):
    if batch_size is None:
        batch_size = n_samples

    problems = load_aime_problems()
    raw_path = f"data/raw_results_{model_label}.jsonl"
    plausible_path = f"data/plausible_{model_label}.jsonl"

    completed = load_completed(raw_path)
    plausible = load_jsonl(plausible_path)
    todo = [p for p in problems if p["idx"] not in completed]

    for i, prob in enumerate(todo):
        if len(plausible) >= target_plausible:
            break

        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        print(f"[{i+1}/{len(todo)}] idx={prob['idx']} sampling...", end=" ", flush=True)

        responses = generate_samples(model, tokenizer, prompt, n_samples, max_tokens, batch_size, temperature)
        print(f"{len(responses)}/{n_samples} valid")

        if not responses:
            print(f"  {gray}-> no valid samples, skipping{endc}")
            continue

        scored = score_problem(responses, prob["gold_answer"])
        result = {
            "idx": prob["idx"],
            "problem": prob["problem"],
            "gold_answer": prob["gold_answer"],
            "level": prob.get("level", ""),
            "type": prob.get("type", ""),
            **scored,
        }

        append_result(raw_path, result)

        if MIN_ACCURACY <= result["accuracy"] <= MAX_ACCURACY:
            plausible.append(result)
            append_result(plausible_path, result)
            print(f"  -> acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['type']}  {green}[PLAUSIBLE]{endc}")
        else:
            print(f"  -> acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['type']}  {gray}[SKIP]{endc}")

    return plausible


def evaluate_from_saved(model, tokenizer, model_label, n_samples=16, n_hardest=None, max_tokens=16384, batch_size=None, temperature=0.7):
    if batch_size is None:
        batch_size = n_samples

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

    for i, prob in enumerate(todo):
        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        print(f"[{i+1}/{len(todo)}] idx={prob['idx']} sampling...", end=" ", flush=True)

        responses = generate_samples(model, tokenizer, prompt, n_samples, max_tokens, batch_size, temperature)
        print(f"{len(responses)}/{n_samples} valid")

        if not responses:
            print(f"  {gray}-> no valid samples, skipping{endc}")
            continue

        scored = score_problem(responses, prob["gold_answer"])
        result = {
            "idx": prob["idx"],
            "problem": prob["problem"],
            "gold_answer": prob["gold_answer"],
            "level": prob.get("level", ""),
            "type": prob.get("type", ""),
            **scored,
        }

        existing_results.append(result)
        append_result(out_path, result)
        print(f"  {cyan}-> acc={result['accuracy']:.2f} ({result['num_correct']}/{result['num_total']}) {result['type']}{endc}")

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
    parser.add_argument("--max-tokens", type=int, default=16384, help="max new tokens per generation")
    parser.add_argument("--batch-size", type=int, default=None, help="sequences per generate call (default: n_samples)")
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--filter-divergent", type=str, default=None, metavar="CROSS_FILE")
    parser.add_argument("--num", type=int, default=15, help="number of top divergent problems (with --filter-divergent)")
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
        model, tokenizer = load_model(MODEL_MAP[args.model])
        evaluate_from_saved(model, tokenizer, args.model, n_samples=args.n_samples, n_hardest=args.n_hardest, max_tokens=args.max_tokens, batch_size=args.batch_size, temperature=args.temperature)
    else:
        print(f"{bold}{cyan}=== Evaluating {args.model} model: {MODEL_MAP[args.model]} ==={endc}")
        model, tokenizer = load_model(MODEL_MAP[args.model])
        plausible = evaluate_model(model, tokenizer, args.model, target_plausible=args.target, n_samples=args.n_samples, max_tokens=args.max_tokens, batch_size=args.batch_size, temperature=args.temperature)
        print(f"{green}Found {len(plausible)} plausible problems for {args.model} model{endc}")
