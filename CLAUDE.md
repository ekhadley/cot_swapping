# CoT Swapping

## Project Goal

Study the effects of swapping chains of thought between a weaker and stronger reasoning model mid-generation. Specifically: one model begins the chain of thought, then the other model completes it to the final answer. We measure how this affects reasoning accuracy compared to each model completing its own CoT end-to-end.

## Models

- **Weak:** Qwen3-8B (OpenRouter: `qwen/qwen3-8b`)
- **Strong:** Qwen3-32B (OpenRouter: `qwen/qwen3-32b`)

Both are reasoning models that produce `<think>...</think>` blocks before answering.

## Dataset

MATH benchmark. For each model, we filter to problems in the "borderline" difficulty range — problems that model answers correctly 25–75% of the time (across multiple samples). This gives us problems where reasoning quality actually matters and there's variance to study.

## Experiment Design

1. **Baseline:** Each model solves problems end-to-end, multiple times, to establish per-problem accuracy rates
2. **Filtering:** Select problems in the 25–75% accuracy band for each model
3. **CoT Swapping:** For selected problems:
   - Model A generates partial CoT (at various truncation points)
   - Model B receives the partial CoT as context and completes reasoning + answer
   - Test all four directions: weak→strong, strong→weak, weak→weak, strong→strong
4. **Analysis:** Compare swapped accuracy vs baseline accuracy. Look at how truncation point affects results.

## Key Questions

- Does a strong model benefit from a weak model's partial reasoning (or vice versa)?
- At what point in the CoT does swapping matter most?
- Does the completing model "override" bad reasoning or get led astray by it?

## Technical Notes

- Inference via OpenRouter API (`openai`-compatible client).
- Need temperature > 0 for multiple samples per problem (currently 0.7, 16 samples).
- MATH answers have a standardized `\boxed{}` format for answer extraction. We use `math-verify` for answer comparison.
- Store intermediate results (generated CoTs, extracted answers) to avoid re-running expensive generations.

## eval.py Usage

- `./eval.py --model {weak,strong}` — Baseline eval: find plausible (25–75% accuracy) problems for a model. Saves to `data/raw_results_{label}.jsonl` and `data/plausible_{label}.jsonl`.
- `./eval.py --model {weak,strong} --from-saved` — Cross-eval: evaluate a model on the *other* model's plausible problems. Saves to `data/cross_{label}_on_{other}.jsonl`.
- `./eval.py --model {weak,strong} --filter-divergent CROSS_FILE --num N` — Find the N problems with largest accuracy divergence between a model's plausible set and a cross-eval file. Saves to `data/target_problems.jsonl` with full per-sample data from both models.

## Data Files

- `data/raw_results_{label}.jsonl` — All evaluated problems for a model (full results).
- `data/plausible_{label}.jsonl` — Subset with accuracy in [0.25, 0.75].
- `data/cross_{label}_on_{other}.jsonl` — One model evaluated on the other's plausible problems.
- `data/target_problems.jsonl` — High-divergence problems selected for CoT swapping experiments.

## Current Phase

**Phase 1: Baseline evaluation & problem filtering.** Run each model on MATH problems multiple times, compute per-problem accuracy, filter to the 25–75% band. Cross-evaluate models on each other's plausible sets. Select high-divergence target problems.
