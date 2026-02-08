# CoT Swapping

## Project Goal

Study the effects of swapping chains of thought between a weaker and stronger reasoning model mid-generation. Specifically: one model begins the chain of thought, then the other model completes it to the final answer. We measure how this affects reasoning accuracy compared to each model completing its own CoT end-to-end.

## Models

- **Weak:** Qwen3-1.7B
- **Strong:** Qwen3-14B

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

- Qwen3-14B needs the HPC (A100, 40GB VRAM). Qwen3-1.7B fits locally.
- Use vLLM or transformers for generation. Need temperature > 0 for multiple samples per problem.
- MATH answers have a standardized `\boxed{}` format for answer extraction.
- Store intermediate results (generated CoTs, extracted answers) to avoid re-running expensive generations.

## Current Phase

**Phase 1: Baseline evaluation & problem filtering.** Run each model on MATH problems multiple times, compute per-problem accuracy, filter to the 25–75% band.
