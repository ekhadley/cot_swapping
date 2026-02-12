#%%
import torch as t
from transformer_lens import HookedTransformer

from utils import load_aime_problems, extract_answer, check_answer

#%%

MODEL_ID = "Qwen/Qwen3-1.7B"  # or "Qwen/Qwen3-14B"
PROMPT_TEMPLATE = "Solve the following math problem. Show your reasoning, then give your final answer in \\boxed{{}}.\n\n{problem}"
model = HookedTransformer.from_pretrained(MODEL_ID, dtype=t.bfloat16, device="cuda")
model.eval()
model.requires_grad_(False)
tokenizer = model.tokenizer

#%%

problems = load_aime_problems()
prob = problems[0]
print(f"idx={prob['idx']} | AIME {prob['year']} #{prob['number']} | answer={prob['answer']}")
print(prob["problem"])
print(prob["answer"])

#%% Generate

prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
input_ids = t.tensor(tokenizer.encode(text), device=model.cfg.device).unsqueeze(0)

with t.no_grad():
    out = model.generate(input_ids, max_new_tokens=16384, do_sample=True, temperature=0.7, verbose=True)

response = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
print(response[-500:])

#%% Score the generated response
extracted = extract_answer(response)
correct = check_answer(extracted, prob["answer"]) if extracted else False
print(f"Extracted: {extracted} | Gold: {prob['answer']} | Correct: {correct}")

#%% Score an arbitrary rollout string
rollout = """<paste a rollout here>"""
ext = extract_answer(rollout)
print(f"Extracted: {ext}")
# check_answer(ext, "expected_answer")
