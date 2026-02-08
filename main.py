import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from eval import evaluate_model


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


load_env()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

WEAK_MODEL = "qwen/qwen3-1.7b"
STRONG_MODEL = "qwen/qwen3-14b"

cyan = '\x1b[38;2;0;255;255m'
green = '\x1b[38;2;0;255;0m'
bold = '\033[1m'
endc = '\033[0m'


@dataclass
class EvalConfig:
    model_id: str
    model_label: str
    openrouter_base_url: str = OPENROUTER_BASE_URL
    openrouter_api_key: str = OPENROUTER_API_KEY
    samples_per_problem: int = 16
    target_plausible: int = 30
    accuracy_low: float = 0.35
    accuracy_high: float = 0.50
    temperature: float = 0.7
    max_tokens: int = 16384
    max_concurrent: int = 30
    data_dir: str = "data"


async def run():
    os.makedirs("data", exist_ok=True)

    weak_cfg = EvalConfig(model_id=WEAK_MODEL, model_label="weak")
    strong_cfg = EvalConfig(model_id=STRONG_MODEL, model_label="strong")

    print(f"{bold}{cyan}=== Evaluating weak model: {WEAK_MODEL} ==={endc}")
    weak_plausible = await evaluate_model(weak_cfg)
    print(f"{green}Found {len(weak_plausible)} plausible problems for weak model{endc}\n")

    print(f"{bold}{cyan}=== Evaluating strong model: {STRONG_MODEL} ==={endc}")
    strong_plausible = await evaluate_model(strong_cfg)
    print(f"{green}Found {len(strong_plausible)} plausible problems for strong model{endc}\n")


if __name__ == "__main__":
    asyncio.run(run())
