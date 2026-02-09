import asyncio
import os
from dataclasses import dataclass

import aiohttp


API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 10
BASE_DELAY = 2.0
MAX_DELAY = 60.0


@dataclass
class APIResult:
    full_text: str      # reasoning + "</think>" + content
    reasoning: str
    content: str
    api_created: int


async def call_openrouter(session: aiohttp.ClientSession, model_id: str, prompt: str, temperature: float = 0.7, max_tokens: int = 16384) -> APIResult:
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cot-swapping",
        "X-Title": "CoT Swapping Eval",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        async with session.post(API_URL, json=payload, headers=headers) as resp:
            if resp.status in (429, 500, 502, 503):
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                await asyncio.sleep(delay)
                continue
            body = await resp.json()
            assert resp.status == 200, f"OpenRouter API error {resp.status}: {body}"

        choice = body["choices"][0]["message"]
        reasoning = choice.get("reasoning", "") or ""
        content = choice.get("content", "") or ""
        if reasoning:
            full_text = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            full_text = content

        return APIResult(
            full_text=full_text,
            reasoning=reasoning,
            content=content,
            api_created=body.get("created", 0),
        )

    assert False, f"OpenRouter API failed after {MAX_RETRIES} retries"
