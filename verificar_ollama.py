from pathlib import Path
from time import perf_counter

from openai import OpenAI

from global_values import (
    KEY_FILE,
    LLM_NAME,
    OPENAI_BASE_URL,
    LLM_HYPER_PARAMS,
)

api_key = Path(KEY_FILE).read_text(encoding="utf-8-sig").strip()

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=api_key,
    timeout=120,
    max_retries=0,
)

print("Modelo:", LLM_NAME)
print("Direccion:", OPENAI_BASE_URL)

start = perf_counter()

response = client.chat.completions.create(
    model=LLM_NAME,
    messages=[
        {
            "role": "user",
            "content": "Responde solamente con la palabra OK.",
        }
    ],
    temperature=LLM_HYPER_PARAMS["temperature"],
    max_tokens=16,
)

print("Respuesta:", response.choices[0].message.content)
print("Segundos:", round(perf_counter() - start, 2))