"""Resume the complete WikiTQ comparison using its saved results."""

import json
import os
from pathlib import Path
import random
import urllib.parse

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "tmp" / "tiktoken-cache"))

from global_values import LLM_NAME, OPENAI_BASE_URL, TABLELLM_VERSION, TASK_TYPE, TMP_DIR
from scripts.experiment import Experiment


def main():
    if TASK_TYPE != "tableqa":
        raise ValueError("Esta reanudacion corresponde a WikiTQ, TASK_TYPE=tableqa.")
    if urllib.parse.urlparse(OPENAI_BASE_URL or "").hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Esta ejecucion utiliza Ollama local.")
    random.seed(42)
    experiment = Experiment(mulprocess_cnt=1, llm_name=LLM_NAME)
    baseline_path = Path(TMP_DIR) / "outputs" / f"eval_results_v{TABLELLM_VERSION}_-1_baseline_nl2sqler.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("total") != 4344:
        raise ValueError("Se requiere la evaluacion NL2SQL completa de las 4344 preguntas.")
    print("NL2SQL ya evaluado:", baseline, flush=True)
    print("Retomando AutoPrep con las tablas guardadas...", flush=True)
    experiment.run_multi_agents_data_prep(sample_num=-1)
    print("WIKITQ_COMPLETO", flush=True)


if __name__ == "__main__":
    main()
