"""Run the published AutoPrep comparison on the public 2,024-case TabFact test set."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "tmp" / "tiktoken-cache"))

import global_values as config

# Configure the task before importing agents, which select their prompts at import time.
config.TASK_TYPE = "tablefact"
config.SPLIT = "test"
config.SAMPLE_NUM = -1
config.MULPROCESS_CNT = 1
config.SAVE_STEP = 1
config.TMP_DIR = "./tmp/tabfact"
config.DATA_PATH = str(ROOT / "tmp" / "datasets")
config.TABLELLM_VERSION = (
    f'{config.VERSION_NUM}AutoPrep-{config.TASK_TYPE}-{config.SPLIT}-'
    f'{config.LLM_NAME.replace("/", "_").split("-")[0]}'
)
config.MULTIA_TEMP_DATA_PATH = f"{config.TMP_DIR}/mul-dp_temp_data_v{config.TABLELLM_VERSION}.pkl"

SOURCE_COMMIT = "1fd716c608f56e7d9d156ead432219c7ab9008af"
SOURCE_URL = f"https://raw.githubusercontent.com/google-research/chain-of-table/{SOURCE_COMMIT}/data.zip"
SOURCE_BLOB = "d761ad106da39ec2f55de6dca82f82db6ca3eee8"


def prepare_test_data():
    # Import through the experiment entry point to respect the repository's import order.
    from scripts.experiment import Experiment
    from src.data import TFVData
    import pandas as pd

    source_dir = ROOT / "tmp" / "datasets" / "TFV" / "tabfact" / "chain_of_table"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "chain-of-table-data.zip"
    if not archive_path.exists():
        archive_path.write_bytes(urllib.request.urlopen(SOURCE_URL, timeout=60).read())
    archive = archive_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(archive)).encode() + b"\0" + archive).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError("La descarga no coincide con el archivo oficial verificado.")
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        test_bytes = package.read("data/tabfact/test.jsonl")
    (source_dir / "test.jsonl").write_bytes(test_bytes)
    records = [json.loads(line) for line in test_bytes.decode("utf-8").splitlines()]
    if len(records) != 2024:
        raise ValueError(f"Se esperaban 2024 ejemplos y se encontraron {len(records)}.")

    instances = []
    for index, record in enumerate(records):
        table = record["table_text"]
        if len(table) < 2 or record["label"] not in (0, 1):
            raise ValueError(f"Ejemplo TabFact invalido en la posicion {index}.")
        instances.append(TFVData(
            dataset_name="tabfact",
            tbl=pd.DataFrame(columns=table[0], data=table[1:]),
            question=record["statement"],
            label=str(record["label"]),
            id=f"test_{index}",
            caption=record["table_caption"],
        ))

    output_dir = ROOT / "tmp" / "tabfact"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "tablefact_test_-1.pkl"
    # Experiment.load_datas consumes this test-only cache, avoiding unrelated train/valid files.
    temporary = cache.with_suffix(".pkl.tmp")
    with temporary.open("wb") as output:
        pickle.dump(instances, output)
    temporary.replace(cache)
    metadata = {
        "source": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "archive_git_blob_sha1": blob,
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "records": len(instances),
        "labels": dict(Counter(record["label"] for record in records)),
        "model": config.LLM_NAME,
        "task": config.TASK_TYPE,
        "temperature": config.LLM_HYPER_PARAMS["temperature"],
        "sample_seed": 42,
        "sample_num": config.SAMPLE_NUM,
        "max_input_limit": config.MAX_INPUT_LIMIT,
        "max_output_limit": config.MAX_OUTPUT_LIMIT,
        "python": sys.version,
    }
    (output_dir / "config_ejecucion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"TabFact preparado: {len(instances)} ejemplos.", flush=True)
    return Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparar", action="store_true", help="Preparar datos sin consultar el modelo")
    args = parser.parse_args()
    if urllib.parse.urlparse(config.OPENAI_BASE_URL or "").hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Esta ejecucion esta configurada para utilizar Ollama local.")
    random.seed(42)
    experiment_class = prepare_test_data()
    if args.preparar:
        return
    experiment = experiment_class(mulprocess_cnt=1, llm_name=config.LLM_NAME)
    print("Ejecutando NL2SQL en TabFact...", flush=True)
    experiment.run_nl2sql_baseline(sample_num=-1)
    print("Ejecutando AutoPrep + NL2SQL en TabFact...", flush=True)
    experiment.run_multi_agents_data_prep(sample_num=-1)
    print("TABFACT_COMPLETO", flush=True)


if __name__ == "__main__":
    main()
