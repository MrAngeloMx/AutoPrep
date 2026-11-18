"""Evalua las respuestas WikiTQ guardadas sin consultar al modelo."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "tmp" / "tiktoken-cache")

from global_values import TASK_TYPE, TABLELLM_VERSION, MULPROCESS_CNT, LLM_NAME
from scripts.experiment import Experiment
from src.tools.utils import is_date
from src.tools.binder_utils.evaluator import Evaluator

if TASK_TYPE != "tableqa":
    raise RuntimeError("Se requiere TASK_TYPE=tableqa")
assert not is_date("9999999999")
assert is_date("2013-05-19")
assert not is_date("esto no es una fecha")
assert not Evaluator().evaluate([9999999999], ["19-May-13"], "wikitq", question="when was the last time that tigers mask won?")

versions = [TABLELLM_VERSION + "_-1_" + suffix for suffix in ("baseline_nl2sqler", "data_prep_run_nl2sql")]
records = []
for version in versions:
    path = ROOT / "tmp" / "outputs" / ("result_v" + version + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) != 4344:
        raise RuntimeError("WikiTQ incompleto: " + str(path))
    records.append(data)
assert records[0].keys() == records[1].keys()
assert all(records[0][key]["label"] == records[1][key]["label"] and records[0][key]["question"] == records[1][key]["question"] for key in records[0])

experiment = Experiment(MULPROCESS_CNT, LLM_NAME)
for version in versions:
    experiment.evaluate_result(version, analyze=True)
print("WIKITQ_EVALUADO", flush=True)
