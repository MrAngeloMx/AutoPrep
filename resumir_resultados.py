"""Verifica y resume las evaluaciones guardadas; no consulta modelos."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "resultados"
DEST.mkdir(exist_ok=True)
summaries = []
hashes = {}
rows = []
for dataset, folder, task, expected in (
    ("WikiTQ", ROOT / "tmp/outputs", "tableqa", 4344),
    ("TabFact", ROOT / "tmp/tabfact/outputs", "tablefact", 2024),
):
    version = "7.0.2.2AutoPrep-" + task + "-test-autoprep_-1"
    variants = []
    for variant, suffix in (("baseline", "baseline_nl2sqler"), ("autoprep", "data_prep_run_nl2sql")):
        result_path = folder / ("result_v" + version + "_" + suffix + ".json")
        metric_path = folder / ("eval_results_v" + version + "_" + suffix + ".json")
        analysis_path = folder / "analyze" / ("analyze_v" + version + "_" + suffix + ".txt")
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        metric = json.loads(metric_path.read_text(encoding="utf-8-sig"))
        with analysis_path.open(encoding="utf-8-sig") as stream:
            incorrect = set(ast.literal_eval(stream.readline()))
        assert len(result) == metric["total"] == expected, (dataset, variant, "total")
        assert incorrect <= set(result), (dataset, variant, "ids")
        assert expected - len(incorrect) == metric["correct"], (dataset, variant, "correct")
        assert abs(metric["accuracy"] - metric["correct"] / expected) < 1e-12
        for path in (result_path, metric_path, analysis_path):
            hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        variants.append((result, metric, incorrect))
        rows.append([dataset, variant, metric["correct"], expected, metric["accuracy"]])
    (baseline, bm, be), (prepared, pm, pe) = variants
    assert baseline.keys() == prepared.keys(), (dataset, "different IDs")
    for key in baseline:
        assert baseline[key]["question"] == prepared[key]["question"], (dataset, key, "question")
        assert baseline[key]["label"] == prepared[key]["label"], (dataset, key, "label")
    paired = {
        "both_correct": expected - len(be | pe),
        "baseline_only_correct": len(pe - be),
        "autoprep_only_correct": len(be - pe),
        "both_wrong": len(be & pe),
    }
    assert sum(paired.values()) == expected
    assert paired["autoprep_only_correct"] - paired["baseline_only_correct"] == pm["correct"] - bm["correct"]
    summaries.append({"dataset": dataset, "baseline": bm, "autoprep": pm,
                      "difference_pp": 100 * (pm["accuracy"] - bm["accuracy"]), "paired": paired})
    with (DEST / ("predicciones_" + dataset.lower() + ".jsonl")).open("w", encoding="utf-8") as stream:
        for key in baseline:
            record = {"id": key, "label": baseline[key]["label"],
                      "baseline_answer": baseline[key]["answer"], "autoprep_answer": prepared[key]["answer"],
                      "baseline_correct": key not in be, "autoprep_correct": key not in pe}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

(DEST / "comparacion.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(DEST / "sha256_fuentes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
with (DEST / "comparacion.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["dataset", "variant", "correct", "total", "accuracy"])
    writer.writerows(rows)

table = "\n".join(
    "| {dataset} | {bc}/{n} ({ba:.2f} %) | {pc}/{n} ({pa:.2f} %) | {d:+.2f} |".format(
        dataset=s["dataset"], bc=s["baseline"]["correct"], n=s["baseline"]["total"],
        ba=s["baseline"]["accuracy"] * 100, pc=s["autoprep"]["correct"],
        pa=s["autoprep"]["accuracy"] * 100, d=s["difference_pp"])
    for s in summaries
)
report = '''# Replica de AutoPrep con un modelo local gratuito

Se ejecutaron las variantes NL2SQL sin preparacion y AutoPrep + NL2SQL sobre
los 4344 ejemplos de prueba WikiTQ y los 2024 ejemplos de prueba TabFact.
Ambas variantes de cada dataset tienen los mismos identificadores, preguntas
y etiquetas. Fecha de finalizacion: 18 de noviembre de 2026.

| Dataset | Sin preparacion | Con AutoPrep | Diferencia (puntos porcentuales) |
| --- | --- | --- | --- |
''' + table + '''

AutoPrep mejoro en WikiTQ y empeoro en TabFact. Estas cifras corresponden a
esta configuracion local. Se replica la comparacion de las dos variantes
principales en los dos datasets; no se reprodujeron todos los modelos,
baselines, estudios de ablacion ni las cifras exactas del articulo.

## Codigo y modelo

- Articulo: https://www.vldb.org/pvldb/vol18/p3504-fan.pdf
- Repositorio original: https://github.com/ruc-datalab/AutoPrep
- Commit original: 823f5d4a1a0d93c6ad3b12910539f7326d298399.
- Modelo: Qwen2.5-Coder 3B, cuantizacion Q4_K_M, alias Ollama autoprep-qwen3b.
- Ollama 0.40.16; Python 3.9.15; Windows; RTX 4050 Laptop de 6 GB; RAM de unos 12 GB.
- Contexto Ollama 8192, limite de entrada del proyecto 7992, salida maxima 1024.
- Temperatura 0.01, semilla 42, un trabajador, guardado por ejemplo.
- Conteo de entrada con tiktoken del proyecto original; no usa el tokenizador
  propio de Qwen para aplicar ese limite.
- API local: http://127.0.0.1:11434/v1. No se utilizo una API de pago.
- WikiTQ: cache wiki_qa.pkl incluida en el repositorio original, test_unseen_data.
- TabFact: test.jsonl de Chain-of-Table, revision fijada en ejecutar_tabfact.py.
  Los hashes, la distribucion de etiquetas y la configuracion estan en
  resultados/tabfact_completo/config_ejecucion.json.

## Ajustes necesarios

ENTORNO_REPLICA.md explica el conflicto de dependencias regex/DataPrep y
el ajuste de metadatos de la rueda local, la retirada de importaciones a
modulos ausentes, la seleccion correcta del evaluador TabFact y el adaptador
de la cache de prueba TabFact.

Se acotaron los reintentos de GPTPOOL.query a MAX_ERR_RAISE_CNT=4 tras fallos
HTTP 500 de Ollama. La preparacion WikiTQ se recupero desde las 426 tablas
guardadas despues de un reinicio. No se descartaron los ejemplos fallidos.

Al evaluar WikiTQ, la prediccion 9999999999 de nu-2839 provocaba OverflowError
en dateutil. is_date ahora devuelve False ante ese error. Se volvieron a
evaluar las dos variantes guardadas: la baseline conservo 2015 aciertos y
AutoPrep completo la evaluacion con 2081. No se modificaron las respuestas.

El evaluador publicado se usa con allow_semantic=True para WikiTQ y con
comparacion binaria para TabFact. Los indicadores por ejemplo exportados
proceden de los registros de ese evaluador, no de una metrica nueva.

## Repetir los pasos

Desde la raiz del repositorio en PowerShell, con conda disponible:

```powershell
conda create -n autoprep python=3.9.15 -y
conda activate autoprep
python instalar_dependencias.py
python -m pip check
ollama pull qwen2.5-coder:3b
ollama create autoprep-qwen3b -f Modelfile.autoprep
Set-Content -LiteralPath .\\keys.txt -Value 'ollama' -Encoding ascii
python verificar_ollama.py
```

Ollama debe estar abierto. keys.txt contiene la cadena local ollama, que
permite inicializar el cliente compatible con OpenAI; no es una clave de pago.
global_values.py debe conservar TASK_TYPE=tableqa, SAMPLE_NUM=-1 y el endpoint
local. ejecutar_tabfact.py configura su propio proceso y carpeta de cache.

Para ejecutar la comparacion completa:

```powershell
python ejecutar_piloto.py
python ejecutar_tabfact.py
python resumir_resultados.py
```

El nombre ejecutar_piloto.py se conserva por continuidad; con SAMPLE_NUM=-1
procesa todos los ejemplos. Los scripts omiten resultados que ya figuran
en sus caches. Para una ejecucion independiente se deben usar una copia
limpia o carpetas nuevas de resultados; repetir un comando sobre la misma
cache permite reanudar, pero no produce una repeticion independiente.

Para recuperar WikiTQ usando la baseline completa ya guardada:

```powershell
python reanudar_wikitq.py
```

Para repetir solo la evaluacion WikiTQ y el resumen, sin generar respuestas:

```powershell
python reevaluar_wikitq.py
python resumir_resultados.py
```

## Archivos de resultados

- resultados/comparacion.csv y comparacion.json: metricas y comparacion por ejemplo.
- resultados/predicciones_wikitq.jsonl y predicciones_tabfact.jsonl: etiquetas,
  respuestas de ambas variantes e indicadores de acierto del evaluador.
- resultados/sha256_fuentes.json: hashes de predicciones completas, metricas y
  analisis originales conservados en tmp.
- resultados/wikitq_completo y tabfact_completo: metricas y configuraciones archivadas.
- tmp/outputs y tmp/tabfact/outputs: caches y registros completos locales.

Las carpetas tmp, datasets, los instaladores y las credenciales estan excluidos
de los archivos nuevos de Git mediante .gitignore. La cache WikiTQ ya incluida
en el commit original sigue formando parte de su historial.

Las cifras se basan en una ejecucion por variante con recuperacion de cache.
No se calcularon intervalos de confianza ni significacion estadistica. Una
semilla fija no garantiza igualdad bit a bit entre distintos equipos o versiones.
'''
(ROOT / "REPLICACION.md").write_text(report, encoding="utf-8")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
print("Comparaciones verificadas e informe generado.")
