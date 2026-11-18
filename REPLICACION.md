# Replica de AutoPrep con un modelo local gratuito

Se ejecutaron las variantes NL2SQL sin preparacion y AutoPrep + NL2SQL sobre
los 4344 ejemplos de prueba WikiTQ y los 2024 ejemplos de prueba TabFact.
Ambas variantes de cada dataset tienen los mismos identificadores, preguntas
y etiquetas. Fecha de finalizacion: 18 de noviembre de 2026.

| Dataset | Sin preparacion | Con AutoPrep | Diferencia (puntos porcentuales) |
| --- | --- | --- | --- |
| WikiTQ | 2015/4344 (46.39 %) | 2081/4344 (47.91 %) | +1.52 |
| TabFact | 1509/2024 (74.56 %) | 1484/2024 (73.32 %) | -1.24 |

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
Set-Content -LiteralPath .\keys.txt -Value 'ollama' -Encoding ascii
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
