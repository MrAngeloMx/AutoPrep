**Entorno de la replica de AutoPrep**

El punto de partida es el repositorio https://github.com/ruc-datalab/AutoPrep y
su archivo original requirements.txt. Se utiliza Python 3.9.15.

La instalacion original presenta requisitos incompatibles para regex.
DataPrep 0.4.5 declara regex >=2021.8.3,<2022.0.0 y tiktoken 0.4.0 declara
regex >=2022.1.18. El resolvedor de pip rechaza esa combinacion.

El script instalar_dependencias.py descarga la rueda oficial de DataPrep 0.4.5,
comprueba su SHA-256 y genera una rueda local con version 0.4.5+autoprep1.
En sus metadatos establece regex >=2022.1.18,<2023.0.0. La rueda conserva
los archivos de implementacion originales byte por byte, lo que verifica el
propio script. Tambien regenera el registro RECORD del paquete.

Esta modificacion de metadatos requiere verificar la compatibilidad en ejecucion.
Se debe declarar como un ajuste del entorno al presentar resultados de replica.

constraints-replica.txt fija regex 2022.10.31, dask 2023.5.0,
Werkzeug 2.2.3 y python-dateutil 2.8.2 para este entorno de Python 3.9.
requirements-replica.txt se genera a partir de los requisitos originales y
utiliza la rueda local de DataPrep.

Para instalar desde la raiz del repositorio, con el entorno autoprep activo,
ejecutar lo siguiente.

```powershell
python instalar_dependencias.py
python -m pip check
python -c "from scripts.experiment import Experiment; print('Importaciones correctas')"
```

El script utiliza tmp/dependencias para descargas y archivos temporales.
Los archivos de esa carpeta se regeneran al ejecutar el script en otra maquina.
El archivo original requirements.txt conserva su contenido original.

El instalador descarga ademas el recurso punkt de NLTK en
share/nltk_data del entorno Python. AutoPrep lo utiliza al tokenizar consultas.

Tambien se corrigio scripts/__init__.py. Se retiraron las importaciones de
scripts.evaluate y scripts.puresql porque ambos modulos faltan en el repositorio.
La importacion de Experiment permite acceder al controlador publicado.

Para registrar las versiones instaladas, ejecutar lo siguiente.

```powershell
python -m pip freeze > entorno-replica.txt
```

Ese registro incluye una referencia local a la rueda de DataPrep. Para instalar
en otra maquina se utiliza instalar_dependencias.py, que la vuelve a generar.

La validacion del entorno y las pruebas con datos de los benchmarks son pasos
distintos. Los resultados completos del articulo requieren los datos, modelos y
configuraciones experimentales correspondientes.

El 17 de noviembre de 2026 se verifico el entorno autoprep con Python 3.9.15.
La instalacion inicial se completo con uv 0.13.9 y despues se ejecuto el script
de instalacion con pip sobre ese entorno. La instalacion desde cero con pip
queda pendiente de verificacion.

Las siguientes comprobaciones finalizaron correctamente.

- pip check, con el resultado No broken requirements found.
- Importacion de Experiment y NL2SQLer.
- Limpieza de encabezados y texto y conversion de numeros con base_clean_dataframe.
- Normalizacion de Spain e Italy a ESP e ITA mediante DataPrep.
- Reconocimiento de twenty one como 21 mediante el normalizador del proyecto.
- Conteo de tokens con tiktoken.
- Tokenizacion de SQL con NLTK y generacion del prompt de NL2SQLer.
- Ejecucion de una suma SQL local mediante NL2SQLer.exe_sql, con resultado 5.

Las comprobaciones iniciales usaron una tabla sintetica y una consulta SQL fija.
Posteriormente se completaron las evaluaciones WikiTQ y TabFact con Ollama local.
REPLICACION.md y resultados/comparacion.json contienen los resultados finales.

**Continuacion con TabFact**

ejecutar_tabfact.py configura el experimento en un proceso independiente con
TASK_TYPE=tablefact, el modelo local configurado y los prompts TabFact del
repositorio. Utiliza tmp/tabfact para muestras, tablas preparadas, registros y
resultados. El archivo global_values.py del experimento WikiTQ conserva su
configuracion en disco.

Los datos proceden del archivo data.zip del repositorio oficial de Chain-of-Table.
La revision fijada es 1fd716c608f56e7d9d156ead432219c7ab9008af y el Git blob SHA-1
del archivo es d761ad106da39ec2f55de6dca82f82db6ca3eee8.

https://github.com/google-research/chain-of-table/blob/1fd716c608f56e7d9d156ead432219c7ab9008af/data.zip

Se utiliza exclusivamente data/tabfact/test.jsonl, con 2024 ejemplos,
1004 etiquetas positivas y 1020 negativas. El script registra ademas el SHA-256
del archivo de prueba en tmp/tabfact/config_ejecucion.json.

El cargador publicado exige archivos train, valid y test al leer JSONL.
Para esta evaluacion se construye directamente la cache de prueba que consume
Experiment.load_datas, usando la representacion TFVData y el mismo mapeo de
campos e identificadores test_N del cargador original. La evaluacion utiliza
todos los ejemplos del archivo de prueba. Esta adaptacion de carga y el origen
publico de los datos deben indicarse en el informe de replica.

Para preparar los datos y comprobar la configuracion, ejecutar lo siguiente.

```powershell
python ejecutar_tabfact.py --preparar
```

Para ejecutar ambas variantes de TabFact, ejecutar lo siguiente.

```powershell
python ejecutar_tabfact.py
```

**Errores de servicio durante la ejecucion local**

Durante la preparacion completa de WikiTQ, Ollama devolvio repetidamente HTTP 500
y el cliente original GPTPOOL.query siguio reintentando de forma recursiva.
La ejecucion dejo de guardar avances despues de 426 tablas preparadas.

Se sustituyo esa recursion por un numero acotado de intentos usando el parametro
MAX_ERR_RAISE_CNT del repositorio, cuyo valor es 4. Al agotar esos intentos se
propaga la excepcion al manejo de errores existente de los agentes. El SDK puede
realizar ademas sus propios reintentos internos.

Se verificaron un fallo persistente que termina al agotar el limite y un fallo
transitorio que se recupera en el tercer intento. Las comprobaciones utilizaron
errores simulados y no consultaron el modelo.

El cambio requiere reiniciar el proceso Python para cargar el cliente actualizado.
La ejecucion puede recuperar las tablas ya preparadas desde sus archivos de cache.


**Desbordamiento al evaluar fechas en WikiTQ**

La generacion completa guardo las 4344 respuestas. La evaluacion se detuvo
en nu-2839: la prediccion 9999999999 se comparaba con 19-May-13 y dateutil
lanzaba OverflowError al intentar interpretarla como fecha. is_date ahora
captura OverflowError ademas de ValueError y devuelve False para esa entrada.
Se conserva la respuesta generada y el resto del evaluador publicado.

reevaluar_wikitq.py comprueba ese caso, una fecha valida y un texto invalido,
verifica los mismos 4344 identificadores, preguntas y etiquetas en ambas
variantes y vuelve a evaluar sus respuestas guardadas sin consultar al modelo.
