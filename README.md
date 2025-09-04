# TFG
Extracción automatizada de variables desde historias clínicas en lenguaje natural para el soporte diagnóstico mediante modelos LLMs y AutoML

Este repositorio contiene las bases de datos y los scripts necesarios para el análisis de variables clínicas relacionadas con el servicio de neumología, incluyendo el procesamiento automático del FEV1 y la extracción de variables mediante modelos de lenguaje.

Bases de datos:

- CLAD_final.csv: dataset con las variables proporcionadas por el servicio de neumología.

- CLAD_solf.csv: dataset enriquecido que combina CLAD_final con las variables obtenidas automáticamente mediante LLMs.

Códigos:

- FicheroGroq.py: extracción de variables clínicas utilizando la API de Groq.

- FicheroOllama.py: extracción de variables clínicas utilizando Ollama.

- SeparadorMedSint.py: procesamiento del dataset para separar síntomas y medicamentos.

- UnionFichero.py: integración de CLAD_final y el dataset procesado para generar CLAD_solf.

- Automl.py: pipeline de AutoML para el procesamiento y análisis del FEV1.

Uso:

Extracción de variables
- Ejecutar FicheroGroq.py o FicheroOllama.py para obtener nuevas variables clínicas a partir de las notas.

Procesamiento del dataset

- Ejecutar GestionSintomasMedicamentos.py para separar síntomas y medicamentos.

- Ejecutar UnionDatasets.py para fusionar el dataset inicial con las nuevas variables.

Análisis con AutoML

- Ejecutar AutoML.py para entrenar y evaluar modelos predictivos relacionados con el FEV1
