import os
import re
import pandas as pd
import json
from groq import Groq

# ================================
# 🔽 Función para extraer JSON de la respuesta
# ================================
def extract_json(raw_text):
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return raw_text  # si no encuentra, devuelve todo

# ================================
# 🔽 Configurar API Key
# ================================
os.environ["GROQ_API_KEY"] = "llave_de_groq"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ================================
# 🔽 Cargar Excel original
# ================================
df = pd.read_excel("dbo_con_cosas.xlsx")

# ================================
# 🔽 Prompt para Groq
# ================================
prompt_template = """Eres un asistente experto en procesamiento de lenguaje médico.
Tu tarea es leer cuidadosamente el siguiente texto clínico y extraer todas las variables clínicas relevantes que se mencionen.

⚠️ La salida debe ser EXCLUSIVAMENTE un JSON válido.
⚠️ Si hay más de un medicamento, la salida debe ser una LISTA de objetos JSON.
En el caso de no salir ningun medicamento poner debe ser una lista vacia

Cada objeto JSON debe seguir exactamente esta estructura:

{{
  "nombre_medicamento": "",
  "cantidad_medicamento": null,
  "estado_medicamento": "No especifica",   // Opciones: "Aumenta la dosis", "Disminuye la dosis", "Mantiene la dosis", "No especifica"
  "sintomas_reportados": "",
  "estado_general": "No especifica",        // Opciones: "Mejora", "Empeora", "Se mantiene igual", "No especifica"
  "estado_nutricional": "No especifica"     // Opciones: "Bajo de peso", "Aumento de peso", "Se mantiene igual", "No especifica"
}}

Texto clínico a analizar:
{linea}
"""

# ================================
# 🔽 Procesar todas las filas
# ================================
resultados = []

for fila_idx, fila in df.iterrows():
    print(f"Procesando fila {fila_idx + 1} de {len(df)}...")

    texto_clinico = fila["Evolucion"]
    if not isinstance(texto_clinico, str) or not texto_clinico.strip():
        continue

    lineas = texto_clinico.strip().split('\n')
    fila_resultado = {"fila": fila_idx + 1, "lineas": []}

    for i, linea in enumerate(lineas):
        if not linea.strip():
            continue

        prompt = prompt_template.format(linea=linea)

        try:
            respuesta = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0
            )

            raw_resultado = respuesta.choices[0].message.content.strip()
            raw_resultado = extract_json(raw_resultado)

            try:
                parsed_result = json.loads(raw_resultado)

                # Filtrar objetos vacíos
                if isinstance(parsed_result, list):
                    parsed_result = [
                        obj for obj in parsed_result
                        if obj.get("nombre_medicamento", "").strip() != ""
                    ]

                fila_resultado["lineas"].append({
                    "linea_num": i + 1,
                    "respuesta": parsed_result
                })
            except json.JSONDecodeError:
                fila_resultado["lineas"].append({
                    "linea_num": i + 1,
                    "error": "JSON inválido",
                    "respuesta_cruda": raw_resultado
                })

        except Exception as e:
            fila_resultado["lineas"].append({
                "linea_num": i + 1,
                "error": str(e)
            })

    if fila_resultado["lineas"]:
        resultados.append(fila_resultado)

# ================================
# 🔽 Convertir resultados a DataFrame plano
# ================================
filas_planas = []

for fila in resultados:
    fila_idx = fila["fila"] - 1  # índice ajustado
    for linea in fila["lineas"]:
        if "respuesta" in linea and isinstance(linea["respuesta"], list) and linea["respuesta"]:
            for obj in linea["respuesta"]:
                filas_planas.append({
                    "fila_idx": fila_idx,
                    "nombre_medicamento": obj.get("nombre_medicamento", ""),
                    "cantidad_medicamento": obj.get("cantidad_medicamento", None),
                    "estado_medicamento": obj.get("estado_medicamento", "No especifica"),
                    "sintomas_reportados": obj.get("sintomas_reportados", ""),
                    "estado_general": obj.get("estado_general", "No especifica"),
                    "estado_nutricional": obj.get("estado_nutricional", "No especifica")
                })
        else:
            filas_planas.append({
                "fila_idx": fila_idx,
                "nombre_medicamento": None,
                "cantidad_medicamento": None,
                "estado_medicamento": None,
                "sintomas_reportados": None,
                "estado_general": None,
                "estado_nutricional": None
            })

df_extras = pd.DataFrame(filas_planas)

# Agrupar por fila (si hay varios medicamentos, concatenar con ;)
df_extras_grouped = df_extras.groupby("fila_idx").agg({
    "nombre_medicamento": lambda x: "; ".join([str(v) for v in x if v]),
    "cantidad_medicamento": lambda x: "; ".join([str(v) for v in x if v]),
    "estado_medicamento": lambda x: "; ".join([str(v) for v in x if v]),
    "sintomas_reportados": lambda x: "; ".join([str(v) for v in x if v]),
    "estado_general": lambda x: "; ".join([str(v) for v in x if v]),
    "estado_nutricional": lambda x: "; ".join([str(v) for v in x if v]),
}).reset_index()


# ================================
# 🔽 Alinear columnas por medicamento y manejar None
# ================================
filas_planas = []

for fila in resultados:
    fila_idx = fila["fila"] - 1
    medicamentos = []
    cantidades = []
    estados_medic = []
    sintomas = []
    estados_gen = []
    estados_nutri = []

    for linea in fila["lineas"]:
        if "respuesta" in linea and isinstance(linea["respuesta"], list) and linea["respuesta"]:
            for obj in linea["respuesta"]:
                medicamentos.append(str(obj.get("nombre_medicamento", "")).strip())
                cantidades.append(str(obj.get("cantidad_medicamento") or ""))
                estados_medic.append(str(obj.get("estado_medicamento") or ""))
                sintomas.append(str(obj.get("sintomas_reportados") or ""))
                estados_gen.append(str(obj.get("estado_general") or ""))
                estados_nutri.append(str(obj.get("estado_nutricional") or ""))

    # longitud máxima (número de medicamentos detectados)
    max_len = len(medicamentos)
    if max_len == 0:
        max_len = 1  # al menos un placeholder

    # Rellenar listas cortas con ""
    while len(cantidades) < max_len:
        cantidades.append("")
    while len(estados_medic) < max_len:
        estados_medic.append("")
    while len(sintomas) < max_len:
        sintomas.append("")
    while len(estados_gen) < max_len:
        estados_gen.append("")
    while len(estados_nutri) < max_len:
        estados_nutri.append("")

    filas_planas.append({
        "fila_idx": fila_idx,
        "nombre_medicamento": "; ".join(medicamentos),
        "cantidad_medicamento": "; ".join(cantidades),
        "estado_medicamento": "; ".join(estados_medic),
        "sintomas_reportados": "; ".join(sintomas),
        "estado_general": "; ".join(estados_gen),
        "estado_nutricional": "; ".join(estados_nutri)
    })

# Convertir a DataFrame
df_extras_grouped = pd.DataFrame(filas_planas)




# ================================
# 🔽 Unir al DataFrame original
# ================================
for col in ["nombre_medicamento","cantidad_medicamento","estado_medicamento",
            "sintomas_reportados","estado_general","estado_nutricional"]:
    df[col] = df_extras_grouped.set_index("fila_idx")[col]

# ================================
# 🔽 Guardar Excel actualizado
# ================================
df.to_excel("resultadofinal.xlsx", index=False)
print("✅ Archivo actualizado con columnas extra: resultadofinal.xlsx")
