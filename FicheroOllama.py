import pandas as pd 
import requests 
import time

# Cargar el archivo Excel
df = pd.read_excel("Escritorio/dbo_TblEvolucionYcomentariosPaciente.xlsx")

# Filtrar filas no vacías en la columna Evolucion
evoluciones = df['Evolucion'].dropna()
evoluciones = evoluciones[:10]

"""Eres un asistente experto en procesamiento de lenguaje médico. Tu tarea es leer cuidadosamente el siguiente texto clínico y extraer **todas** las variables clínicas relevantes que se mencionen. Para cada variable identificada, proporciona la siguiente información de forma estructurada:

- **Nombre de la variable**: el concepto clínico identificado (por ejemplo, "diabetes tipo 2", "fiebre", "ibuprofeno").
- **Valor extraido**: el valor o descripción asociado si corresponde (por ejemplo, "38.5°C", "500 mg cada 8 horas", "moderado", "positivo", etc.). Si no hay un valor explícito, escribe "No especificado".
- **Tipo de variable**: clasifica la variable en una de las siguientes categorías: `diagnostico`, `sintoma`, `signo`, `tratamiento`, `medicacion`, `prueba diagnostica`, `resultado de prueba`, `procedimiento`, `antecedente`, u `otro` (si no encaja en las anteriores).
- **Unidad o escala (si aplica)**: indica la unidad de medida, escala o instrumento si se menciona (por ejemplo, "mg/dL", "ECG", "escala de Glasgow", etc.).

Texto clínico a analizar:
\"\"\"{texto}\"\"\""""
# Prompt base
prompt_template = """
Eres un asistente experto en procesamiento de lenguaje médico. Tu tarea es leer cuidadosamente el siguiente texto clínico y extraer **todas** las variables clínicas relevantes que se mencionen. Para cada variable identificada, proporciona la siguiente información de forma estructurada:

- **Nombre del medicamento**: Extrae el nombre del medicamento 
- **Cantidad del medicamento**: Extrae la cantidad del medicamento 


Ejemplos practicos:
**Nombre del medicamento**: Prednisona - **Cantidad del medicamento**: 15 mg/dia  
**Nombre del medicamento**: Prograf - **Cantidad del medicamento**: 4-0-4  
**Nombre del medicamento**: Prograf - **Cantidad del medicamento**: -  
**Nombre del medicamento**: Tapendol - **Cantidad del medicamento**: 25 mg 
**Nombre del medicamento**: Advagraf - **Cantidad del medicamento**: 2,5 mg -
**Nombre del medicamento**: Valcyte - **Cantidad del medicamento**: Retirada = 0 
**Nombre del medicamento**: Colobreaath - **Cantidad del medicamento**:-  
**Nombre del medicamento**: Avdagraf - **Cantidad del medicamento**:2/24h  
**Nombre del medicamento**: Prograf - **Cantidad del medicamento**:5-0-5 


si no hay medicamento pasar a al siguiente
Texto clínico a analizar:
\"\"\"{texto}\"\"\"
"""

# Función para llamar al modelo de Ollama
def query_ollama(texto, model="gemma3:4b"):
    prompt = prompt_template.format(texto=texto)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print("Error:", response.text)
        return None

# Procesar los primeros N textos
resultados = []
for i, texto in enumerate(evoluciones[:10]):  # Ajusta el rango para más textos
    print(f"Procesando texto {i+1}/{len(evoluciones)}...")
    salida = query_ollama(texto)
    resultados.append({
        "texto_original": texto,
        "respuesta_modelo": salida
    })
    time.sleep(1)  # Evita sobrecargar el servidor de Ollama

# Guardar resultados
result_df = pd.DataFrame(resultados)
result_df.to_csv("resultados_gemma3_4b.csv", index=False)
print("¡Hecho! Resultados guardados en 'resultados_ollama.csv'")
