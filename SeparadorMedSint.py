import pandas as pd

# Cargar el Excel
df = pd.read_excel("medicamentos_separados.xlsx")

# Lista de síntomas que quieres como columnas
sintomas = [
    "febricula", 
    "anemia", 
    "diarrea",  # detectará tanto 'diarrea' como 'diarreas'
    "CMV", 
    "dolor", 
    "hipogammaglobulinemia", 
    "Infeccion", 
    "insuficiencia renal", 
    "catarro",
    "vómitos",
    "osteoporosis",
    "trombopenia",
    "tos",
    "temblores",
    "taquicardia",
    "secreciones"
]

# Lista de medicamentos que quieres como columnas
meds = [
    "tacrolimus", "advagraf", "fluouroracilo", "Myfortic", "Valcyte", 
    "Prograf", "Abelcet", "Prednisona", "Vitamina D", "Anfotericina", 
    "Certican", "CMV", "corticoides", "hierro", "inmunosupresores", 
    "magnesio", "Septrin"
]

# Crear columnas con 1 si aparece, 0 si no
for med in meds:
    df[med] = df['nombre_medicamento'].str.contains(med, case=False, na=False)\
                                             .map({True: "1", False: ""})

# Crear columnas con "1" si aparece, "" si no aparece
for sintoma in sintomas:
    df[sintoma] = df['sintomas_reportados'].str.contains(sintoma, case=False, na=False)\
                                             .map({True: "1", False: ""})

# Guardar el resultado
df.to_excel("sint_sep4.xlsx", index=False)