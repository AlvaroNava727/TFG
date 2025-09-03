import pandas as pd


# Cargar los dos ficheros
df1 = pd.read_csv("CLAD_final.csv", sep=";", encoding="latin-1")
df2 = pd.read_excel("sint_sep4.xlsx")

# Unir por la columna 'id', solo los que coinciden en ambos (INNER JOIN)
df_union = pd.merge(df1, df2, on="IdCaso", how="left")

# Eliminar la columna que no necesitas (ejemplo: "columna_inutil")
df_union = df_union.drop(columns=["Evolucion_y"])
df_union = df_union.drop(columns=["FechaRegistro"])
df_union = df_union.drop(columns=["nombre_medicamento"])
df_union = df_union.drop(columns=["sintomas_reportados"])
df_union = df_union.drop(columns=["cantidad_medicamento"])


# Guardar en un nuevo Excel
df_union.to_csv("CLAD_solf.csv", index=False, sep=";", decimal=".")

print("✅ Archivos unidos por ID y columna eliminada -> CLAD_solf.csv")

