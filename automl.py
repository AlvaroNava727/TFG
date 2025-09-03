import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, r2_score


# 📂 Cargar el archivo CSV (ya contiene 'FechaTrasplante')
file_path = "CLAD_solf.csv"
df = pd.read_csv(file_path, sep=';', encoding='latin1', low_memory=False)

df = df.dropna(subset=['FEV1por_Actualpor'])

# 🔹 Convertir 'Fecha' y 'FechaTrasplante' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
df['FechaTrasplante'] = pd.to_datetime(df['FechaTrasplante'], format='%d/%m/%Y', errors='coerce')

# 🔹 Calcular "DiasDesdeTransplante"
df['DiasDesdeTransplante'] = np.where(
    df['Fecha'] < df['FechaTrasplante'], 0,  # Si la fecha es anterior al trasplante, poner 0
    (df['Fecha'] - df['FechaTrasplante']).dt.days  # Si es posterior, calcular días transcurridos
)

# 🔹 Crear la columna 'Transplante' (1 si la fecha es posterior o igual al trasplante, 0 si es anterior)
df['Transplante'] = np.where(df['Fecha'] >= df['FechaTrasplante'], 1, 0)

# 🔹 Eliminar columnas no utilizadas
df = df.drop(columns=['IdCaso','Evolucion', 'talla', 'FEV1por_Actualml', 'FechaTrasplante', 'Ausencia de rechazo agudo'], errors='ignore')
df['Altura'] = df['Altura'].round(2)
df = df.dropna(thresh=15)

# 🔹 Eliminar columnas con más del 97% de valores NaN


# 🔹 Separar tensión arterial en dos columnas
df[['Tension_Sistolica', 'Tension_Diastolica']] = df['TensionArterial'].str.split('/', expand=True).astype(float)
df = df.drop(columns=['TensionArterial'])

# 🔹 Convertir variables categóricas a numéricas
temp_mapping = {
    'Hipotermia': 0,
    'Normotermia': 1,
    'Febrícula': 2,
    'Hipertermia': 3,
    
}
df['GradoTemperatura'] = df['GradoTemperatura'].map(temp_mapping)

sexo_mapping = {
    'M': 1,  # Mujer → 1
    'V': 0   # Varón → 0
}
df['Sexo'] = df['Sexo'].map(sexo_mapping)

test6min_mapping = {
    'Sí': 1,  # Sí → 1
    'No': 0   # No → 0
}
df['Test6min'] = df['Test6min'].map(test6min_mapping)

df["Tratamiento con ADO"] = np.where(df["DiabetesMellitus"] == "Tratamiento con ADO", 1, 
                                     np.where(df["DiabetesMellitus"].isna(), np.nan, 0))

df["Tratamiento con Insulina"] = np.where(df["DiabetesMellitus"] == "Tratamiento con Insulina", 1, 
                                          np.where(df["DiabetesMellitus"].isna(), np.nan, 0))

# 🔹 Eliminar la columna original "DiabetesMellitus"
df.drop(columns=["DiabetesMellitus"], inplace=True)

hta_mapping = {
    'No': 0,
    'Sí': 1
}
df['HTA'] = df['HTA'].map(hta_mapping)

dislip_mapping = {
    'No': 0,
    'Sí': 1
}
df['Dislipemia'] = df['Dislipemia'].map(dislip_mapping)

gs_mapping = {'M': 1, 'AB': 2, 'B': 3}
df['GrupoSanguineo'] = df['GrupoSanguineo'].map(gs_mapping)

# 🔹 Lista de los valores únicos en la columna "Tipo"
medicamentos = ["Tacrolimus", "Sirolimus", "Everolimus", "Ciclosporina 0", "Ciclosporina 2", "MMF", "Digoxina", "Levetiracetam", "Voriconazol"]

# 🔹 Crear columnas binarias (1 si "Tipo" contiene ese medicamento, 0 si no)
for medicamento in medicamentos:
    df[medicamento] = (df["Tipo"] == medicamento).astype(int)

# 🔹 Eliminar la columna original "Tipo" para evitar duplicación
df.drop(columns=["Tipo"], inplace=True)

descripcion_mapping = {
    "Secreciones purulentas": 5,
    "Estenosis vía aérea": 4,
    "Dehiscencia": 3,
    "Otros": 2,
    "Normal": 1
}
df["DescripcionBroncoscopia"] = df["DescripcionBroncoscopia"].map(descripcion_mapping)





# 🔹 Eliminar columnas con más del 97% de valores NaN
umbral = 0.97
df = df.loc[:, df.isnull().mean() < umbral]

df.columns = [col.replace("[", "").replace("]", "").replace("<", "_") for col in df.columns]

print(len(df))

# 🔹 Imprimir número de pacientes únicos en la columna "Registro"
num_pacientes = df["Registro"].nunique()
print(f'🔹 Número total de pacientes: {num_pacientes}')

# 🔹 Separar el 10% final de cada paciente como test (solo si hay al menos 10 registros)
# Se conserva el índice original en test_data para que la separación sea correcta.
test_data = df.groupby("Registro").apply(
    lambda x: x.iloc[-max(1, int(len(x) * 0.1)):]).reset_index(level=0, drop=True)
train_data = df[~df.index.isin(test_data.index)]

# 🔹 Eliminar filas donde FEV1por_Actualpor sea NaN
train_data = train_data.dropna(subset=['FEV1por_Actualpor'])
test_data = test_data.dropna(subset=['FEV1por_Actualpor'])
print("Número de muestras en test_data:", len(test_data))
print("Número de muestras en train_data:", len(train_data))

# 🔹 Preparar los datos para AutoGluon
# Se eliminan columnas que no se desean usar como características (por ejemplo, 'Registro' y 'Fecha')
drop_cols = ['Registro', 'Fecha']
train_data_ag = train_data.drop(columns=drop_cols, errors='ignore')
test_data_ag = test_data.drop(columns=drop_cols, errors='ignore')

# 🔹 Entrenar el modelo con AutoGluon
# AutoGluon utiliza el DataFrame completo (con la variable objetivo incluida) y se encarga de la selección de modelos.
predictor = TabularPredictor(label='FEV1por_Actualpor', problem_type='regression', path='AutogluonModels').fit(train_data_ag, hyperparameters={
            'GBM': {}, 
            'CAT': {}, 
            'XT': {},           
            'XGB': {},
            'RF': {} 
        }, presets='best_quality_v082')

# 🔹 Realizar predicciones en el conjunto de test
# Se debe eliminar la columna objetivo para la predicción.
X_test_ag = test_data_ag.drop(columns=['FEV1por_Actualpor'], errors='ignore')
y_test = test_data_ag['FEV1por_Actualpor']

# 📂 Guardar el conjunto de test sin la variable objetivo
test_csv_path = r"Dataset_Test.csv"  # Ajusta la ruta si es necesario
X_test_ag.to_csv(test_csv_path, index=False, sep=';', encoding="utf-8")

print(f"✅ Archivo de test guardado en: {test_csv_path}")


y_pred = predictor.predict(X_test_ag)

# 🔹 Calcular RMSE y R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'✅ RMSE: {rmse:.4f}')
print(f'✅ R^2: {r2:.4f}')

# 🔹 Opcional: Mostrar el leaderboard con los modelos entrenados
print(predictor.leaderboard(test_data_ag, silent=True))

X_all = pd.concat([train_data, test_data], axis=0)  # Unimos todos los datos
X_all = X_all.drop(columns=['Registro', 'Fecha'], errors='ignore')  # Eliminar columnas no numéricas

feature_importance = predictor.feature_importance(X_all, subsample_size=None)

# 📌 Ordenar por importancia descendente
feature_importance = feature_importance.sort_values(by="importance", ascending=False)

# 📌 Mostrar las primeras 15 características más importantes
print(feature_importance.head(15))

# 📊 Visualizar la importancia de las características
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance.index[:15], feature_importance["importance"][:15], color="royalblue")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de las Características según AutoGluon")
plt.gca().invert_yaxis()  # Para que la característica más importante aparezca arriba
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()