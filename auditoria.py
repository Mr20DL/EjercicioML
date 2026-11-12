import pandas as pd

# 1. Cargar los datos
# Usamos utf-8-sig en predicciones_test para evitar problemas con el BOM (\ufeffid)
df_reclamos = pd.read_csv("reclamos.csv", encoding="utf-8")
df_pred = pd.read_csv("predicciones_test.csv", encoding="utf-8-sig")

# 2. Unir los datos por el 'id' (solo traemos la columna 'grupo' del archivo original)
df_auditoria = pd.merge(df_pred, df_reclamos[['id', 'grupo']], on='id', how='left')

# 3. Calcular si la predicción fue correcta
df_auditoria["es_correcto"] = df_auditoria["categoria_real"] == df_auditoria["categoria_predicha"]

# 4. Calcular si requirió revisión manual (según la columna estado)
df_auditoria["requiere_revision"] = df_auditoria["estado"] == "revisión manual"

# 5. Agrupar por 'grupo' y calcular las métricas
reporte = df_auditoria.groupby("grupo").agg(
    total_muestras=("id", "count"),
    precision_modelo=("es_correcto", "mean"),
    tasa_revision_manual=("requiere_revision", "mean")
).reset_index()

# 6. Formatear los resultados como porcentajes
reporte["precision_modelo"] = (reporte["precision_modelo"] * 100).round(2).astype(str) + "%"
reporte["tasa_revision_manual"] = (reporte["tasa_revision_manual"] * 100).round(2).astype(str) + "%"

print("--- REPORTE DE EQUIDAD POR GRUPO ---")
print(reporte.to_string(index=False))

# 7. Identificar a qué grupo le toca más revisión humana
# Volvemos a calcular en valores numéricos para hallar el máximo
df_num = df_auditoria.groupby("grupo")["requiere_revision"].mean().reset_index()
grupo_max = df_num.loc[df_num["requiere_revision"].idxmax()]

print("\n--- CONCLUSIÓN ---")
print(f"El grupo con mayor tasa de revisión manual es: '{grupo_max['grupo']}' "
      f"con un {(grupo_max['requiere_revision']*100):.2f}% de sus predicciones derivadas a revisión.")