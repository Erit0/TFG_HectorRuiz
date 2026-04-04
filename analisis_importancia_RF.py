import joblib
import pandas as pd

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODELO_PATH = 'random_forest_cic17_best.pkl'

def listar_importancias(modelo_path):
    try:
        # 1. Cargar el modelo Random Forest
        model = joblib.load(modelo_path)
        
        # 2. Extraer las importancias y pasarlas a porcentaje (x 100)
        importances = model.feature_importances_
        
        # 3. Obtener los nombres de las columnas
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
        # 4. Crear un DataFrame para ordenar de mayor a menor fácilmente
        df_importances = pd.DataFrame({
            'Caracteristica': feature_names,
            'Importancia_Porcentaje': importances * 100
        }).sort_values(by='Importancia_Porcentaje', ascending=False).reset_index(drop=True)
        
        # 5. Imprimir con el formato solicitado
        print("Distribución completa del peso en el Random Forest (100%):")
        
        for index, row in df_importances.iterrows():
            nombre = row['Caracteristica']
            valor = row['Importancia_Porcentaje']
            
            # Formatear el valor a 4 decimales
            if valor == 0.0:
                print(f"{nombre}: {valor:.4f}% (Esta variable fue completamente ignorada por la IA al crear las reglas de decisión).")
            else:
                print(f"{nombre}: {valor:.4f}%")
                
    except Exception as e:
        print(f"[-] Error al cargar o analizar el modelo: {e}")

if __name__ == "__main__":
    listar_importancias(MODELO_PATH)