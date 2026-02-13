import zfit
from zfit import z
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PDFs import FullAngular_Transformed_PDF # Tu archivo

# --- 1. CONFIGURACIÓN DEL ESPACIO DE FASES ---
cos_l = zfit.Space('cos_l', limits=(-1, 1))
cos_k = zfit.Space('cos_k', limits=(-1, 1))
phi   = zfit.Space('phi',   limits=(-np.pi, np.pi))
obs   = cos_l * cos_k * phi

# --- 2. DATOS UNIFORMES (La prueba de la verdad) ---
# Generamos 50,000 puntos aleatorios UNIFORMES una sola vez.
# Esto barre todo el espacio, incluso donde la PDF podría ser negativa.
n_grid = 50000
uniform_np = np.stack([
    np.random.uniform(-1, 1, n_grid),
    np.random.uniform(-1, 1, n_grid),
    np.random.uniform(-np.pi, np.pi, n_grid)
], axis=-1)
uniform_data = zfit.Data.from_numpy(obs=obs, array=uniform_np)

# --- 3. GENERADOR DE PARÁMETROS TRANSFORMADOS ---
def get_random_transformed_params():
    # Usamos una sigma amplia (2.0) para saturar las tanh y probar bordes
    return {
        'rFL': np.random.normal(0, 2),
        'rS3': np.random.normal(0, 2),
        'rS9': np.random.normal(0, 2),
        'rAFB': np.random.normal(0, 2),
        'rS4': np.random.normal(0, 2),
        'rS7': np.random.normal(0, 2),
        'rS5': np.random.normal(0, 2),
        'rS8': np.random.normal(0, 2)
    }

# --- 4. INICIALIZACIÓN ---
# Nota: Los nombres en zfit.Parameter deben coincidir con los que espera tu clase
# Si tu __init__ hace: params = {'rFL': raw_FL ...}
# Entonces aquí pasamos los zfit.Parameter correspondientes.

params_container = {
    'raw_FL': zfit.Parameter('rFL', 0.1), 
    'raw_S3': zfit.Parameter('rS3', 0.1),
    'raw_S9': zfit.Parameter('rS9', 0.1), 
    'raw_AFB': zfit.Parameter('rAFB', 0.1),
    'raw_S4': zfit.Parameter('rS4', 0.1), 
    'raw_S7': zfit.Parameter('rS7', 0.1),
    'raw_S5': zfit.Parameter('rS5', 0.1), 
    'raw_S8': zfit.Parameter('rS8', 0.1)
}

pdf_func = FullAngular_Transformed_PDF(obs, **params_container)

# --- 5. BUCLE DE PRUEBA ---
n_iterations = 2000
epsilon = 1e-2 # Tolerancia un poco más estricta para float64 default

print(f"{'ITER':<5} | {'STATUS':<10} | {'MIN VALUE':<12} | {'NEGATIVES':<10}")
print("-" * 55)

for i in range(n_iterations):
    # 1. Nuevos parámetros aleatorios
    new_vals = get_random_transformed_params()
    
    # 2. Asignar valores a los zfit.Parameters
    # Mapeamos las claves del diccionario 'new_vals' a los objetos Parameter
    params_container['raw_FL'].set_value(new_vals['rFL'])
    params_container['raw_S3'].set_value(new_vals['rS3'])
    params_container['raw_S9'].set_value(new_vals['rS9'])
    params_container['raw_AFB'].set_value(new_vals['rAFB'])
    params_container['raw_S4'].set_value(new_vals['rS4'])
    params_container['raw_S7'].set_value(new_vals['rS7'])
    params_container['raw_S5'].set_value(new_vals['rS5'])
    params_container['raw_S8'].set_value(new_vals['rS8'])
    
    # 3. EVALUAR EN DATA UNIFORME (No resamplear)
    # Esto evalúa la PDF en la malla fija
    probs = pdf_func.pdf(uniform_data).numpy()
    
    # 4. Chequeo
    min_val = np.min(probs)
    negatives = np.sum(probs < -epsilon)
    
    status = "OK" if negatives == 0 else "FAIL"
    print(f"{i+1:<5} | {status:<10} | {min_val:.6e}  | {negatives:<10}")

    if status == "FAIL":
        print(f"   >>> ¡Detectado! Parametrización incompleta.")
        print(f"   >>> Cumple desigualdades 2x2 pero viola positividad global.")
        