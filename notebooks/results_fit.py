import numpy as np

def leer_resultados():
    filename = "last_fit_results.npz"
    
    print(f"--- Cargando datos de: {filename} ---")
    
    # Cargamos el archivo
    data = np.load(filename)
    
    # Extraemos los arrays
    values = data['values']
    errors = data['errors']      # Ahora podemos acceder a esto directamente
    covariance = data['covariance']
    keys = data['keys']
    
    print("\nResultados del Ajuste Guardado:")
    print(f"{'Observable':<12} | {'Valor':<10} | {'Error':<10}")
    print("-" * 38)
    
    for i, key in enumerate(keys):
        val = values[i]
        err = errors[i]
        print(f"{key:<12} | {val:.5f}    | {err:.5f}")
        
    print("\nMatriz de Covarianza (Shape):", covariance.shape)
    # Si quieres ver la matriz completa:
    # print(covariance)

if __name__ == "__main__":
    leer_resultados()