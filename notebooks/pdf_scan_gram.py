import zfit
from zfit import z
import tensorflow as tf
import numpy as np
import pandas as pd 
from PDFs import FullAngular_Physical_PDF 
import math

# --- Configuración del Espacio ---
cos_l = zfit.Space('cos_l', limits=(-1, 1))
cos_k = zfit.Space('cos_k', limits=(-1, 1))
phi   = zfit.Space('phi',   limits=(-np.pi, np.pi))
obs   = cos_l * cos_k * phi

def solve_for_s9(fl, s3, afb, s4, s5, s7, s8):
    """Resuelve la ecuación cuadrática Det(G) = 0 para encontrar S9."""
    N_pos = (1 - fl + 2 * s3)
    N_neg = (1 - fl - 2 * s3)
    
    # Coeficientes a*x^2 + b*x + c = 0 para S9
    a = -fl
    b = (4 * s4 * s8 - s5 * s7)
    
    # Términos constantes del determinante
    diag = 0.25 * fl * (1 - fl)**2 - fl * s3**2
    off_diag_1 = -fl * (4.0/9.0) * afb**2
    off_diag_2 = -0.25 * N_pos * (4 * s4**2 + s7**2)
    off_diag_3 = -0.25 * N_neg * (s5**2 + 4 * s8**2)
    interf_const = -(2.0/3.0) * afb * (2 * s4 * s5 + 2 * s7 * s8)
    
    c = diag + off_diag_1 + off_diag_2 + off_diag_3 + interf_const

    discriminante = b**2 - 4 * a * c
    if discriminante < 0: return None
    
    sqrt_disc = math.sqrt(discriminante)
    # Devolver ambas soluciones posibles para probar cuál es PSD
    return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]

def get_valid_matrix_parameters():
    """
    Genera parámetros verificando que la matriz de Gram completa sea Positiva Semi-Definida.
    """
    while True:
        # 1. Generación aleatoria básica (Pre-filtrado suave)
        fl = np.random.uniform(0.3, 0.9)
        s3 = np.random.uniform(-0.4, 0.4)
        
        # Filtro rápido de orden 1
        if abs(s3) > 0.5 * (1 - fl): continue
        
        afb = np.random.uniform(-0.3, 0.3)
        s4, s5, s7, s8 = np.random.uniform(-0.2, 0.2, 4)

        # 2. Resolver S9 para imponer Det(G) = 0
        s9_solutions = solve_for_s9(fl, s3, afb, s4, s5, s7, s8)
        if s9_solutions is None: continue

        # Probamos ambas soluciones de S9
        np.random.shuffle(s9_solutions) # Aleatorizar cual probamos primero
        
        for s9_cand in s9_solutions:
            # 3. CONSTRUCCIÓN DE LA MATRIZ DE GRAM (Sin el factor N global)
            # Definimos los elementos complejos G_ij
            
            # G11 = FL
            # G22 = 0.5 * (1 - FL - 2*S3)
            # G33 = 0.5 * (1 - FL + 2*S3)
            g11 = fl
            g22 = 0.5 * (1 - fl - 2 * s3)
            g33 = 0.5 * (1 - fl + 2 * s3)
            
            # Elementos fuera de la diagonal (parte real e imaginaria)
            # G12 = (1/sqrt(2)) * (-2S4 + iS7)
            g12 = (1/np.sqrt(2)) * complex(-2*s4, s7)
            
            # G13 = (1/sqrt(2)) * (S5 - 2iS8)
            g13 = (1/np.sqrt(2)) * complex(s5, -2*s8)
            
            # G23 = (2/3)AFB + iS9
            g23 = complex((2.0/3.0)*afb, s9_cand)
            
            # Matriz Hermítica 3x3
            gram_matrix = np.array([
                [g11,         g12,           g13],
                [np.conj(g12), g22,           g23],
                [np.conj(g13), np.conj(g23),  g33]
            ], dtype=complex)

            # 4. CRITERIO ESPECTRAL: Calcular Valores Propios
            # eigvalsh es optimizado para matrices hermíticas
            eigenvalues = np.linalg.eigvalsh(gram_matrix)
            
            # Verificamos que el menor eigenvalue sea >= 0 (con pequeña tolerancia numérica)
            if np.min(eigenvalues) > -1e-6:
                # ¡ENCONTRADO! Esta configuración es físicamente válida
                return {
                    'FL': fl, 'S3': s3, 'S9': s9_cand, 'AFB': afb,
                    'S4': s4, 'S7': s7, 'S5': s5, 'S8': s8
                }
        
        # Si ninguna solución de S9 funcionó, el loop continúa

# --- Generación de Datos Toy ---
n_grid = 2000
uniform_data_np = np.stack([
    np.random.uniform(-1, 1, n_grid),
    np.random.uniform(-1, 1, n_grid),
    np.random.uniform(-np.pi, np.pi, n_grid)
], axis=-1)
uniform_data = zfit.Data.from_numpy(obs=obs, array=uniform_data_np)

# --- Inicialización zfit ---
params_container = {
    'FL': zfit.Parameter('FL', 0.5), 'S3': zfit.Parameter('S3', 0.0),
    'S9': zfit.Parameter('S9', 0.0), 'AFB': zfit.Parameter('AFB', 0.0),
    'S4': zfit.Parameter('S4', 0.0), 'S7': zfit.Parameter('S7', 0.0),
    'S5': zfit.Parameter('S5', 0.0), 'S8': zfit.Parameter('S8', 0.0)
}

pdf = FullAngular_Physical_PDF(obs, **params_container)

print(f"\n{'ITER':<5} | {'STATUS':<10} | {'MIN PDF':<12} | {'MIN EIGEN':<12}")
print("-" * 55)

failed_count = 0
n_iterations = 200

for i in range(n_iterations):
    p = get_valid_matrix_parameters()
    
    for key, val in p.items():
        params_container[key].set_value(val)
        
    probs = pdf.pdf(uniform_data).numpy()
    min_val = np.min(probs)
    
    # Verificación
    negatives = np.sum(probs < -1e-5) 
    status = "OK" if negatives == 0 else "FAIL"
    
    # Chequeo visual del eigenvalue para confirmar
    # Reconstruimos la matriz solo para el print
    # (En producción esto no es necesario porque ya lo validó la función)
    
    if status == "FAIL":
        failed_count += 1
        print(f"\033[91m{i+1:<5} | {status:<10} | {min_val:.4e} | FATAL ERROR\033[0m")
    else:
        if i % 20 == 0:
            print(f"{i+1:<5} | {status:<10} | {min_val:.4e} | Validated")

print(f"\n[DONE] Scan completado. Fallos: {failed_count}/{n_iterations}")