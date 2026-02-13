import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools
import os
import mplhep as hep
hep.style.use("CMS")


# --- 2. LÓGICA FÍSICA Y MATEMÁTICA ---

def get_gram_determinant_vectorized(fl, s3, afb, s9, s4, s7, s5, s8):
    """Calcula el determinante de Gram vectorizado."""
    term1 = 0.25 * fl * (1 - fl)**2
    term2 = - fl * (s3**2) - fl * ((4.0/9.0) * afb**2 + s9**2)
    term3 = - 0.25 * (1 - fl + 2 * s3) * (4 * s4**2 + s7**2)
    term4 = - 0.25 * (1 - fl - 2 * s3) * (s5**2 + 4 * s8**2)
    interf_bracket = (2.0/3.0) * afb * (2 * s4 * s5 + 2 * s7 * s8) \
                   - s9 * (4 * s4 * s8 - s5 * s7)
    
    return term1 + term2 + term3 + term4 - interf_bracket

def generate_data(n_try=1000000):
    """Genera puntos físicos y calcula Det(G)."""
    fl = np.random.uniform(0, 1, n_try)
    s3 = np.random.uniform(-0.5, 0.5, n_try)
    afb = np.random.uniform(-0.75, 0.75, n_try)
    s9 = np.random.uniform(-0.5, 0.5, n_try)
    s4 = np.random.uniform(-0.5, 0.5, n_try)
    s5 = np.random.uniform(-0.5, 0.5, n_try)
    s7 = np.random.uniform(-0.5, 0.5, n_try)
    s8 = np.random.uniform(-0.5, 0.5, n_try)

    # Filtros 
    mask = np.abs(s3) <= 0.5 * (1 - fl)
    mask &= (s3**2 + (4.0/9.0)*afb**2 + s9**2) <= 0.25 * (1 - fl)**2
    
    term_pos = (1 - fl + 2 * s3)
    term_neg = (1 - fl - 2 * s3)
    mask &= (term_neg > 1e-9) & (term_pos > 1e-9)
    
    cond_transv_1 = (4 * s4**2 + s7**2) <= fl * term_neg
    cond_transv_2 = (s5**2 + 4 * s8**2) <= fl * term_pos
    mask &= cond_transv_1 & cond_transv_2

    data = {
        'FL': fl[mask], 'S3': s3[mask], 'AFB': afb[mask], 'S9': s9[mask],
        'S4': s4[mask], 'S5': s5[mask], 'S7': s7[mask], 'S8': s8[mask]
    }
    
    det_vals = get_gram_determinant_vectorized(
        data['FL'], data['S3'], data['AFB'], data['S9'],
        data['S4'], data['S7'], data['S5'], data['S8']
    )
    
    df = pd.DataFrame(data)
    df['DetG'] = det_vals
    return df

print("Generando datos Monte Carlo...")
df_total = generate_data(n_try=2000000)

TOLERANCIA = 0.002
mask_zero = df_total['DetG'].abs() < TOLERANCIA
df_red = df_total[~mask_zero]   # Físicos generales
df_green = df_total[mask_zero]  # Límite Massless 

print(f"Total puntos físicos: {len(df_total)}")
print(f" - Rojos (Volumen): {len(df_red)}")
print(f" - Verdes (Superficie Det=0): {len(df_green)}")


output_dir = "CMS_Plots_2D"
os.makedirs(output_dir, exist_ok=True)

# Lista de parámetros y sus etiquetas LaTeX
params = ['FL', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9', 'AFB']
latex_labels = {
    'FL': r'$F_L$', 
    'S3': r'$S_3$', 
    'S4': r'$S_4$', 
    'S5': r'$S_5$', 
    'S7': r'$S_7$', 
    'S8': r'$S_8$', 
    'S9': r'$S_9$', 
    'AFB': r'$A_{FB}$'
}

combinations = list(itertools.combinations(params, 2))

print(f"\nGenerando {len(combinations)} gráficas en la carpeta '{output_dir}'...")

color_red = 'red'
color_green = 'green'

for i, (x_var, y_var) in enumerate(combinations):
    
    # Crear figura
    fig, ax = plt.subplots()
    ax.scatter(df_red[x_var], df_red[y_var], c=color_red, s=5, alpha=0.8, rasterized=True, label='Physical Region')
    ax.scatter(df_green[x_var], df_green[y_var],  c=color_green, s=8, alpha=1, rasterized=True, label=r'Massless Limit ($\det(G)\approx 0$)')
    ax.set_xlabel(latex_labels[x_var], fontweight='bold')
    ax.set_ylabel(latex_labels[y_var], fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    leg = ax.legend(loc='upper right', frameon=True, fontsize=12, fancybox=False, edgecolor='black')
    leg.get_frame().set_linewidth(1.0)
    ax.autoscale(enable=True, axis='both', tight=True)
    # Dar un 5% de margen 
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    dx = (x_max - x_min) * 0.05
    dy = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    # Guardar
    filename = f"{output_dir}/Plot_{x_var}_vs_{y_var}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig) # Importante cerrar la figura para liberar memoria
    print(f"[{i+1}/{len(combinations)}] Guardado: {filename}")

print("\n¡Proceso terminado! Revisa la carpeta 'CMS_Plots_2D'.")