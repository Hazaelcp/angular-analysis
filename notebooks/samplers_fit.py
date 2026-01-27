import os
import sys
import numpy as np
import zfit
from zfit import z
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats 
import PDFs 

def plot_pull_distribution(vals, errs, true_val, param_name, save_path):

    errs = np.array(errs)
    vals = np.array(vals)
    valid_mask = errs > 0
    
    vals = vals[valid_mask]
    errs = errs[valid_mask]
    
    pulls = (vals - true_val) / errs
    
    # Un pull > 5 sigma es extremadamente raro en una distribución normal
    pulls_clipped = pulls[np.abs(pulls) < 5.0]
    
    if len(pulls_clipped) < 5:
        print(f"Advertencia: No hay suficientes datos válidos para graficar {param_name}")
        return

    mu, std = scipy.stats.norm.fit(pulls_clipped)    
    plt.figure(figsize=(8, 6))
    
    count, bins, ignored = plt.hist(pulls_clipped, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Toys')
    
    # Dibujar la curva del ajuste Gaussiano (Roja)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2, label=f'Fit: $\mu={mu:.2f}, \sigma={std:.2f}$')
    
    # Dibujar la curva Ideal Standard Normal (Negra Punteada) -> Objetivo: mu=0, sigma=1
    plt.plot(x, scipy.stats.norm.pdf(x, 0, 1), 'k--', linewidth=1, label='Ideal (0, 1)')
    
    plt.title(f'Pull Distribution for {param_name}', fontsize=16)
    plt.xlabel(f'({param_name} - Truth) / Error', fontsize=14)
    plt.ylabel('Normalized Entries', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    # Caja de texto con estadísticas
    text_str = f"Mean: {np.mean(pulls_clipped):.3f}\nRMS: {np.std(pulls_clipped):.3f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Guardar
    file_name = os.path.join(save_path, f"Pull_{param_name}.pdf")
    plt.savefig(file_name)
    plt.close()
    print(f"Gráfico guardado: {file_name}")

def main():
    # --- 1. Configuración del Espacio de Fases (3 Dimensiones) ---
    obs_ctK = zfit.Space('cosThetaK', limits=(-1, 1))
    obs_ctL = zfit.Space('cosThetaL', limits=(-1, 1))
    obs_phi = zfit.Space('phi', limits=(-np.pi, np.pi))
    angular_obs = obs_ctK * obs_ctL * obs_phi

    # --- 2. Valores de Verdad (Inyección de Señal) ---
    true_vals = {
        'FL':  0.70, 
        'AFB': -0.10,
        'S3':  0.01, 
        'S4':  0.05, 
        'S5':  -0.10,
        'S7':  0.01, 
        'S8':  0.01, 
        'S9':  0.01
    }

    # --- 3. Definición de Parámetros de Ajuste ---
    # Inicializamos en el valor verdadero para estabilidad, pero damos libertad (lower, upper)
    FL_p  = zfit.Parameter('FL', true_vals['FL'], 0.0, 1.0)
    AFB_p = zfit.Parameter('AFB', true_vals['AFB'], -1.0, 1.0) # AFB físico está entre -1 y 1 teóricamente (aprox)
    S3_p  = zfit.Parameter('S3', true_vals['S3'], -2.0, 2.0)
    S4_p  = zfit.Parameter('S4', true_vals['S4'], -2.0, 2.0)
    S5_p  = zfit.Parameter('S5', true_vals['S5'], -2.0, 2.0)
    S7_p  = zfit.Parameter('S7', true_vals['S7'], -2.0, 2.0)
    S8_p  = zfit.Parameter('S8', true_vals['S8'], -2.0, 2.0)
    S9_p  = zfit.Parameter('S9', true_vals['S9'], -2.0, 2.0)

    # --- 4. Instancia del Modelo ---
    # Usamos la clase AngularPDF_B0KstMuMu que está en tu archivo PDFs.py
    # NOTA: Los nombres de argumentos deben coincidir con el __init__ de PDFs.py
    model = PDFs.AngularPDF_B0KstMuMu(
        angular_obs, 
        FL=FL_p, 
        AFB=AFB_p,
        S3=S3_p, 
        S4=S4_p, 
        S5=S5_p, 
        S7=S7_p, 
        S8=S8_p, 
        S9=S9_p
    )

    # --- 5. Configuración de Pseudoexperimentos (Toys) ---
    n_toys = 100       # Número de experimentos (empieza con 100, sube a 500-1000 para tesis final)
    n_events = 2000    # Eventos por experimento (simulando estadística de datos reales)
    
    # Crear sampler (generador de datos)
    sampler = model.create_sampler(n=n_events)

    # Diccionario para guardar resultados
    results = {k: {'vals': [], 'errs': []} for k in true_vals}

    print(f"--- Iniciando generación de {n_toys} pseudoexperimentos ---")

    # --- 6. Loop de Toys ---
    for i in range(n_toys):
        if i % 10 == 0: 
            sys.stdout.write(f"\rProcesando Toy {i}/{n_toys}...")
            sys.stdout.flush()
        
        # A. Generar nuevos datos aleatorios
        sampler.resample()
        
        # B. Resetear parámetros a valores iniciales (ayuda a Minuit a no perderse)
        # Randomizar ligeramente el punto de inicio puede probar la robustez del fit
        for p_name, val in true_vals.items():
            model.params[p_name].set_value(val) # Aquí reseteamos al valor verdadero

        # C. Construir la función de pérdida (Negative Log Likelihood)
        nll = zfit.loss.UnbinnedNLL(model, sampler)
        
        # D. Minimizar
        minimizer = zfit.minimize.Minuit()
        # minimizer.verbosity = 0 # Descomentar para silenciar output de Minuit
        
        try:
            result = minimizer.minimize(nll)
            
            if result.converged:
                # E. Calcular errores (Hesse) - OBLIGATORIO PARA PULLS
                result.hesse()
                
                # F. Guardar resultados si el fit es válido
                if result.valid:
                    for p_name in true_vals:
                        p_obj = model.params[p_name]
                        
                        val = result.params[p_obj]['value']
                        err = result.params[p_obj]['hesse']['error']
                        
                        results[p_name]['vals'].append(val)
                        results[p_name]['errs'].append(err)
            else:
                pass # Si no converge, simplemente saltamos este toy
                
        except Exception as e:
            # Captura errores numéricos raros
            continue

    print("\n--- Generación finalizada. Creando gráficas de validación ---")

    # --- 7. Graficado y Validación ---
    out_dir = "Validation_Plots"
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    for p_name in true_vals:
        vals = results[p_name]['vals']
        errs = results[p_name]['errs']
        
        if len(vals) > 10: # Solo graficar si tenemos suficientes fits exitosos
            plot_pull_distribution(vals, errs, true_vals[p_name], p_name, out_dir)
        else:
            print(f"Saltando {p_name}: No hay suficientes datos convergentes ({len(vals)}).")

    print(f"\n¡Listo! Revisa la carpeta '{out_dir}' para ver tus gráficos de validación.")

if __name__ == "__main__":
    main()
# import os 
# import sys
# import pandas as pd
# import numpy as np
# import copy
# import random
# import scipy
# import math
# import json
# import zfit
# from zfit import z
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from iminuit import Minuit as Minuit_contour
# sys.path.append('/home/ghcp/Documentos/CINVESTAV/SEMESTRE_1/scripts')
# import mass_models
# import plot_tools
# import common_tools
# import PDFs


# def main():

#     obs_ctK = zfit.Space('cosThetaK', limits=(-1, 1))
#     obs_ctL = zfit.Space('cosThetaL', limits=(-1, 1))
#     obs_phi = zfit.Space('phi', limits=(-np.pi, np.pi))
#     angular_obs = obs_ctK * obs_ctL * obs_phi

#     true_vals = {
#         'FL':  0.70, 'AFB': -0.10,
#         'S3':  0.01, 'S4':  0.05, 'S5':  -0.10,
#         'S7':  0.01, 'S8':  0.01, 'S9':  0.01
#     }


#     FL_param  = zfit.Parameter('FL', true_vals['FL'], -0.1, 1.1) 
#     AFB_param = zfit.Parameter('AFB', true_vals['AFB'], -1.0, 1.0)
#     S3_param  = zfit.Parameter('S3', true_vals['S3'], -1.5, 1.5)
#     S4_param  = zfit.Parameter('S4', true_vals['S4'], -1.5, 1.5)
#     S5_param  = zfit.Parameter('S5', true_vals['S5'], -1.5, 1.5)
#     S7_param  = zfit.Parameter('S7', true_vals['S7'], -1.5, 1.5)
#     S8_param  = zfit.Parameter('S8', true_vals['S8'], -1.5, 1.5)
#     S9_param  = zfit.Parameter('S9', true_vals['S9'], -1.5, 1.5)

#     model = PDFs.Theoretical_Physical_PDF(
#         angular_obs, 
#         FL=FL_param, S3=S3_param, S4=S4_param, S5=S5_param, 
#         AFB=AFB_param, S7=S7_param, S8=S8_param, S9=S9_param
#     )

#     n_toys = 1000      
#     n_events = 2000  
    
#     sampler = model.create_sampler(n=n_events)

#     fit_results = {key: {'vals': [], 'errs': []} for key in true_vals}

#     print(f"Generando {n_toys} toys...")

#     for i in range(n_toys):
#         if i % 50 == 0: print(f"Toy {i}/{n_toys}")
        
#         sampler.resample()
#         for p_name, val in true_vals.items():
#             getattr(model.params[p_name], 'set_value')(val)

#         nll = zfit.loss.UnbinnedNLL(model, sampler)
#         minimizer = zfit.minimize.Minuit()
        
#         try:
#             result = minimizer.minimize(nll)
#             # result.hesse() # Hesse es costoso, úsalo si necesitas pulls precisos
#             # Si quieres velocidad, minuit suele dar error aprox automáticamente, 
#             # pero para Pulls rigurosos, HESSE es obligatorio.
#             result.hesse() 
#         except Exception as e:
#             print(f"Fallo ajuste Toy {i}: {e}")
#             continue

#         if result.converged and result.valid:
#             params = result.params
#             for key in true_vals:
#                 p_obj = model.params[key]
#                 fit_results[key]['vals'].append(params[p_obj]['value'])
#                 fit_results[key]['errs'].append(params[p_obj]['hesse']['error'])
        
#         # Limpieza de memoria ocasional (como hace Oscar)
#         if i % 50 == 0:
#             zfit.run.clear_graph_cache()

#     # 4. Análisis Riguroso (Meta-Fitting) y Graficado con plot_tools
#     print("\n--- Iniciando Meta-Análisis y Graficado ---")
    
#     # Crear carpetas
#     if not os.path.exists("Plots/Pulls_CMS_Style"):
#         os.makedirs("Plots/Pulls_CMS_Style")

#     for key in true_vals:
#         print(f"Analizando parámetro: {key}...")
        
#         vals = np.array(fit_results[key]['vals'])
#         errs = np.array(fit_results[key]['errs'])
#         true_v = true_vals[key]

#         # Filtrar NaNs o infinitos
#         mask = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
#         vals = vals[mask]
#         errs = errs[mask]

#         if len(vals) < 10:
#             print(f"Muy pocos toys válidos para {key}, saltando...")
#             continue

#         # Fit a la Distribución del Parámetro (Bias Check)
#         # Definir espacio para el histograma del valor recuperado
#         delta = (np.max(vals) - np.min(vals)) * 0.1
#         limit_min = np.min(vals) - delta
#         limit_max = np.max(vals) + delta
        
#         obs_param = zfit.Space(f'{key}_meas', limits=(limit_min, limit_max))
#         data_param = zfit.Data.from_numpy(obs_param, vals)

#         # Gaussiana para ajustar la distribución de valores recuperados
#         mu_param = zfit.Parameter(f'mu_{key}', np.mean(vals), limit_min, limit_max)
#         sigma_param = zfit.Parameter(f'sigma_{key}', np.std(vals), 0, (limit_max-limit_min))
#         gauss_param = zfit.pdf.Gauss(mu_param, sigma_param, obs_param)

#         # Ajuste
#         nll_p = zfit.loss.UnbinnedNLL(gauss_param, data_param)
#         min_p = zfit.minimize.Minuit()
#         res_p = min_p.minimize(nll_p)
#         res_p.hesse()

#         # Graficar Distribución de Valores (Usando plot_tools)
#         fig = plt.figure(figsize=(10, 8))
#         axes = plot_tools.create_axes_for_pulls(fig)
#         axes[0].set_title(f'Distribución de {key} (Truth={true_v})', fontsize=20)
        
#         plot_tools.plot_model(
#             vals, gauss_param, 
#             bins=40, 
#             axis=axes[0], 
#             axis_pulls=axes[1],
#             chi_x=0.05, chi_y=0.9,
#             pulls=True, # Muestra pulls del ajuste gaussiano vs histograma
#             print_params=res_p, # Imprime mu y sigma en el plot
#             params_text_opts=dict(x=0.65, y=0.6, ncol=1, fontsize=12)
#         )
#         axes[0].set_xlabel(f'{key} Value')
#         plt.savefig(f"Plots/Pulls_CMS_Style/Dist_{key}.pdf")
#         plt.close()

#         # Fit a la Distribución de PULLS (Calibration Check)
#         # Pull = (Val - Truth) / Error
#         pulls = (vals - true_v) / errs
        
#         # Espacio de Pulls 
#         obs_pull = zfit.Space(f'Pull_{key}', limits=(-5, 5))
#         # Recortar outliers extremos para el fit
#         pulls_clipped = pulls[(pulls > -5) & (pulls < 5)]
#         data_pull = zfit.Data.from_numpy(obs_pull, pulls_clipped)

#         # Gaussiana para los Pulls (Esperamos mu=0, sigma=1)
#         mu_pull = zfit.Parameter(f'mu_pull_{key}', 0.0, -1, 1)
#         sigma_pull = zfit.Parameter(f'sigma_pull_{key}', 1.0, 0.5, 1.5)
#         gauss_pull = zfit.pdf.Gauss(mu_pull, sigma_pull, obs_pull)

#         # Ajuste
#         nll_pull = zfit.loss.UnbinnedNLL(gauss_pull, data_pull)
#         res_pull = min_p.minimize(nll_pull)
#         res_pull.hesse()

#         # Graficar Distribución de Pulls
#         fig = plt.figure(figsize=(10, 8))
#         axes = plot_tools.create_axes_for_pulls(fig)
#         axes[0].set_title(f'Pull Distribution {key}', fontsize=20)

#         plot_tools.plot_model(
#             pulls_clipped, gauss_pull, 
#             bins=30, 
#             axis=axes[0], 
#             chi_x=0.05, chi_y=0.9,
#             axis_pulls=axes[1], # Muestra residuos del ajuste
#             pulls=True, 
#             print_params=res_pull, 
#             params_text_opts=dict(x=0.65, y=0.6, ncol=1, fontsize=12)
#         )
#         axes[0].set_xlabel(f'({key} - Truth)/Error')
#         plt.savefig(f"Plots/Pulls/Pull_{key}.pdf")
#         plt.close()

# ¿
# if __name__ == "__main__":
#     main()