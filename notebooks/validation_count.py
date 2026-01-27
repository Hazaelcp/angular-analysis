import zfit
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import warnings
import os
from itertools import combinations
import pandas as pd

# --- TUS IMPORTS ---
from PDFs import (
    FullAngular_Physical_PDF, 
    FullAngular_Transformed_PDF, 
    get_inverse_values, 
    apply_transformation_equations,
    get_physical_region_scan
)

# Configuración
zfit.settings.set_seed(42)
np.random.seed(42)
warnings.simplefilter('ignore')

# --- FUNCIÓN AUXILIAR (El Motor de la Espada - Covarianza) ---
def calculate_jacobian_numerical(func, params, keys_out, epsilon=1e-5):
    """Calcula la matriz Jacobiana numéricamente."""
    n_params = len(params)
    n_out = len(keys_out)
    J = np.zeros((n_out, n_params))
    
    def func_wrapper(p_args):
        res_dict = func(*p_args)
        return np.array([res_dict[k] for k in keys_out])

    for i in range(n_params):
        p_plus = np.copy(params)
        p_minus = np.copy(params)
        p_plus[i] += epsilon
        p_minus[i] -= epsilon
        
        f_plus = func_wrapper(p_plus)
        f_minus = func_wrapper(p_minus)
        
        deriv = (f_plus - f_minus) / (2 * epsilon)
        J[:, i] = deriv
    return J

def run_multi_view_analysis():
    print(">>> 1. Iniciando Protocolo Gundam HeavyArms (Corrección S8)...")
    
    # --- DEFINICIÓN DE CARPETAS ---
    folder_trans = "Plots/Transformed_Space_vFinal"
    folder_phys_zoom = "Plots/Physical_Space_Zoom_vFinal"
    folder_phys_full = "Plots/Physical_Space_Full_vFinal"
    
    for f in [folder_trans, folder_phys_zoom, folder_phys_full]:
        os.makedirs(f, exist_ok=True)
    
    print(f">>> Directorios listos:\n    - {folder_trans}\n    - {folder_phys_zoom}\n    - {folder_phys_full}")

    # --- SETUP Y GENERACIÓN (JUGUETE) ---
    obs = zfit.Space('cosThetaL', limits=(-1, 1)) * \
          zfit.Space('cosThetaK', limits=(-1, 1)) * \
          zfit.Space('phi', limits=(-np.pi, np.pi))

    # Valores verdaderos físicos
    true_vals_phys = [0.684, -0.013, 0.029, 0.050, -0.145, -0.136, -0.204, 0.077]
    phys_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
    true_dict = dict(zip(phys_keys, true_vals_phys))

    print(">>> Generando datos (Toy MC)...")
    # Aumentamos estadística para estabilidad
    pdf_gen = FullAngular_Physical_PDF(obs, *true_vals_phys)
    sampler = pdf_gen.create_sampler(n=100000) 
    sampler.resample()

    # --- AJUSTE (ESPACIO TRANSFORMADO) ---
    raw_init = get_inverse_values(true_vals_phys)
    r_keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
    param_names_fit = [f"{k}_fit" for k in r_keys]
    
    true_vals_trans_dict = dict(zip(param_names_fit, raw_init))
    
    params = {k: zfit.Parameter(p_name, v, step_size=0.01) 
              for k, p_name, v in zip(r_keys, param_names_fit, raw_init)}
    
    pdf_fit = FullAngular_Transformed_PDF(
        obs, 
        params['rFL'], params['rS3'], params['rS9'], params['rAFB'],
        params['rS4'], params['rS7'], params['rS5'], params['rS8']
    )

    print(">>> 2. Ejecutando Minimización (Minuit)...")
    nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
    minimizer = zfit.minimize.Minuit(tol=0.01) 
    result = minimizer.minimize(nll)
    m = result.info['minuit'] 
    
    print(">>> 3. Calculando Errores MINOS...")
    m.minos() 

    # ==============================================================================
    # >>> REPORTE 1: ESPACIO TRANSFORMADO (VALORES DE MINUIT) <<<
    # ==============================================================================
    print("\n" + "="*80)
    print(">>> REPORTE TÉCNICO 1: Errores MINOS (Espacio Transformado)")
    print("="*80)
    print(f"{'PARAM':<10} | {'VALOR':<10} | {'ERROR -':<10} | {'ERROR +':<10} | {'VERDAD':<10}")
    print("-" * 80)
    
    # Aquí guardamos los valores CORRECTOS de Minuit para usarlos después
    minuit_best_values = []
    
    for rk, pname in zip(r_keys, param_names_fit):
        val = m.values[pname]
        minuit_best_values.append(val) # Guardamos el valor exacto de Minuit
        
        err_low = m.merrors[pname].lower  
        err_high = m.merrors[pname].upper 
        truth = true_vals_trans_dict[pname]
        
        print(f"{rk:<10} | {val:.4f}     | {err_low:.4f}     | +{err_high:.4f}    | {truth:.4f}")
    print("="*80 + "\n")

    # ==============================================================================
    # >>> REPORTE 2: ESPACIO FÍSICO (CORREGIDO PARA USAR MINUIT) <<<
    # ==============================================================================
    print("\n" + "="*80)
    print(">>> REPORTE TÉCNICO 2: Errores Gaussianos (Matriz de Covarianza Propagada)")
    print("="*80)
    
    # 1. Recuperamos la matriz de covarianza (Alineada con los parámetros)
    param_objs_ordered = [params[k] for k in r_keys]
    cov_trans = result.covariance(params=param_objs_ordered)
    
    # 2. IMPORTANTE: Usamos los valores de MINUIT, no los de zfit.Parameter
    best_fit_r_values = np.array(minuit_best_values)
    
    # 3. Jacobiano y Propagación
    J = calculate_jacobian_numerical(apply_transformation_equations, best_fit_r_values, phys_keys)
    cov_phys = J @ cov_trans @ J.T
    phys_errors_sigma = np.sqrt(np.diag(cov_phys))
    phys_errors_dict = dict(zip(phys_keys, phys_errors_sigma))
    
    # 4. Tabla Física (Calculada con los valores de Minuit)
    print(f"{'PARAM':<10} | {'VALOR':<10} | {'ERROR (+/-)':<15} | {'VERDAD':<10}")
    print("-" * 80)
    
    best_fit_phys_dict = apply_transformation_equations(*best_fit_r_values)
    
    for k in phys_keys:
        val = best_fit_phys_dict[k]
        err = phys_errors_dict[k]
        tru = true_dict[k]
        print(f"{k:<10} | {val:.4f}     | +/- {err:.4f}      | {tru:.4f}")
    print("="*80 + "\n")
    # ==============================================================================

    # --- GENERACIÓN DEL MAPA FÍSICO ---
    print(">>> 4. Mapeando la Región Física y Generando Gráficos...")
    df_phys_region = get_physical_region_scan(n_points=10000)

    # --- BUCLE DE PLOTEO ---
    indices = list(range(8))
    pairs_indices = list(combinations(indices, 2))
    total_plots = len(pairs_indices)
    
    for i, (idx_x, idx_y) in enumerate(pairs_indices):
        
        rx, ry = param_names_fit[idx_x], param_names_fit[idx_y]
        px, py = phys_keys[idx_x], phys_keys[idx_y]

        print(f"    [{i+1}/{total_plots}] {px} vs {py} ...", end="\r")

        try:
            # 1. OBTENER CONTORNO (Espacio R)
            contour_r = m.mncontour(rx, ry, cl=0.3935, size=50)

            # ---------------------------------------------------------
            # PLOT A: Espacio Transformado
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 7))
            plt.plot(contour_r[:, 0], contour_r[:, 1], 'b-', linewidth=2, label='1$\sigma$ Contour')
            plt.plot(m.values[rx], m.values[ry], 'bo', label='Best Fit', zorder=10)
            true_tx = true_vals_trans_dict[rx]
            true_ty = true_vals_trans_dict[ry]
            plt.plot(true_tx, true_ty, 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=15)

            # MINOS Errors visuales
            val_x = m.values[rx]; val_y = m.values[ry]
            minos_x_low = val_x + m.merrors[rx].lower; minos_x_high = val_x + m.merrors[rx].upper
            minos_y_low = val_y + m.merrors[ry].lower; minos_y_high = val_y + m.merrors[ry].upper
            
            plt.axvline(minos_x_low, color='k', linestyle='--', alpha=0.4)
            plt.axvline(minos_x_high, color='k', linestyle='--', alpha=0.4)
            plt.axhline(minos_y_low, color='k', linestyle='--', alpha=0.4)
            plt.axhline(minos_y_high, color='k', linestyle='--', alpha=0.4, label='MINOS Errors')
            
            width_x = minos_x_high - minos_x_low; width_y = minos_y_high - minos_y_low
            margin = 0.4
            plt.xlim(minos_x_low - width_x * margin, minos_x_high + width_x * margin)
            plt.ylim(minos_y_low - width_y * margin, minos_y_high + width_y * margin)

            plt.xlabel(rx); plt.ylabel(ry)
            plt.title(f"Transformed: {rx} vs {ry}")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.savefig(f"{folder_trans}/Trans_{rx}_vs_{ry}.png")
            plt.close()

            # ---------------------------------------------------------
            # PLOTS FÍSICOS: ESTILO TESIS
            # ---------------------------------------------------------
            n_pts = len(contour_r)
            r_matrix = np.tile(best_fit_r_values, (n_pts, 1))
            r_matrix[:, idx_x] = contour_r[:, 0]
            r_matrix[:, idx_y] = contour_r[:, 1]
            
            trans_contour_dict = apply_transformation_equations(
                r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2], r_matrix[:, 3],
                r_matrix[:, 4], r_matrix[:, 5], r_matrix[:, 6], r_matrix[:, 7]
            )
            cx_phys = trans_contour_dict[px]
            cy_phys = trans_contour_dict[py]

            # CÁLCULO DE ERRORES GEOMÉTRICOS
            x_min_cont = np.min(cx_phys)
            x_max_cont = np.max(cx_phys)
            y_min_cont = np.min(cy_phys)
            y_max_cont = np.max(cy_phys)
            
            bf_x = best_fit_phys_dict[px]
            bf_y = best_fit_phys_dict[py]
            
            err_x_up = x_max_cont - bf_x
            err_x_down = bf_x - x_min_cont
            err_y_up = y_max_cont - bf_y
            err_y_down = bf_y - y_min_cont

            def plot_physical(view_mode, x_lims=None, y_lims=None):
                plt.figure(figsize=(9, 8))
                alpha_cloud = 0.15 if view_mode == 'Full' else 0.05
                
                # plt.scatter(df_phys_region[px], df_phys_region[py],c='gray', s=1, alpha=alpha_cloud, label='Allowed Region', zorder=0)
                
                plt.plot(cx_phys, cy_phys, 'g-', linewidth=2.5, label='1$\sigma$ Contour')
                plt.plot(bf_x, bf_y, 'bo', markersize=6, label='Best Fit', zorder=10)
                plt.plot(true_dict[px], true_dict[py], 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=11)
                
                plt.axvline(x_min_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
                plt.axvline(x_max_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
                plt.axhline(y_min_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
                plt.axhline(y_max_cont, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Contour Limits')

                plt.xlabel(px, fontsize=14); plt.ylabel(py, fontsize=14)
                
                if view_mode == 'Zoom':
                    title_str = (f"{px} = {bf_x:.4f} (+{err_x_up:.4f}/-{err_x_down:.4f})\n"
                                 f"{py} = {bf_y:.4f} (+{err_y_up:.4f}/-{err_y_down:.4f})")
                    plt.title(title_str, fontsize=11, color='blue')
                    if x_lims: plt.xlim(x_lims); 
                    if y_lims: plt.ylim(y_lims)
                else:
                    plt.title(f"FULL: {px} vs {py}", fontsize=12)
                    plt.xlim(df_phys_region[px].min(), df_phys_region[px].max())
                    plt.ylim(df_phys_region[py].min(), df_phys_region[py].max())

                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper right', frameon=True, fontsize=9)
                
                folder = folder_phys_zoom if view_mode == 'Zoom' else folder_phys_full
                plt.savefig(f"{folder}/Phys_{px}_vs_{py}_{view_mode}.png", dpi=100)
                plt.close()

            # Definir límites de zoom
            mx = (x_max_cont - x_min_cont) * 0.4
            my = (y_max_cont - y_min_cont) * 0.4
            
            plot_physical('Zoom', x_lims=(x_min_cont-mx, x_max_cont+mx), y_lims=(y_min_cont-my, y_max_cont+my))
            plot_physical('Full')

        except Exception as e:
            print(f"\n    [ERROR] Par {px}-{py}: {e}")
            plt.close()

    print(f"\n\n>>> ¡ANÁLISIS COMPLETADO! S8 corregido y verificado.")

if __name__ == "__main__":
    run_multi_view_analysis()



















# import zfit
# import numpy as np
# import matplotlib.pyplot as plt
# from iminuit import Minuit
# import warnings
# import os
# from itertools import combinations
# import pandas as pd

# from PDFs import (
#     FullAngular_Physical_PDF, 
#     FullAngular_Transformed_PDF, 
#     get_inverse_values, 
#     apply_transformation_equations,
#     get_physical_region_scan
# )

# zfit.settings.set_seed(42)
# np.random.seed(42)
# warnings.simplefilter('ignore')

# def calculate_jacobian_numerical(func, params, keys_out, epsilon=1e-5):
#     """Calcula la matriz Jacobiana numéricamente."""
#     n_params = len(params)
#     n_out = len(keys_out)
#     J = np.zeros((n_out, n_params))
    
#     def func_wrapper(p_args):
#         res_dict = func(*p_args)
#         return np.array([res_dict[k] for k in keys_out])

#     for i in range(n_params):
#         p_plus = np.copy(params)
#         p_minus = np.copy(params)
#         p_plus[i] += epsilon
#         p_minus[i] -= epsilon
        
#         f_plus = func_wrapper(p_plus)
#         f_minus = func_wrapper(p_minus)
        
#         deriv = (f_plus - f_minus) / (2 * epsilon)
#         J[:, i] = deriv
#     return J

# def run_multi_view_analysis():
    
#     folder_trans = "Plots/Transformed_Space_v3"
#     folder_phys_zoom = "Plots/Physical_Space_Zoom_v3"
#     folder_phys_full = "Plots/Physical_Space_Full_v3"
    
#     for f in [folder_trans, folder_phys_zoom, folder_phys_full]:
#         os.makedirs(f, exist_ok=True)
    

#     # espacio de variables
#     obs = zfit.Space('cosThetaL', limits=(-1, 1)) * \
#           zfit.Space('cosThetaK', limits=(-1, 1)) * \
#           zfit.Space('phi', limits=(-np.pi, np.pi))

#     # Valores verdaderos físicos lhcb
#     true_vals_phys = [0.684, -0.013, 0.029, 0.050, -0.145, -0.136, -0.204, 0.077]
#     phys_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
#     true_dict = dict(zip(phys_keys, true_vals_phys))

#     print(">>> generando datos (Toy MC)...")
#     pdf_gen = FullAngular_Physical_PDF(obs, *true_vals_phys)
#     sampler = pdf_gen.create_sampler(n=100000)
#     sampler.resample()

#     # fit espacio transformado
#     raw_init = get_inverse_values(true_vals_phys)
#     r_keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
#     param_names_fit = [f"{k}_fit" for k in r_keys]
    
#     true_vals_trans_dict = dict(zip(param_names_fit, raw_init))
    
#     params = {k: zfit.Parameter(p_name, v, step_size=0.01) 
#               for k, p_name, v in zip(r_keys, param_names_fit, raw_init)}
    
#     pdf_fit = FullAngular_Transformed_PDF(
#         obs, 
#         params['rFL'], params['rS3'], params['rS9'], params['rAFB'],
#         params['rS4'], params['rS7'], params['rS5'], params['rS8']
#     )

#     print(">>> 2. minimización (Minuit)...")
#     nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
#     minimizer = zfit.minimize.Minuit(tol=0.01) 
#     result = minimizer.minimize(nll)
#     m = result.info['minuit'] 
    
#     print(">>> 3. calculando errores miinos...")
#     m.minos() 

#     # calculo de la matriz de covarianza y propagación de errores
#     print("\n" + "="*60)
#     print(">>> REPORTE TÉCNICO: Errores Gaussianos (Matriz de Covarianza)")
#     print("="*60)

#     for rk, pname in zip(r_keys, param_names_fit):
#         val = m.values[pname]
#         err_low = m.merrors[pname].lower  
#         err_high = m.merrors[pname].upper 
#         truth = true_vals_trans_dict[pname]
#         print(f"{rk:<10} | {val:.4f}     | {err_low:.4f}     | +{err_high:.4f}    | {truth:.4f}")
#     print("="*80 + "\n")

#     param_objs_ordered = [params[k] for k in r_keys]
#     cov_trans = result.covariance(params=param_objs_ordered)
#     best_fit_r_values = np.array([p.value() for p in param_objs_ordered])
    
#     # jacobiano y Propagación
#     J = calculate_jacobian_numerical(apply_transformation_equations, best_fit_r_values, phys_keys)
#     cov_phys = J @ cov_trans @ J.T
#     phys_errors_sigma = np.sqrt(np.diag(cov_phys))
#     phys_errors_dict = dict(zip(phys_keys, phys_errors_sigma)) # Diccionario de errores sigma
    
#     print(f"{'PARAM':<6} | {'VALOR':<10} | {'ERROR (Gauss)':<15} | {'VERDAD':<10}")
#     print("-" * 50)
#     best_fit_phys_dict = apply_transformation_equations(*best_fit_r_values)
#     for k in phys_keys:
#         val = best_fit_phys_dict[k]
#         err = phys_errors_dict[k]
#         tru = true_dict[k]
#         print(f"{k:<6} | {val:.4f}     | +/- {err:.4f}      | {tru:.4f}")
#     print("="*60 + "\n")

#     # transformación al espacio físico
#     # genero puntos que cumplan las constraints físicas
#     df_phys_region = get_physical_region_scan(n_points=100000)

    
#     # todo esto es solo para plots
#     indices = list(range(8))
#     pairs_indices = list(combinations(indices, 2))
#     total_plots = len(pairs_indices)
    
#     for i, (idx_x, idx_y) in enumerate(pairs_indices):
        
#         rx, ry = param_names_fit[idx_x], param_names_fit[idx_y]
#         px, py = phys_keys[idx_x], phys_keys[idx_y]

#         print(f"    [{i+1}/{total_plots}] {px} vs {py} ...", end="\r")

#         try:
#             # obtener contorno
#             contour_r = m.mncontour(rx, ry, cl=0.3935, size=50)
#             #plot espacio transformado
#             plt.figure(figsize=(8, 7))
#             plt.plot(contour_r[:, 0], contour_r[:, 1], 'b-', linewidth=2, label='1$\sigma$ Contour')
#             plt.plot(m.values[rx], m.values[ry], 'bo', label='Best Fit', zorder=10)
#             true_tx = true_vals_trans_dict[rx]
#             true_ty = true_vals_trans_dict[ry]
#             plt.plot(true_tx, true_ty, 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=15)

#             # errores minos
#             val_x = m.values[rx]; val_y = m.values[ry]
#             minos_x_low = val_x + m.merrors[rx].lower; minos_x_high = val_x + m.merrors[rx].upper
#             minos_y_low = val_y + m.merrors[ry].lower; minos_y_high = val_y + m.merrors[ry].upper
            
#             plt.axvline(minos_x_low, color='k', linestyle='--', alpha=0.4)
#             plt.axvline(minos_x_high, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_low, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_high, color='k', linestyle='--', alpha=0.4, label='MINOS Errors')
            
#             # limites para el plot
#             width_x = minos_x_high - minos_x_low; width_y = minos_y_high - minos_y_low
#             margin = 0.4
#             plt.xlim(minos_x_low - width_x * margin, minos_x_high + width_x * margin)
#             plt.ylim(minos_y_low - width_y * margin, minos_y_high + width_y * margin)

#             plt.xlabel(rx); plt.ylabel(ry)
#             plt.title(f"Transformed: {rx} vs {ry}")
#             plt.grid(True, alpha=0.3); plt.legend()
#             plt.savefig(f"{folder_trans}/Trans_{rx}_vs_{ry}.png")
#             plt.close()

#             # plots físicos
#             n_pts = len(contour_r)
#             r_matrix = np.tile(best_fit_r_values, (n_pts, 1))
#             r_matrix[:, idx_x] = contour_r[:, 0]
#             r_matrix[:, idx_y] = contour_r[:, 1]
            
#             # transforma TODOS los puntos del contorno al espacio físico
#             trans_contour_dict = apply_transformation_equations(
#                 r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2], r_matrix[:, 3],
#                 r_matrix[:, 4], r_matrix[:, 5], r_matrix[:, 6], r_matrix[:, 7]
#             )
#             cx_phys = trans_contour_dict[px]
#             cy_phys = trans_contour_dict[py]
#             # calculo de errores geométricos
#             x_min_cont = np.min(cx_phys)
#             x_max_cont = np.max(cx_phys)
#             y_min_cont = np.min(cy_phys)
#             y_max_cont = np.max(cy_phys)
            
#             # Valores centrales
#             bf_x = best_fit_phys_dict[px]
#             bf_y = best_fit_phys_dict[py]
            
#             # Calculamos las distancias
#             err_x_up = x_max_cont - bf_x
#             err_x_down = bf_x - x_min_cont
#             err_y_up = y_max_cont - bf_y
#             err_y_down = bf_y - y_min_cont

#             def plot_physical(view_mode, x_lims=None, y_lims=None):
#                 plt.figure(figsize=(9, 8))
#                 alpha_cloud = 0.15 if view_mode == 'Full' else 0.05
                
#                 plt.scatter(df_phys_region[px], df_phys_region[py], c='gray', s=1, alpha=alpha_cloud, label='Allowed Region', zorder=0)
#                 plt.plot(cx_phys, cy_phys, 'b-', linewidth=2.5, label='1$\sigma$ Contour')
                
#                 # best Fit y valores verdades
#                 plt.plot(bf_x, bf_y, 'bo', markersize=6, label='Best Fit', zorder=10)
#                 plt.plot(true_dict[px], true_dict[py], 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=11)
                
#                 # limitess del contorno
#                 plt.axvline(x_min_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
#                 plt.axvline(x_max_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
#                 plt.axhline(y_min_cont, color='red', linestyle='--', linewidth=1, alpha=0.7)
#                 plt.axhline(y_max_cont, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Contour Limits')

#                 plt.xlabel(px, fontsize=14); plt.ylabel(py, fontsize=14)
                
#                 if view_mode == 'Zoom':
#                     # mostrando el error asimétrico calculado geométricamente
#                     title_str = (f"{px} = {bf_x:.4f} (+{err_x_up:.4f}/-{err_x_down:.4f})\n"
#                                  f"{py} = {bf_y:.4f} (+{err_y_up:.4f}/-{err_y_down:.4f})")
#                     plt.title(title_str, fontsize=11, color='blue')
                    
#                     if x_lims: plt.xlim(x_lims); 
#                     if y_lims: plt.ylim(y_lims)
#                 else:
#                     plt.title(f"FULL: {px} vs {py}", fontsize=12)
#                     plt.xlim(df_phys_region[px].min(), df_phys_region[px].max())
#                     plt.ylim(df_phys_region[py].min(), df_phys_region[py].max())

#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right', frameon=True, fontsize=9)
                
#                 folder = folder_phys_zoom if view_mode == 'Zoom' else folder_phys_full
#                 plt.savefig(f"{folder}/Phys_{px}_vs_{py}_{view_mode}.png", dpi=100)
#                 plt.close()

#             # Definir límites de zoom
#             mx = (x_max_cont - x_min_cont) * 0.4
#             my = (y_max_cont - y_min_cont) * 0.4
            
#             plot_physical('Zoom', x_lims=(x_min_cont-mx, x_max_cont+mx), y_lims=(y_min_cont-my, y_max_cont+my))
#             plot_physical('Full')

#         except Exception as e:
#             print(f"\n    [ERROR] Par {px}-{py}: {e}")
#             plt.close()

#     print(f"\n\n>>> ¡ANÁLISIS COMPLETADO!")

# if __name__ == "__main__":
#     run_multi_view_analysis()

















# import zfit
# import numpy as np
# import matplotlib.pyplot as plt
# from iminuit import Minuit
# import warnings
# import os
# from itertools import combinations
# import pandas as pd

# # --- TUS IMPORTS ---
# from PDFs import (
#     FullAngular_Physical_PDF, 
#     FullAngular_Transformed_PDF, 
#     get_inverse_values, 
#     apply_transformation_equations,
#     get_physical_region_scan
# )

# # Configuración
# zfit.settings.set_seed(42)
# np.random.seed(42)
# warnings.simplefilter('ignore')

# # --- NUEVA FUNCIÓN AUXILIAR (El Motor de la Espada) ---
# def calculate_jacobian_numerical(func, params, keys_out, epsilon=1e-5):
#     """
#     Calcula la matriz Jacobiana numéricamente.
#     Adapta la salida de diccionario de tu función original a un vector numpy.
#     """
#     n_params = len(params)
#     n_out = len(keys_out)
#     J = np.zeros((n_out, n_params))
    
#     # Wrapper para convertir dict -> array asegurando el orden
#     def func_wrapper(p_args):
#         res_dict = func(*p_args)
#         return np.array([res_dict[k] for k in keys_out])

#     for i in range(n_params):
#         p_plus = np.copy(params)
#         p_minus = np.copy(params)
#         p_plus[i] += epsilon
#         p_minus[i] -= epsilon
        
#         f_plus = func_wrapper(p_plus)
#         f_minus = func_wrapper(p_minus)
        
#         deriv = (f_plus - f_minus) / (2 * epsilon)
#         J[:, i] = deriv
#     return J

# def run_multi_view_analysis():
#     print(">>> 1. Iniciando Protocolo de Análisis Multi-Escala...")
    
#     # --- DEFINICIÓN DE CARPETAS ---
#     folder_trans = "Plots/Transformed_Space_v2"
#     folder_phys_zoom = "Plots/Physical_Space_Zoom_v2"
#     folder_phys_full = "Plots/Physical_Space_Full_v2"
    
#     for f in [folder_trans, folder_phys_zoom, folder_phys_full]:
#         os.makedirs(f, exist_ok=True)
    
#     print(f">>> Directorios listos:\n    - {folder_trans}\n    - {folder_phys_zoom}\n    - {folder_phys_full}")

#     # --- SETUP Y GENERACIÓN (JUGUETE) ---
#     obs = zfit.Space('cosThetaL', limits=(-1, 1)) * \
#           zfit.Space('cosThetaK', limits=(-1, 1)) * \
#           zfit.Space('phi', limits=(-np.pi, np.pi))

#     # Valores verdaderos físicos
#     true_vals_phys = [0.684, -0.013, 0.029, 0.050, -0.145, -0.136, -0.204, 0.077]
#     phys_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
#     true_dict = dict(zip(phys_keys, true_vals_phys))

#     print(">>> Generando datos (Toy MC)...")
#     pdf_gen = FullAngular_Physical_PDF(obs, *true_vals_phys)
#     sampler = pdf_gen.create_sampler(n=10000)
#     sampler.resample()

#     # --- AJUSTE (ESPACIO TRANSFORMADO) ---
#     raw_init = get_inverse_values(true_vals_phys)
#     r_keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
#     param_names_fit = [f"{k}_fit" for k in r_keys]
    
#     true_vals_trans_dict = dict(zip(param_names_fit, raw_init))
    
#     params = {k: zfit.Parameter(p_name, v, step_size=0.01) 
#               for k, p_name, v in zip(r_keys, param_names_fit, raw_init)}
    
#     pdf_fit = FullAngular_Transformed_PDF(
#         obs, 
#         params['rFL'], params['rS3'], params['rS9'], params['rAFB'],
#         params['rS4'], params['rS7'], params['rS5'], params['rS8']
#     )

#     print(">>> 2. Ejecutando Minimización (Minuit)...")
#     nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
#     minimizer = zfit.minimize.Minuit(tol=0.01) 
#     result = minimizer.minimize(nll)
#     m = result.info['minuit'] 
    
#     print(">>> 3. Calculando Errores MINOS...")
#     m.minos() 

# # ==============================================================================
#     # >>> 3.5 GUNDAM UPGRADE: PROPAGACIÓN DE ERRORES CON COVARIANZA (CORREGIDO) <<<
#     # ==============================================================================
#     print(">>> 3.5 Propagando errores vía Matriz de Covarianza...")
    
#     # 1. Obtenemos los OBJETOS de los parámetros en el orden correcto
#     #    NOTA: result.covariance necesita los objetos zfit.Parameter, no strings.
#     #    Usamos r_keys para asegurar que el orden sea exactamente: FL, S3, S9...
#     param_objs_ordered = [params[k] for k in r_keys]

#     # 2. Obtenemos la matriz de covarianza usando los OBJETOS
#     cov_trans = result.covariance(params=param_objs_ordered)
    
#     # 3. Obtenemos los valores centrales (Best Fit) para el Jacobiano
#     #    Extraemos el .value() de cada objeto
#     best_fit_r_values = np.array([p.value() for p in param_objs_ordered])
    
#     # 4. Calculamos Jacobiano (J)
#     #    Nota: r_keys tiene el orden de entrada, phys_keys el de salida
#     J = calculate_jacobian_numerical(apply_transformation_equations, best_fit_r_values, phys_keys)
    
#     # 5. Propagación matricial: V_phys = J * V_trans * J.T
#     cov_phys = J @ cov_trans @ J.T
    
#     # 6. Extraer errores (diagonal)
#     phys_errors_sigma = np.sqrt(np.diag(cov_phys))
#     phys_errors_dict = dict(zip(phys_keys, phys_errors_sigma))
    
#     print(f"    [INFO] Error propagado para FL: {phys_errors_dict['FL']:.5f}")
#     # ==============================================================================
#     # ==============================================================================

#     # --- GENERACIÓN DEL MAPA FÍSICO ---
#     print(">>> 4. Mapeando la Región Física (Background)...")
#     df_phys_region = get_physical_region_scan(n_points=100000)

#     # best_fit_r_values ya lo tenemos arriba como numpy array, pero tu código usa lista
#     best_fit_phys_dict = apply_transformation_equations(*best_fit_r_values)

#     # --- BUCLE DE PLOTEO ---
#     indices = list(range(8))
#     pairs_indices = list(combinations(indices, 2))
#     total_plots = len(pairs_indices)
    
#     print(f"\n>>> Generando {total_plots} combinaciones...\n")

#     for i, (idx_x, idx_y) in enumerate(pairs_indices):
        
#         rx, ry = param_names_fit[idx_x], param_names_fit[idx_y]
#         px, py = phys_keys[idx_x], phys_keys[idx_y]

#         print(f"    [{i+1}/{total_plots}] {px} vs {py} ...", end="\r")

#         try:
#             # 1. OBTENER CONTORNO (Espacio R)
#             contour_r = m.mncontour(rx, ry, cl=0.3935, size=50)

#             # ---------------------------------------------------------
#             # PLOT A: Espacio Transformado
#             # ---------------------------------------------------------
#             plt.figure(figsize=(8, 7))
#             plt.plot(contour_r[:, 0], contour_r[:, 1], 'b-', linewidth=2, label='1$\sigma$ Contour')
#             plt.plot(m.values[rx], m.values[ry], 'bo', label='Best Fit', zorder=10)
            
#             true_tx = true_vals_trans_dict[rx]
#             true_ty = true_vals_trans_dict[ry]
#             plt.plot(true_tx, true_ty, 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=15)

#             val_x = m.values[rx]
#             val_y = m.values[ry]
#             minos_x_low = val_x + m.merrors[rx].lower
#             minos_x_high = val_x + m.merrors[rx].upper
#             minos_y_low = val_y + m.merrors[ry].lower
#             minos_y_high = val_y + m.merrors[ry].upper
            
#             plt.axvline(minos_x_low, color='k', linestyle='--', alpha=0.4)
#             plt.axvline(minos_x_high, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_low, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_high, color='k', linestyle='--', alpha=0.4, label='MINOS Errors')
            
#             width_x = minos_x_high - minos_x_low
#             width_y = minos_y_high - minos_y_low
#             margin = 0.4
            
#             x_lim_low = minos_x_low - width_x * margin
#             x_lim_high = minos_x_high + width_x * margin
#             y_lim_low = minos_y_low - width_y * margin
#             y_lim_high = minos_y_high + width_y * margin

#             if true_tx < x_lim_low: x_lim_low = true_tx - width_x * 0.1
#             if true_tx > x_lim_high: x_lim_high = true_tx + width_x * 0.1
#             if true_ty < y_lim_low: y_lim_low = true_ty - width_y * 0.1
#             if true_ty > y_lim_high: y_lim_high = true_ty + width_y * 0.1

#             plt.xlim(x_lim_low, x_lim_high)
#             plt.ylim(y_lim_low, y_lim_high)

#             plt.xlabel(rx); plt.ylabel(ry)
#             plt.title(f"Transformed: {rx} vs {ry} (Fit vs Truth)")
#             plt.grid(True, alpha=0.3)
#             plt.legend()
#             plt.savefig(f"{folder_trans}/Trans_{rx}_vs_{ry}.png")
#             plt.close()

#             # ---------------------------------------------------------
#             # PLOTS B y C (FÍSICOS)
#             # ---------------------------------------------------------
#             n_pts = len(contour_r)
#             r_matrix = np.tile(best_fit_r_values, (n_pts, 1))
#             r_matrix[:, idx_x] = contour_r[:, 0]
#             r_matrix[:, idx_y] = contour_r[:, 1]
            
#             # Nota: Usamos apply_transformation_equations pero debemos extraer los arrays del dict resultante
#             trans_contour_dict = apply_transformation_equations(
#                 r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2], r_matrix[:, 3],
#                 r_matrix[:, 4], r_matrix[:, 5], r_matrix[:, 6], r_matrix[:, 7]
#             )
#             # Extraemos los canales específicos
#             cx_phys = trans_contour_dict[px]
#             cy_phys = trans_contour_dict[py]

#             def plot_physical(view_mode, x_lims=None, y_lims=None):
#                 plt.figure(figsize=(9, 8))
#                 alpha_cloud = 0.15 if view_mode == 'Full' else 0.05
#                 plt.scatter(df_phys_region[px], df_phys_region[py], 
#                            c='gray', s=1, alpha=alpha_cloud, label='Allowed Region', zorder=0)
#                 plt.plot(cx_phys, cy_phys, 'b-', linewidth=2.5, label='1$\sigma$ Contour')
#                 plt.plot(best_fit_phys_dict[px], best_fit_phys_dict[py], 
#                         'bo', markersize=6, label='Best Fit', zorder=10)
#                 plt.plot(true_dict[px], true_dict[py], 
#                         'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=11)
                
#                 # ---> NUEVO: DIBUJAR LA ESPADA (ERROR DE COVARIANZA) EN EL PLOT <---
#                 # Obtenemos el error calculado en el bloque 3.5
#                 sigma_x = phys_errors_dict[px]
#                 sigma_y = phys_errors_dict[py]
#                 plt.errorbar(best_fit_phys_dict[px], best_fit_phys_dict[py], 
#                              xerr=sigma_x, yerr=sigma_y, 
#                              fmt='none', ecolor='cyan', elinewidth=2, capsize=4, 
#                              label=r'Cov. Error ($\pm 1\sigma$)', zorder=12)
#                 # -------------------------------------------------------------------

#                 plt.xlabel(px, fontsize=12); plt.ylabel(py, fontsize=12)
                
#                 if view_mode == 'Zoom':
#                     # Agregamos los errores numéricos al título
#                     plt.title(f"ZOOM: {px}($\pm${sigma_x:.3f}) vs {py}($\pm${sigma_y:.3f})", fontsize=10)
#                     if x_lims: plt.xlim(x_lims); 
#                     if y_lims: plt.ylim(y_lims)
#                 else:
#                     plt.title(f"FULL: {px} vs {py}", fontsize=10)
#                     plt.xlim(df_phys_region[px].min(), df_phys_region[px].max())
#                     plt.ylim(df_phys_region[py].min(), df_phys_region[py].max())

#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right', frameon=True, fontsize=9)
                
#                 folder = folder_phys_zoom if view_mode == 'Zoom' else folder_phys_full
#                 plt.savefig(f"{folder}/Phys_{px}_vs_{py}_{view_mode}.png", dpi=100)
#                 plt.close()

#             xmin, xmax = cx_phys.min(), cx_phys.max()
#             ymin, ymax = cy_phys.min(), cy_phys.max()
#             mx = (xmax - xmin) * 0.3 if (xmax-xmin)>0 else 0.05
#             my = (ymax - ymin) * 0.3 if (ymax-ymin)>0 else 0.05
#             plot_physical('Zoom', x_lims=(xmin-mx, xmax+mx), y_lims=(ymin-my, ymax+my))
#             plot_physical('Full')

#         except Exception as e:
#             print(f"\n    [ERROR] Par {px}-{py}: {e}")
#             plt.close()

#     print(f"\n\n>>> ¡ANÁLISIS COMPLETADO!")

# if __name__ == "__main__":
#     run_multi_view_analysis()















# import zfit
# import numpy as np
# import matplotlib.pyplot as plt
# from iminuit import Minuit
# import warnings
# import os
# from itertools import combinations
# import pandas as pd

# # --- TUS IMPORTS ---
# from PDFs import (
#     FullAngular_Physical_PDF, 
#     FullAngular_Transformed_PDF, 
#     get_inverse_values, 
#     apply_transformation_equations,
#     get_physical_region_scan
# )

# # Configuración
# zfit.settings.set_seed(42)
# np.random.seed(42)
# warnings.simplefilter('ignore')

# def run_multi_view_analysis():
#     print(">>> 1. Iniciando Protocolo de Análisis Multi-Escala...")
    
#     # --- DEFINICIÓN DE CARPETAS ---
#     folder_trans = "Plots/Transformed_Space"
#     folder_phys_zoom = "Plots/Physical_Space_Zoom"
#     folder_phys_full = "Plots/Physical_Space_Full"
    
#     for f in [folder_trans, folder_phys_zoom, folder_phys_full]:
#         os.makedirs(f, exist_ok=True)
    
#     print(f">>> Directorios listos:\n    - {folder_trans}\n    - {folder_phys_zoom}\n    - {folder_phys_full}")

#     # --- SETUP Y GENERACIÓN (JUGUETE) ---
#     obs = zfit.Space('cosThetaL', limits=(-1, 1)) * \
#           zfit.Space('cosThetaK', limits=(-1, 1)) * \
#           zfit.Space('phi', limits=(-np.pi, np.pi))

#     # Valores verdaderos físicos
#     true_vals_phys = [0.684, -0.013, 0.029, 0.050, -0.145, -0.136, -0.204, 0.077]
#     phys_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
#     true_dict = dict(zip(phys_keys, true_vals_phys))

#     print(">>> Generando datos (Toy MC)...")
#     pdf_gen = FullAngular_Physical_PDF(obs, *true_vals_phys)
#     sampler = pdf_gen.create_sampler(n=10000)
#     sampler.resample()

#     # --- AJUSTE (ESPACIO TRANSFORMADO) ---
#     # raw_init contiene la VERDAD transformada (inversa)
#     raw_init = get_inverse_values(true_vals_phys)
#     r_keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
#     param_names_fit = [f"{k}_fit" for k in r_keys]
    
#     # Diccionario maestro de Verdad en Espacio Transformado
#     true_vals_trans_dict = dict(zip(param_names_fit, raw_init))
    
#     params = {k: zfit.Parameter(p_name, v, step_size=0.01) 
#               for k, p_name, v in zip(r_keys, param_names_fit, raw_init)}
    
#     pdf_fit = FullAngular_Transformed_PDF(
#         obs, 
#         params['rFL'], params['rS3'], params['rS9'], params['rAFB'],
#         params['rS4'], params['rS7'], params['rS5'], params['rS8']
#     )

#     print(">>> 2. Ejecutando Minimización (Minuit)...")
#     nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
#     minimizer = zfit.minimize.Minuit(tol=0.01) 
#     result = minimizer.minimize(nll)
#     m = result.info['minuit'] 
    
#     print(">>> 3. Calculando Errores MINOS...")
#     m.minos() 

#     # --- GENERACIÓN DEL MAPA FÍSICO ---
#     print(">>> 4. Mapeando la Región Física (Background)...")
#     df_phys_region = get_physical_region_scan(n_points=100000)

#     best_fit_r_values = [m.values[p] for p in param_names_fit]
#     best_fit_phys_dict = apply_transformation_equations(*best_fit_r_values)

#     # --- BUCLE DE PLOTEO ---
#     indices = list(range(8))
#     pairs_indices = list(combinations(indices, 2))
#     total_plots = len(pairs_indices)
    
#     print(f"\n>>> Generando {total_plots} combinaciones...\n")

#     for i, (idx_x, idx_y) in enumerate(pairs_indices):
        
#         rx, ry = param_names_fit[idx_x], param_names_fit[idx_y]
#         px, py = phys_keys[idx_x], phys_keys[idx_y]

#         print(f"    [{i+1}/{total_plots}] {px} vs {py} ...", end="\r")

#         try:
#             # 1. OBTENER CONTORNO (Espacio R)
#             contour_r = m.mncontour(rx, ry, cl=0.3935, size=50)

#             # ---------------------------------------------------------
#             # PLOT A: Espacio Transformado (Fit + MINOS + TRUE)
#             # ---------------------------------------------------------
#             plt.figure(figsize=(8, 7))
            
#             # Contorno
#             plt.plot(contour_r[:, 0], contour_r[:, 1], 'b-', linewidth=2, label='1$\sigma$ Contour')
#             # Best Fit
#             plt.plot(m.values[rx], m.values[ry], 'bo', label='Best Fit', zorder=10)
            
#             # ---> NUEVO: True Value en Espacio Transformado
#             true_tx = true_vals_trans_dict[rx]
#             true_ty = true_vals_trans_dict[ry]
#             plt.plot(true_tx, true_ty, 'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=15)

#             # MINOS Errors
#             val_x = m.values[rx]
#             val_y = m.values[ry]
#             minos_x_low = val_x + m.merrors[rx].lower
#             minos_x_high = val_x + m.merrors[rx].upper
#             minos_y_low = val_y + m.merrors[ry].lower
#             minos_y_high = val_y + m.merrors[ry].upper
            
#             plt.axvline(minos_x_low, color='k', linestyle='--', alpha=0.4)
#             plt.axvline(minos_x_high, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_low, color='k', linestyle='--', alpha=0.4)
#             plt.axhline(minos_y_high, color='k', linestyle='--', alpha=0.4, label='MINOS Errors')
            
#             # Ajuste de Zoom Dinámico (Considerando también el True Value para que no quede fuera)
#             width_x = minos_x_high - minos_x_low
#             width_y = minos_y_high - minos_y_low
#             margin = 0.4 # Un poco más de margen para asegurar que quepa todo
            
#             # Definimos límites tentativos basados en MINOS
#             x_lim_low = minos_x_low - width_x * margin
#             x_lim_high = minos_x_high + width_x * margin
#             y_lim_low = minos_y_low - width_y * margin
#             y_lim_high = minos_y_high + width_y * margin

#             # Expandimos si el True Value cae fuera
#             if true_tx < x_lim_low: x_lim_low = true_tx - width_x * 0.1
#             if true_tx > x_lim_high: x_lim_high = true_tx + width_x * 0.1
#             if true_ty < y_lim_low: y_lim_low = true_ty - width_y * 0.1
#             if true_ty > y_lim_high: y_lim_high = true_ty + width_y * 0.1

#             plt.xlim(x_lim_low, x_lim_high)
#             plt.ylim(y_lim_low, y_lim_high)

#             plt.xlabel(rx); plt.ylabel(ry)
#             plt.title(f"Transformed: {rx} vs {ry} (Fit vs Truth)")
#             plt.grid(True, alpha=0.3)
#             plt.legend()
#             plt.savefig(f"{folder_trans}/Trans_{rx}_vs_{ry}.png")
#             plt.close()

#             # ---------------------------------------------------------
#             # PLOTS B y C (FÍSICOS) - Sin cambios, se mantienen igual
#             # ---------------------------------------------------------
#             n_pts = len(contour_r)
#             r_matrix = np.tile(best_fit_r_values, (n_pts, 1))
#             r_matrix[:, idx_x] = contour_r[:, 0]
#             r_matrix[:, idx_y] = contour_r[:, 1]
            
#             trans_contour = apply_transformation_equations(
#                 r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2], r_matrix[:, 3],
#                 r_matrix[:, 4], r_matrix[:, 5], r_matrix[:, 6], r_matrix[:, 7]
#             )
#             cx_phys = trans_contour[px]
#             cy_phys = trans_contour[py]

#             def plot_physical(view_mode, x_lims=None, y_lims=None):
#                 plt.figure(figsize=(9, 8))
#                 alpha_cloud = 0.15 if view_mode == 'Full' else 0.05
#                 plt.scatter(df_phys_region[px], df_phys_region[py], 
#                            c='gray', s=1, alpha=alpha_cloud, label='Allowed Region', zorder=0)
#                 plt.plot(cx_phys, cy_phys, 'b-', linewidth=2.5, label='1$\sigma$ CL')
#                 plt.plot(best_fit_phys_dict[px], best_fit_phys_dict[py], 
#                         'bo', markersize=6, label='Best Fit', zorder=10)
#                 plt.plot(true_dict[px], true_dict[py], 
#                         'r*', markersize=14, markeredgecolor='k', label='True Value', zorder=11)
                
#                 plt.xlabel(px, fontsize=12); plt.ylabel(py, fontsize=12)
                
#                 if view_mode == 'Zoom':
#                     plt.title(f"ZOOM: {px} vs {py}", fontsize=10)
#                     if x_lims: plt.xlim(x_lims); 
#                     if y_lims: plt.ylim(y_lims)
#                 else:
#                     plt.title(f"FULL: {px} vs {py}", fontsize=10)
#                     plt.xlim(df_phys_region[px].min(), df_phys_region[px].max())
#                     plt.ylim(df_phys_region[py].min(), df_phys_region[py].max())

#                 plt.grid(True, alpha=0.3)
#                 plt.legend(loc='upper right', frameon=True, fontsize=9)
                
#                 folder = folder_phys_zoom if view_mode == 'Zoom' else folder_phys_full
#                 plt.savefig(f"{folder}/Phys_{px}_vs_{py}_{view_mode}.png", dpi=100)
#                 plt.close()

#             xmin, xmax = cx_phys.min(), cx_phys.max()
#             ymin, ymax = cy_phys.min(), cy_phys.max()
#             mx = (xmax - xmin) * 0.3 if (xmax-xmin)>0 else 0.05
#             my = (ymax - ymin) * 0.3 if (ymax-ymin)>0 else 0.05
#             plot_physical('Zoom', x_lims=(xmin-mx, xmax+mx), y_lims=(ymin-my, ymax+my))
#             plot_physical('Full')

#         except Exception as e:
#             print(f"\n    [ERROR] Par {px}-{py}: {e}")
#             plt.close()

#     print(f"\n\n>>> ¡ANÁLISIS COMPLETADO!")

# if __name__ == "__main__":
#     run_multi_view_analysis()






































# import zfit
# import numpy as np
# import matplotlib.pyplot as plt
# from iminuit import Minuit
# import warnings
# import os # Para gestionar carpetas
# from itertools import combinations # La clave para hacer todos los pares

# # --- TUS IMPORTS ---
# from PDFs import FullAngular_Physical_PDF, FullAngular_Transformed_PDF, get_inverse_values

# # Configuración
# zfit.settings.set_seed(42)
# np.random.seed(42)
# warnings.simplefilter('ignore')

# def run_massive_comparison():
#     print(">>> 1. Iniciando Protocolo de Generación Masiva...")
    
#     # --- SETUP (Igual que siempre) ---
#     obs = zfit.Space('cosThetaL', limits=(-1, 1)) * \
#           zfit.Space('cosThetaK', limits=(-1, 1)) * \
#           zfit.Space('phi', limits=(-np.pi, np.pi))

#     true_vals = [0.684, -0.013, 0.029, 0.050, -0.145, -0.136, -0.204, 0.077]
#     pdf_gen = FullAngular_Physical_PDF(obs, *true_vals)
#     sampler = pdf_gen.create_sampler(n=2000)
#     sampler.resample()

#     raw_init = get_inverse_values(true_vals)
#     keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
    
#     # Nombres reales de los parámetros en el Minuit (con el sufijo _fit)
#     param_names = [f"{k}_fit" for k in keys]
    
#     params = {k: zfit.Parameter(p_name, v, step_size=0.01) for k, p_name, v in zip(keys, param_names, raw_init)}
    
#     pdf_fit = FullAngular_Transformed_PDF(
#         obs, 
#         params['rFL'], params['rS3'], params['rS9'], params['rAFB'],
#         params['rS4'], params['rS7'], params['rS5'], params['rS8']
#     )

#     print(">>> 2. Ajustando (Fit)...")
#     nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
#     minimizer = zfit.minimize.Minuit(tol=0.01) 
#     result = minimizer.minimize(nll)
    
#     m = result.info['minuit'] 
    
#     print(">>> 3. Calculando Errores MINOS (Base de Verdad)...")
#     m.minos() 

#     # --- PREPARACIÓN DEL LOOP ---
    
#     # Crear carpeta para guardar los plots
#     output_folder = "Plots/Comparative_Contours"
#     os.makedirs(output_folder, exist_ok=True)
#     print(f"\n>>> Los gráficos se guardarán en la carpeta: ./{output_folder}/")
    
#     # Generar todos los pares posibles (8 parámetros tomados de 2 en 2 = 28 combinaciones)
#     pairs = list(combinations(param_names, 2))
#     total_plots = len(pairs)
    
#     print(f">>> Se generarán {total_plots} gráficos comparativos. Esto puede tardar unos minutos...\n")

#     # --- BUCLE DE GENERACIÓN MASIVA ---
#     for i, (p_x, p_y) in enumerate(pairs):
        
#         print(f"    [{i+1}/{total_plots}] Procesando: {p_x} vs {p_y} ...", end="\r")
        
#         try:
#             # 1. Contorno "Incorrecto" (Heriberto)
#             pts_tiny = m.mncontour(p_x, p_y, cl=0.017, size=30) # size=30 para ir más rápido
            
#             # 2. Contorno "Correcto" (Validación)
#             pts_valid = m.mncontour(p_x, p_y, cl=0.3935, size=30)
            
#             # --- GRAFICADO ---
#             plt.figure(figsize=(9, 8))
            
#             # Azul (Correcto)
#             plt.plot(pts_valid[:, 0], pts_valid[:, 1], 'b-', linewidth=2, 
#                      label='($\Delta\chi^2=1$, cl=0.39)')
            
#             # Rojo (Incorrecto)
#             plt.plot(pts_tiny[:, 0], pts_tiny[:, 1], 'r-', linewidth=2, 
#                      label='(cl=0.017)')
            
#             # Best Fit
#             plt.plot(m.values[p_x], m.values[p_y], 'ko', zorder=10)
            
#             # Límites MINOS (La Verdad Numérica)
#             minos_x_low = m.values[p_x] + m.merrors[p_x].lower
#             minos_x_high = m.values[p_x] + m.merrors[p_x].upper
#             minos_y_low = m.values[p_y] + m.merrors[p_y].lower
#             minos_y_high = m.values[p_y] + m.merrors[p_y].upper
            
#             plt.axvline(minos_x_low, color='k', linestyle='--', alpha=0.3)
#             plt.axvline(minos_x_high, color='k', linestyle='--', alpha=0.3)
#             plt.axhline(minos_y_low, color='k', linestyle='--', alpha=0.3)
#             plt.axhline(minos_y_high, color='k', linestyle='--', alpha=0.3, label='errors MINOS')
            
#             # Ajuste de Zoom (Importante para que no se vea minúsculo)
#             margin_x = (minos_x_high - minos_x_low) * 0.25
#             margin_y = (minos_y_high - minos_y_low) * 0.25
#             plt.xlim(minos_x_low - margin_x, minos_x_high + margin_x)
#             plt.ylim(minos_y_low - margin_y, minos_y_high + margin_y)

#             # Textos
#             plt.xlabel(p_x)
#             plt.ylabel(p_y)
#             plt.title(f"{p_x} vs {p_y}")
#             # Leyenda solo en el primer plot para no tapar, o en 'best'
#             plt.legend(loc='best', fontsize=9)
#             plt.grid(True, alpha=0.3)
            
#             # Guardar
#             filename = f"{output_folder}/Comp_{p_x}_vs_{p_y}.png"
#             plt.savefig(filename)
#             plt.close() # CRUCIAL: Cerrar la figura para no saturar la memoria RAM
            
#         except Exception as e:
#             print(f"\n    [ERROR] Falló el par {p_x} vs {p_y}: {e}")
#             plt.close()

#     print(f"\n\n>>> ¡PROCESO TERMINADO! Revisa la carpeta '{output_folder}'.")

# if __name__ == "__main__":
#     run_massive_comparison()

    