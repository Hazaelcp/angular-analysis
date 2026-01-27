import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools
import tensorflow as tf
import zfit
from iminuit import Minuit
from PDFs import FullAngular_Physical_PDF, FullAngular_Transformed_PDF, get_inverse_values, apply_transformation_equations, get_physical_region_scan

# --- Configuración Estética ---
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 2
})
# ==========================================
# 2. PUENTE ZFIT -> MINUIT
# ==========================================
def zfit_to_minuit_bridge(nll, zfit_result):
    """
    Convierte NLL de Zfit a Minuit vivo, con manejo de errores para NaNs.
    Esto evita que el scan de contornos rompa el programa si toca regiones prohibidas.
    """
    params = nll.get_params()
    param_names = [p.name for p in params]
    # Convertimos a float explícitamente para evitar tipos de numpy/tf en Minuit
    start_values = {p.name: float(zfit_result.params[p]['value']) for p in params}
    
    def cost_func(*args):
        # 1. Asignar valores a los parámetros
        for p, val in zip(params, args):
            p.set_value(val)
        
        try:
            # 2. Calcular NLL
            # Usamos nll.value() de zfit. 
            # Es vital capturar excepciones de TF (InvalidArgumentError) que lanza CheckNumerics
            val_tf = nll.value()
            val = val_tf.numpy() # Convertir a numpy
            
            # 3. Verificar NaNs o Infinitos matemáticos
            if np.isnan(val) or np.isinf(val):
                return 1e18  # Penalización gigante (Soft Wall)
            
            return val

        except (tf.errors.InvalidArgumentError, tf.errors.OpError):
            # 4. Capturar errores de Grafo (CheckNumerics fallando)
            return 1e18 # Penalización gigante
            
        except Exception as e:
            # Captura genérica por seguridad
            return 1e18 
    
    # Inicializamos Minuit con la función blindada
    m = Minuit(cost_func, name=param_names, **start_values)
    m.errordef = 0.5 # Likelihood
    return m

# ==========================================
# 3. LÓGICA DE GRAFICADO (TRANSFORMANDO CONTORNOS)
# ==========================================
def plot_transformed_contours(minuit_obj, best_fit_r_values, output_dir="Plots/IntegratedStudy"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print(">>> Generando fondo de región física...")
    df_phys = get_physical_region_scan(n_points=200000)
    
    # Definimos los pares FÍSICOS que queremos graficar
    # Nota: El fit se hizo en 'rFL', pero graficamos 'FL'.
    # Necesitamos saber qué r-parametro corresponde a qué fisico
    param_map = {
        'FL': 'rFL', 'S3': 'rS3', 'S9': 'rS9', 'AFB': 'rAFB',
        'S4': 'rS4', 'S7': 'rS7', 'S5': 'rS5', 'S8': 'rS8'
    }
    phys_keys = list(param_map.keys())
    pairs = list(itertools.combinations(phys_keys, 2))
    
    # Obtenemos los valores 'r' del Best Fit para usarlos como base fija
    # cuando escaneamos otros pares
    r_best = best_fit_r_values.copy() 

    print(f">>> Generando {len(pairs)} gráficos de contorno...")

    for i, (p1_phys, p2_phys) in enumerate(pairs):
        p1_r = param_map[p1_phys] # ej: 'rFL'
        p2_r = param_map[p2_phys] # ej: 'rS3'
        
        print(f"  Plotting {p1_phys} vs {p2_phys} (Escaneando {p1_r} vs {p2_r})...")
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # A. Fondo Físico
        ax.plot(df_phys[p1_phys], df_phys[p2_phys], '.', color='#e0e0e0', markersize=1, zorder=0)

        # B. Calcular Contornos en espacio R usando Minuit
        try:
            # mncontour devuelve puntos en el espacio R (ej: rFL vs rS3)
            # Necesitamos transformarlos al espacio físico
            
            contours_to_plot = []
            styles = [('--', 'skyblue', r'Minuit $2\sigma$'), ('-', 'blue', r'Minuit $1\sigma$')]
            confidence_levels = [0.9545, 0.6827] # 2 sigma, 1 sigma

            for cl, (ls, col, lab) in zip(confidence_levels, styles):
                # 1. Obtener contorno en R
                ctr_r = minuit_obj.mncontour(p1_r, p2_r, cl=cl, size=50) 
                pts_r = np.array(ctr_r)

                # --- AGREGAR ESTA VALIDACIÓN ---
                if pts_r.ndim != 2 or pts_r.shape[0] < 3:
                    print(f"    Warning: Minuit no pudo cerrar el contorno {cl}sigma para {p1_phys}-{p2_phys}.")
                    continue # Salta este contorno y ve al siguiente
                # -------------------------------
                # 2. Transformar estos puntos a Físico
                # Para transformar, necesitamos los valores de los OTROS 6 parámetros fijos en su BestFit
                # Creamos un diccionario con todos los parametros en BestFit
                current_r_dict = {k: np.full(len(pts_r), v) for k, v in r_best.items()}
                
                # Sobrescribimos los que están variando en el contorno
                current_r_dict[p1_r] = pts_r[:, 0]
                current_r_dict[p2_r] = pts_r[:, 1]
                
                # Aplicamos la transformación matemática a estos puntos
                # OJO: Los nombres de entrada de apply_transformation deben coincidir con las keys
                # Como usas 'rFL_fit' o similar en zfit, quizás debamos limpiar nombres.
                # ASUNCIÓN: apply_transformation espera argumentos posicionales o keys limpias 'rFL'.
                
                # Preparamos argumentos para la función de transformación
                args_trans = {
                    'rFL': current_r_dict[param_map['FL']],
                    'rS3': current_r_dict[param_map['S3']],
                    'rS9': current_r_dict[param_map['S9']],
                    'rAFB': current_r_dict[param_map['AFB']],
                    'rS4': current_r_dict[param_map['S4']],
                    'rS7': current_r_dict[param_map['S7']],
                    'rS5': current_r_dict[param_map['S5']],
                    'rS8': current_r_dict[param_map['S8']]
                }
                
                phys_out = apply_transformation_equations(**args_trans)
                
                # Extraemos las coordenadas transformadas
                x_phys = phys_out[p1_phys]
                y_phys = phys_out[p2_phys]
                
                # Cerramos el polígono
                x_phys = np.append(x_phys, x_phys[0])
                y_phys = np.append(y_phys, y_phys[0])
                
                ax.plot(x_phys, y_phys, linestyle=ls, color=col, linewidth=2, label=lab)

        except Exception as e:
            print(f"    Warning: Falló mncontour para {p1_phys}-{p2_phys}: {e}")

        # C. Best Fit Point (Transformado)
        # Calculamos el punto central físico
        args_best = {k: r_best[param_map[k_phys]] for k_phys, k in param_map.items()} # map keys correctly if needed
        # Simplificación: Pasamos los valores r_best a la transformación
        args_best_clean = {
            'rFL': r_best[param_map['FL']], 'rS3': r_best[param_map['S3']],
            'rS9': r_best[param_map['S9']], 'rAFB': r_best[param_map['AFB']],
            'rS4': r_best[param_map['S4']], 'rS7': r_best[param_map['S7']],
            'rS5': r_best[param_map['S5']], 'rS8': r_best[param_map['S8']]
        }
        phys_best = apply_transformation_equations(**args_best_clean)
        
        ax.plot(phys_best[p1_phys], phys_best[p2_phys], 'rX', markersize=10, markeredgecolor='white', label='Best Fit', zorder=10)

        # Estética
        ax.set_xlabel(f'{p1_phys} (Physical)')
        ax.set_ylabel(f'{p2_phys} (Physical)')
        ax.set_title(f'Contours: {p1_phys} vs {p2_phys}')
        
        margin_x = (df_phys[p1_phys].max() - df_phys[p1_phys].min()) * 0.1
        margin_y = (df_phys[p2_phys].max() - df_phys[p2_phys].min()) * 0.1
        ax.set_xlim(df_phys[p1_phys].min() - margin_x, df_phys[p1_phys].max() + margin_x)
        ax.set_ylim(df_phys[p2_phys].min() - margin_y, df_phys[p2_phys].max() + margin_y)
        
        if i == 0: ax.legend(loc='upper right')
        plt.savefig(f"{output_dir}/Contour_{p1_phys}_vs_{p2_phys}.png")
        plt.close()

# ==========================================
# 4. MAIN EJECUTABLE
# ==========================================
if __name__ == "__main__":
    print("--- INICIANDO STUDIO INTEGRADO (TOY + FIT + CONTOURS) ---")
    
    # A. CONFIGURACIÓN DEL ESPACIO Y DATOS
    obs = zfit.Space('cosThetaL', limits=(-1, 1)) * zfit.Space('cosThetaK', limits=(-1, 1)) * zfit.Space('phi', limits=(-np.pi, np.pi))
    
    # Valores Verdaderos (LHCb / SM)
    true_vals = {'FL': 0.684, 'S3': -0.013, 'S9': 0.029, 'AFB': 0.050, 'S4': -0.145, 'S7': -0.136, 'S5': -0.204, 'S8': 0.077}
    ordered_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
    
    print("1. Generando Toy Monte Carlo...")
    # PDF de Generación (Física)
    params_gen = {k: zfit.Parameter(f"{k}_gen", v, floating=False) for k, v in true_vals.items()}
    pdf_gen = FullAngular_Physical_PDF(obs, **params_gen)
    
    # Generar datos
    sampler = pdf_gen.create_sampler(n=2000) # 2000 eventos
    sampler.resample()
    
    print("2. Configurando el Ajuste...")
    # Valores iniciales transformados (r-space)
    true_list_vals = [true_vals[k] for k in ordered_keys]
    raw_init = get_inverse_values(true_list_vals) # Asumo que tienes esta función importada
    
    # Parámetros flotantes para el ajuste
    # Nota: Usamos nombres estándar rFL, rS3 para facilitar el mapeo luego
    params_fit = {
        f"r{k}": zfit.Parameter(f"r{k}", init, step_size=0.1) 
        for k, init in zip(ordered_keys, raw_init)
    }
    
    # PDF de Ajuste (Transformada)
    pdf_fit = FullAngular_Transformed_PDF(
        obs, 
        params_fit['rFL'], params_fit['rS3'], params_fit['rS9'], params_fit['rAFB'],
        params_fit['rS4'], params_fit['rS7'], params_fit['rS5'], params_fit['rS8']
    )
    
    print("3. Ejecutando Minimitación (Migrad)...")
    nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    
    if result.valid:
        print("   ¡Convergencia Exitosa!")
        print(result.params)
        
        # Guardar los valores "r" del best fit para pasarlos al ploteador
        # (Necesitamos diccionarios limpios: {'rFL': 0.123, ...})
        best_fit_r = {p.name: float(res['value']) for p, res in result.params.items()}
        
        print("\n4. Generando Contornos Físicos...")
        # A. Crear Puente
        minuit_bridge = zfit_to_minuit_bridge(nll, result)
        
        # B. Graficar transformando al vuelo
        plot_transformed_contours(minuit_bridge, best_fit_r, output_dir="Plots/IntegratedRun")
        
        print("\n--- PROCESO COMPLETADO. Revisa 'Plots/IntegratedRun' ---")
        
    else:
        print("EL AJUSTE FALLÓ. No se generarán contornos.")