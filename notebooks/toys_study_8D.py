import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zfit
from zfit import z
import pandas as pd
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os 
from PDFs import FullAngular_Physical_PDF, FullAngular_Transformed_PDF, get_inverse_values


def transform_raw_to_phys_with_errors(raw_values, raw_covariance, pdf_instance):
    param_keys = ['rFL', 'rS3', 'rS9', 'rAFB', 'rS4', 'rS7', 'rS5', 'rS8']
    zfit_params = [pdf_instance.params[k] for k in param_keys]

    for p, val in zip(zfit_params, raw_values):
        p.set_value(val)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(zfit_params)
        phys_out = pdf_instance._get_physical_coeffs()
    
    jacobian = []
    for p_out in phys_out:
        grad = tape.gradient(p_out, zfit_params)
        row = [g.numpy() if g is not None else 0.0 for g in grad]
        jacobian.append(row)
    del tape

    J = np.array(jacobian) 
    cov_phys = J @ raw_covariance @ J.T
    phys_vals = [p.numpy() for p in phys_out]    
    phys_errors = np.sqrt(np.maximum(np.diag(cov_phys), 1e-18))
    return phys_vals, phys_errors, cov_phys


# ==================
# MAIN TOY STUDY
# ==================

def run_toy_study(n_toys=100, n_events=1000):
    print(f"--- Iniciando Estudio de Monte Carlo: {n_toys} toys de {n_events} eventos ---")
    
    if not os.path.exists("Plots/Pulls"): os.makedirs("Plots/Pulls")
    if not os.path.exists("Plots/Values"): os.makedirs("Plots/Values")

    # espacio observable
    obs = zfit.Space('cosThetaL', limits=(-1, 1)) * zfit.Space('cosThetaK', limits=(-1, 1)) * zfit.Space('phi', limits=(-np.pi, np.pi))

    # # Valores Verdaderos SM
    # true_vals = {'FL': 0.70, 'S3': -0.10, 'S9': 0.00, 'AFB': 0.10,'S4': 0.15, 'S7': 0.02, 'S5': 0.20, 'S8': -0.05}
    # ordered_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
    # true_list = [true_vals[k] for k in ordered_keys]

    # Valores inicialesLHCb Collaboration)
    true_vals = {'FL':  0.684,  'S3': -0.013,  'S9':  0.029,  'AFB': 0.050, 'S4': -0.145,  'S7': -0.136, 'S5': -0.204, 'S8':  0.077 }
    ordered_keys = ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']
    true_list = [true_vals[k] for k in ordered_keys]

    # Generación
    params_gen = {k: zfit.Parameter(f"{k}_gen", v, floating=False) for k, v in true_vals.items()}
    pdf_gen = FullAngular_Physical_PDF(obs, **params_gen)

    # Ajuste
    raw_init = get_inverse_values(true_list)
    params_fit = {
        f"r{k}": zfit.Parameter(f"r{k}_fit", init, step_size=0.1) 
        for k, init in zip(ordered_keys, raw_init)
    }

    pdf_fit = FullAngular_Transformed_PDF(
        obs, 
        params_fit['rFL'], params_fit['rS3'], params_fit['rS9'], params_fit['rAFB'],
        params_fit['rS4'], params_fit['rS7'], params_fit['rS5'], params_fit['rS8']
    )

    results = {k: [] for k in ordered_keys}
    pulls = {k: [] for k in ordered_keys}
    
    # Bucle de Toys
    for i in range(n_toys):
        if i % 10 == 0: print(f"Procesando Toy {i}/{n_toys}...")
        try:
            sampler = pdf_gen.create_sampler(n=n_events)
            sampler.resample()
            
            # Reset valores iniciales
            for idx, k in enumerate(ordered_keys):
                params_fit[f"r{k}"].set_value(raw_init[idx]) 
            
            nll = zfit.loss.UnbinnedNLL(model=pdf_fit, data=sampler)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(nll)
            result.hesse()
            
            raw_vals_fit = [params_fit[f"r{k}"].value().numpy() for k in ordered_keys]
            cov_raw = result.covariance(params=[params_fit[f"r{k}"] for k in ordered_keys])
            cov_raw = np.array(cov_raw)
            phys_fit, phys_err, phys_cov_matrix = transform_raw_to_phys_with_errors(raw_vals_fit, cov_raw, pdf_fit)
            
            for idx, key in enumerate(ordered_keys):
                val = phys_fit[idx]
                err = phys_err[idx]
                truth = true_list[idx]
                
                if err > 1e-9:
                    pull = (val - truth) / err
                    results[key].append(val)
                    pulls[key].append(pull) 
            # GUARDAR RESULTADOS DEL ÚLTIMO TOY
            np.savez("last_fit_results.npz", 
                    values=phys_fit, 
                    errors=phys_err,
                    covariance=phys_cov_matrix, 
                    keys=ordered_keys)
        except Exception as e:
            print(f"Fallo en Toy {i}: {e}")
            continue

    # ==========================
    # PLOTTING PULLS
    # ==========================
    print("\nGenerando gráficas individuales de Pulls...")
    
    for key in ordered_keys:
        data = np.array(pulls[key])
        data = data[np.abs(data) < 5]
        
        plt.figure(figsize=(8, 6))
        
        # histograma
        n, bins, patches = plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Toys Pulls')
        
        # ajustegaussiano
        if len(data) > 1:
            mu, std = norm.fit(data)            
            N = len(data)
            mu_err = std / np.sqrt(N)
            std_err = std / np.sqrt(2 * N)
            x_plot = np.linspace(min(data), max(data), 100)
            p = norm.pdf(x_plot, mu, std)
            
            label_fit = (f'Gauss Fit:\n'
                         f'$\mu = {mu:.3f} \pm {mu_err:.3f}$\n'
                         f'$\sigma = {std:.3f} \pm {std_err:.3f}$')
            
            plt.plot(x_plot, p, 'r-', linewidth=2, label=label_fit)
        
        text_str = f"Gen Value {key}: {true_vals[key]:.3f}"
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.title(f'Pull Distribution: {key}')
        plt.xlabel(r'$(Val_{fit} - Val_{true}) / \sigma_{fit}$')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)        
        plt.savefig(f'Plots/Pulls/Pull_{key}.png')
        plt.close()

    # ==========================
    # PLOTTING 2: VALORES (ESTIMADORES)
    # ==========================
    print("Generando gráficas individuales de Estimadores (Valores)...")
    
    for key in ordered_keys:
        data = np.array(results[key])        
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=20, density=True, alpha=0.6, color='blue', edgecolor='black', label='Fitted Values')
        
        # Ajuste Gaussiano con Errores
        if len(data) > 1:
            mu, std = norm.fit(data)
            N = len(data)
            mu_err = std / np.sqrt(N)
            std_err = std / np.sqrt(2 * N)
            x_plot = np.linspace(min(data), max(data), 100)
            p = norm.pdf(x_plot, mu, std)
            
            label_fit = (f'Gauss Fit:\n'
                         f'$\mu = {mu:.4f} \pm {mu_err:.4f}$\n'
                         f'$\sigma = {std:.4f} \pm {std_err:.4f}$')
            
            plt.plot(x_plot, p, 'r-', linewidth=2, label=label_fit)
            
        true_val = true_vals[key]
        plt.axvline(true_val, color='blue', linestyle='--', linewidth=2, label=f'True: {true_val}')
        plt.text(0.05, 0.95, f"Observable: {key}", transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.title(f'Estimator Distribution: {key}')
        plt.xlabel(f'{key} (Physical Unit)')
        plt.ylabel('Density')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'Plots/Values/Val_{key}.png')
        plt.close()

    
    #TABLA DE RESULTADOS (LATEX)
    print("\n" + "="*40)
    print("      TABLA DE RESULTADOS DE PULLS      ")
    print("="*40)
    
    # Crear una lista para pandas
    table_data = []
    
    for key in ordered_keys:
        data = np.array(pulls[key])
        data_clean = data[np.abs(data) < 5]
        
        if len(data_clean) > 1:
            mu, std = norm.fit(data_clean)
            N = len(data_clean)
            mu_err = std / np.sqrt(N)
            std_err = std / np.sqrt(2*N)          
            table_data.append({
                "Observable": f"${key}$",
                "Mean Pull ($\mu$)": f"{mu:.3f} $\pm$ {mu_err:.3f}",
                "Pull Width ($\sigma$)": f"{std:.3f} $\pm$ {std_err:.3f}",
                "Status": "OK" if (abs(mu) < 0.1 and abs(std-1) < 0.2) else "Check"
            })

    df_results = pd.DataFrame(table_data)
    print(df_results)
    latex_code = df_results.to_latex(index=False, escape=False, caption="Resumen de los Pulls obtenidos con Toy Monte Carlo.", label="tab:pulls_summary")
    
    print("\n--- CÓDIGO LATEX PARA TU TESIS ---\n")
    print(latex_code)
    print("----------------------------------\n")
    print("--- Estudio finalizado. Revisar carpeta 'Plots/' ---")

if __name__ == "__main__":
    run_toy_study(n_toys=1000, n_events=2000)


