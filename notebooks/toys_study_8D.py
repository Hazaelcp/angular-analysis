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
import mplhep as hep
from matplotlib.gridspec import GridSpec
hep.style.use("CMS")


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

    # Valores iniciales LHCb Collaboration
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
    
    # Bucle de Toys (Lógica intacta)
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

    # ==============================================================================
    # PLOTEO
    # ==============================================================================

    latex_labels = {'FL': r'F_L', 'AFB': r'A_{FB}', 'S3': r'S_3', 'S4': r'S_4', 'S5': r'S_5', 'S7': r'S_7', 'S8': r'S_8', 'S9': r'S_9'}

    def plot_cms_style(data_array, var_name, filename, is_pull=True, true_val=None):

        tex_name = latex_labels.get(var_name, var_name)
        data = np.array(data_array)
        
        # --- CONFIGURACIÓN DE RANGOS ---
        if is_pull:
            # Quitamos outliers extremos solo para el ajuste gaussiano
            data_clean_for_fit = data[np.abs(data) < 5] 
            
            # PERO usamos todos los datos (o un rango fijo) para el histograma
            # Esto FIJA el eje X entre -5 y 5. El 0 siempre estará al medio.
            plot_range = (-5, 5) 
            xlabel_text = rf'$({{{tex_name}}}_{{Fit}}-{{{tex_name}}}_{{True}})/\sigma$'
            bins_n = 40 # Un buen número para pulls
            fit_line_color = '#005293' # Azul CMS
            color_hist = 'black'
            
        else:
            data_clean_for_fit = data 
            # Para valores físicos, dejamos que el rango sea automático (None)
            plot_range = None 
            
            xlabel_text = rf'${{{tex_name}}}_{{Fit}}$'
            bins_n = 40
            fit_line_color = '#CC0000' # Rojo
            color_hist = 'black'
            
        # 1. Crear Histograma con RANGO CONTROLADO
        # Si plot_range es (-5,5), el eje no se moverá nunca.
        counts, bin_edges = np.histogram(data, bins=bins_n, range=plot_range)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        y_err = np.sqrt(counts)
        
        # 2. Ajuste Gaussiano (usando data limpia de outliers para no sesgar el fit)
        mu, std = norm.fit(data_clean_for_fit)        
        
        # Generar puntos para la línea suave del ajuste
        x_plot = np.linspace(bin_edges[0], bin_edges[-1], 200)
        # Normalización correcta: PDF * NumeroEventos * AnchoBin
        # Nota: Usamos len(data) si están todos en rango, o sum(counts) para ser precisos en lo visible
        scale_factor = np.sum(counts) * bin_width
        y_gauss = norm.pdf(x_plot, mu, std) * scale_factor
        
        # Calcular residuos para el panel inferior
        y_gauss_bins = norm.pdf(bin_centers, mu, std) * scale_factor
        y_err_safe = np.where(y_err == 0, 1, y_err) 
        residuals = (counts - y_gauss_bins) / y_err_safe
        residuals[counts == 0] = 0 

        # --- PLOTEO ---
        fig = plt.figure(figsize=(9, 9))
        gs = GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.08)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

        # Panel Superior
        ax0.errorbar(bin_centers, counts, yerr=y_err, xerr=bin_width/2, fmt='o', 
                     color=color_hist, markersize=4, elinewidth=1, capsize=0, label='Toys')                
        ax0.plot(x_plot, y_gauss, color=fit_line_color, linewidth=2, label='Gaussian Fit')
        
        if not is_pull and true_val is not None:
             ax0.axvline(true_val, color='gray', linestyle='--', alpha=0.7, label='Gen Value')

        # Estética Panel Superior
        max_y = max(np.max(counts + y_err), np.max(y_gauss))
        ax0.set_ylim(0, max_y * 1.4) # Margen arriba para el texto
        ax0.set_ylabel(f'Events / {bin_width:.2f}', fontsize=16)
        ax0.tick_params(labelbottom=False, labelsize=12)
        ax0.legend(loc='upper right', frameon=False, fontsize=13)
        hep.cms.label(data=False, loc=0, ax=ax0, rlabel="(13.6 TeV)")
        ax0.grid(True, alpha=0.4)

        # Caja de Texto con Estadísticas
        mu_err = std / np.sqrt(len(data_clean_for_fit))
        std_err = std / np.sqrt(2 * len(data_clean_for_fit))
        stats_text = (rf'$\mathbf{{{tex_name}}}$' +'\n'+  
                      rf'$\mu = {mu:.3f} \pm {mu_err:.3f}$' + '\n' +  
                      rf'$\sigma = {std:.3f} \pm {std_err:.3f}$')
        ax0.text(0.05, 0.92, stats_text, transform=ax0.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

        # Panel Inferior (Residuos)
        ax1.errorbar(bin_centers, residuals, yerr=1.0, xerr=0, fmt='o', color='black', markersize=3, elinewidth=1)
        ax1.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Bandas de sigma en residuos
        ax1.fill_between([bin_edges[0], bin_edges[-1]], -3, 3, color='black', alpha=0.1) # 2 sigma
        
        ax1.set_xlabel(xlabel_text, fontsize=14)
        ax1.set_ylabel(r'$Pull$', fontsize=12)
        ax1.set_ylim(-3.9, 3.9)
        
        # FORZAR LÍMITES DEL EJE X
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        
        ax1.grid(True, alpha=0.4, axis='x')
        ax1.tick_params(labelsize=12)

        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close(fig)

    print("\nGenerando gráficas de Pulls estilo CMS...")
    for key in ordered_keys:
        plot_cms_style(pulls[key], key, f'Plots/Pulls/Pull_{key}.png', is_pull=True)

    print("Generando gráficas de Estimadores estilo CMS...")
    for key in ordered_keys:
        plot_cms_style(results[key], key, f'Plots/Values/Val_{key}.png', is_pull=False, true_val=true_vals[key])


    # ==========================
    # TABLA DE RESULTADOS (LATEX)
    # ==========================
    print("\n" + "="*40)
    print("      TABLA DE RESULTADOS DE PULLS      ")
    print("="*40)
    
    table_data = []
    
    for key in ordered_keys:
        data = np.array(pulls[key])
        data_clean = data[np.abs(data) < 5]
        tex_name = latex_labels.get(key, key)
        
        if len(data_clean) > 1:
            mu, std = norm.fit(data_clean)
            N = len(data_clean)
            mu_err = std / np.sqrt(N)
            std_err = std / np.sqrt(2*N)          
            table_data.append({
                "Observable": f"${tex_name}$",
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
    run_toy_study(n_toys=800, n_events=2000)