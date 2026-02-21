import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Configurar el estilo general de CMS
hep.style.use(hep.style.CMS)

# --- DATOS EXPERIMENTALES (LHCb) ---
# (Se asume la misma lista 'experimental_data' que ya teníamos)
experimental_data = [
    {'label': '0.10 < q² < 0.98', 'params': {'FL': (0.255, 0.032, 0.007), 'S3': (0.034, 0.044, 0.003), 'S4': (0.059, 0.050, 0.004), 'S5': (0.227, 0.041, 0.008), 'AFB': (-0.004, 0.040, 0.004), 'S7': (0.006, 0.042, 0.002), 'S8': (-0.003, 0.051, 0.001), 'S9': (-0.055, 0.041, 0.002)}},
    {'label': '1.1 < q² < 2.5', 'params': {'FL': (0.655, 0.046, 0.017), 'S3': (-0.107, 0.052, 0.003), 'S4': (-0.038, 0.070, 0.011), 'S5': (0.174, 0.060, 0.007), 'AFB': (-0.229, 0.046, 0.009), 'S7': (-0.107, 0.063, 0.004), 'S8': (-0.174, 0.075, 0.002), 'S9': (-0.112, 0.054, 0.005)}},
    {'label': '2.5 < q² < 4.0', 'params': {'FL': (0.756, 0.047, 0.023), 'S3': (0.020, 0.053, 0.002), 'S4': (-0.187, 0.074, 0.008), 'S5': (-0.064, 0.068, 0.010), 'AFB': (-0.070, 0.043, 0.006), 'S7': (-0.066, 0.065, 0.004), 'S8': (0.016, 0.074, 0.002), 'S9': (-0.012, 0.055, 0.003)}},
    {'label': '4.0 < q² < 6.0', 'params': {'FL': (0.684, 0.035, 0.015), 'S3': (0.014, 0.038, 0.003), 'S4': (-0.145, 0.057, 0.004), 'S5': (-0.204, 0.051, 0.013), 'AFB': (0.050, 0.033, 0.002), 'S7': (-0.136, 0.053, 0.002), 'S8': (0.077, 0.062, 0.001), 'S9': (0.029, 0.045, 0.002)}},
    {'label': '6.0 < q² < 8.0', 'params': {'FL': (0.645, 0.030, 0.011), 'S3': (-0.013, 0.038, 0.004), 'S4': (-0.275, 0.045, 0.006), 'S5': (-0.279, 0.043, 0.013), 'AFB': (0.110, 0.027, 0.005), 'S7': (-0.074, 0.046, 0.003), 'S8': (-0.062, 0.047, 0.001), 'S9': (0.024, 0.035, 0.002)}},
    {'label': '11.0 < q² < 12.5', 'params': {'FL': (0.461, 0.031, 0.010), 'S3': (-0.124, 0.037, 0.003), 'S4': (-0.245, 0.047, 0.007), 'S5': (-0.310, 0.043, 0.011), 'AFB': (0.333, 0.030, 0.008), 'S7': (-0.096, 0.050, 0.003), 'S8': (0.009, 0.049, 0.001), 'S9': (0.042, 0.040, 0.003)}},
    {'label': '15.0 < q² < 17.0', 'params': {'FL': (0.352, 0.026, 0.009), 'S3': (-0.166, 0.034, 0.007), 'S4': (-0.299, 0.033, 0.008), 'S5': (-0.341, 0.034, 0.009), 'AFB': (0.385, 0.024, 0.007), 'S7': (0.029, 0.039, 0.001), 'S8': (0.003, 0.042, 0.002), 'S9': (0.000, 0.037, 0.002)}}
]

# --- DATOS DEL USUARIO (CMS) ---
cms_q2_bins = {
    "bin1": [1.1, 2.0], "bin2": [2.0, 4.0], "bin3": [4.0, 6.0], 
    "bin4": [6.0, 7.0], "bin5": [7.0, 8.0], "bin7": [11.0, 12.5], 
    "bin9": [15.0, 17.0], "bin10": [17.0, 23.0]
}
bin_indices = [1, 2, 3, 4, 5, 7, 9, 10]
# Ajusta esta ruta base según el directorio de tu máquina (ej. local o en lxplus)
cms_base_dir = "fit_results/gen_phy" 

def generate_comparison_plots(lhcb_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    observables = list(lhcb_data[0]['params'].keys())
    
    print(f"Iniciando la generación de gráficos de comparación para {len(observables)} observables...")

    for obs_name in observables:
        fig, ax0 = plt.subplots(figsize=(10, 8))

        # ==========================================
        # 1. PLOTEAR DATOS LHCb
        # ==========================================
        lhcb_q2_centers, lhcb_q2_xerrs, lhcb_vals, lhcb_errs = [], [], [], []
        for bin_data in lhcb_data:
            label_parts = bin_data['label'].split('<')
            q2_min = float(label_parts[0].strip())
            q2_max = float(label_parts[2].strip())
            
            lhcb_q2_centers.append((q2_max + q2_min) / 2.0)
            lhcb_q2_xerrs.append((q2_max - q2_min) / 2.0)
            
            val, err_stat, err_sys = bin_data['params'][obs_name]
            lhcb_vals.append(val)
            lhcb_errs.append(np.sqrt(err_stat**2 + err_sys**2)) # Suma en cuadratura
            
        # Graficar LHCb (usamos cuadrados azules para contrastar fuertemente con CMS)
        ax0.errorbar(lhcb_q2_centers, lhcb_vals, xerr=lhcb_q2_xerrs, yerr=lhcb_errs, 
                     fmt='s', color='#1f77b4', markersize=7, elinewidth=1.5, capsize=0, 
                     label='LHCb Measurement', alpha=0.8)

        # ==========================================
        # 2. PLOTEAR TUS DATOS (CMS Fits)
        # ==========================================
        cms_q2_centers, cms_q2_xerrs, cms_vals = [], [], []
        cms_errs_low, cms_errs_up = [], []

        for b in bin_indices:
            bin_key = f"bin{b}"
            q2_min, q2_max = cms_q2_bins[bin_key]
            
            # Construir la ruta al JSON
            json_path = os.path.join(cms_base_dir, bin_key, f"fit_results_gen_physical_{bin_key}.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    fit_data = json.load(f)
                
                # La llave en el JSON tiene el formato 'FL_bin1'
                json_obs_key = f"{obs_name}_{bin_key}"
                
                if json_obs_key in fit_data['parameters']:
                    param_data = fit_data['parameters'][json_obs_key]
                    
                    cms_q2_centers.append((q2_max + q2_min) / 2.0)
                    cms_q2_xerrs.append((q2_max - q2_min) / 2.0)
                    cms_vals.append(param_data['value'])
                    
                    # Matplotlib requiere que los errores sean positivos para asimétricos
                    cms_errs_low.append(abs(param_data['error_low']))
                    cms_errs_up.append(param_data['error_up'])
                else:
                    print(f"  [Advertencia] Observable {json_obs_key} no encontrado en {json_path}")
            else:
                print(f"  [Advertencia] Archivo no encontrado: {json_path}")

        # Graficar tus mediciones si se encontraron datos
        if cms_vals:
            # yerr asimétrico en matplotlib debe tener forma (2, N): [errores_inferiores, errores_superiores]
            cms_yerr = np.array([cms_errs_low, cms_errs_up])
            
            # Graficamos CMS como puntos negros (estándar de publicación)
            ax0.errorbar(cms_q2_centers, cms_vals, xerr=cms_q2_xerrs, yerr=cms_yerr, 
                         fmt='o', color='black', markersize=8, elinewidth=2.0, capsize=0, 
                         label='CMS Measurements')

        # ==========================================
        # 3. ESTÉTICA Y GUARDADO
        # ==========================================
        hep.cms.label(data=False, loc=0, ax=ax0, rlabel="13 TeV", fontname="sans-serif", fontsize=16)
        
        # Legend con un poco de diseño para no estorbar
        ax0.legend(frameon=True, edgecolor='white', framealpha=0.8, fontsize=13, loc='upper right')
        ax0.grid(True, alpha=0.3)
        
        # Etiquetas de ejes con LaTeX
        if obs_name == 'AFB':
            y_label = r'$A_{\mathrm{FB}}$'
        elif obs_name == 'FL':
            y_label = r'$F_{\mathrm{L}}$'
        else:
            y_label = rf'${obs_name[0]}_{{{obs_name[1:]}}}$' 

        ax0.set_xlabel(r'$q^2 \quad [\mathrm{GeV}^2/c^4]$', fontsize=18)
        ax0.set_ylabel(y_label, fontsize=18)
        ax0.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Establecer límites razonables (en base a tus bines, llegas a 23.0)
        ax0.set_xlim(0, 24)

        plt.tight_layout()
        file_path = os.path.join(output_dir, f"Comparison_{obs_name}.png")
        plt.savefig(file_path, format='png', bbox_inches='tight')
        plt.close(fig)
        
        print(f"Guardado: {file_path}")

if __name__ == "__main__":
    output_directory = "plots/comparisons"
    generate_comparison_plots(experimental_data, output_directory)