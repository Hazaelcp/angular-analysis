import pandas as pd  
from pandas import Series, DataFrame 
import uproot 
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import comb
from scipy.stats import chi2
from scipy.special import comb
from scipy.optimize import lsq_linear
import sys
from plot_tools import *
from customStats import *
#import tools
import common_tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
# from selection_cuts import selection_nominal
import mplhep as hep
from sklearn.model_selection import train_test_split
plt.style.use(hep.style.CMS)
plt.rcParams['figure.figsize'] = [10,8]
plt.rcParams['font.size'] = 24
plt.figure()
plt.close()
plt.rcParams.update({'figure.figsize':[10,8]})
plt.rcParams.update({'font.size':24})
import tensorflow as tf
import math
import zfit
from zfit import z
import xgboost as xgb
from scipy.interpolate import make_interp_spline
# from loadCutXGB import load_and_cutXGBclfs
from scipy.special import comb
from scipy.optimize import lsq_linear
zfit.settings.set_verbosity(0)
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Oculta los mensajes de INFO y WARNING
from PDFs import *









def save_fit_results(result, bin_n, base_dir="fit_results", name="fit_results"):
    """
    Guarda resultados buscando errores con nombres personalizados ('minos') 
    o por defecto ('minuit_minos', 'minuit_hesse').
    """
    
    output_folder = os.path.join(base_dir, f"{bin_n}")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{name}.json")
    
    params_dict = {}
    
    for p in result.params:
        val = result.params[p]['value']
        p_data = result.params[p] 

        lower_err = 0.0
        upper_err = 0.0
        sym_err = 0.0
        error_type = "none"

        # busca minos
        if 'minos' in p_data:
            err_data = p_data['minos']
            lower_err = err_data.get('lower', 0.0)
            upper_err = err_data.get('upper', 0.0)
            sym_err = (abs(lower_err) + abs(upper_err)) / 2.0
            error_type = "minos (custom)"

        # busca minuit_minos
        elif 'minuit_minos' in p_data:
            err_data = p_data['minuit_minos']
            lower_err = err_data.get('lower', 0.0)
            upper_err = err_data.get('upper', 0.0)
            sym_err = (abs(lower_err) + abs(upper_err)) / 2.0
            error_type = "minos (default)"
            
        # busca Hesse
        elif 'minuit_hesse' in p_data:
            err_data = p_data['minuit_hesse']
            sym_err = err_data.get('error', -999.0)
            lower_err = -sym_err
            upper_err = sym_err
            error_type = "hesse"
            

        params_dict[p.name] = {'value': float(val), 'error': float(sym_err), 'error_low': float(lower_err), 'error_up': float(upper_err), 'error_source': error_type}


    cov_matrix = result.covariance()
    cov_list = np.array(cov_matrix).tolist()
    data_to_save = {'bin_index': str(bin_n), 'valid': bool(result.valid), 'converged': bool(result.converged), 'fmin': float(result.fmin), 'status': result.status,'parameters': params_dict,'covariance': cov_list}
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"[CheckPoint] Resultados guardados en: {output_file}")
    return output_file


def save_correlation_matrix(result, params_list, bin_name, out_dir="plots/correlations"):

    os.makedirs(out_dir, exist_ok=True)    
    corr_matrix_raw = result.correlation()    
    zfit_params = list(result.params.keys())
    n_params = len(params_list)
    corr_matrix = np.zeros((n_params, n_params))    
    param_names = [p.name.split('_')[0] for p in params_list]    
    for i, p1 in enumerate(params_list):
        for j, p2 in enumerate(params_list):
            idx1 = zfit_params.index(p1)
            idx2 = zfit_params.index(p2)
            corr_matrix[i, j] = corr_matrix_raw[idx1, idx2]
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=param_names, yticklabels=param_names, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title(f"Matriz de Correlación - {bin_name}", fontsize=14, pad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filepath = os.path.join(out_dir, f"corr_matrix_{bin_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close() 


def save_phy_results(result_zfit, phys_dict, cov_phys, obs_order, bin_n, base_dir="fit_results", name="fit_results_phys"):
    """
    Guarda los resultados de los observables físicos transformados y su matriz 
    de covarianza propagada en el mismo formato JSON que los resultados del fit.
    """
    output_folder = os.path.join(base_dir, f"{bin_n}")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{name}.json")
    
    params_dict = {}
    
    # Extraemos los errores de la diagonal de la matriz de covarianza física
    errors_phys = np.sqrt(np.diag(cov_phys))
    
    for i, obs_name in enumerate(obs_order):
        val = phys_dict.get(obs_name, 0.0)
        sym_err = errors_phys[i]
        
        # El Método Delta asume errores simétricos
        params_dict[obs_name] = {
            'value': float(val), 
            'error': float(sym_err), 
            'error_low': float(-sym_err), 
            'error_up': float(sym_err), 
            'error_source': "delta_method"
        }

    # Convertimos la matriz de covarianza de numpy a lista para JSON
    cov_list = cov_phys.tolist()
    
    # Reutilizamos la metadata del fit original (convergencia, fmin, status)
    data_to_save = {
        'bin_index': str(bin_n), 
        'valid': bool(result_zfit.valid), 
        'converged': bool(result_zfit.converged), 
        'fmin': float(result_zfit.fmin), 
        'status': result_zfit.status,
        'parameters': params_dict,
        'covariance': cov_list
    }
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"[CheckPoint] Resultados físicos (transformados) guardados en: {output_file}")
    return output_file

 
def plot_nll_profiles(result, params_list, bin_name, base_dir):
    bin_dir = os.path.join(base_dir)
    os.makedirs(bin_dir, exist_ok=True)
    
    loss = result.loss
    nll_min = result.fmin
    minimizer = zfit.minimize.Minuit(verbosity=0)
    best_fit_values = {p: result.params[p]['value'] for p in params_list}
    
    for param in params_list:
        clean_name = param.name.split('_')[0]
        print(f"--- Calculando perfil NLL para {param.name} ---")
        
        val_opt = best_fit_values[param]
        minos_data = result.params[param].get('minos', {})
        err_lower = minos_data.get('lower', -0.1)
        err_upper = minos_data.get('upper', 0.1)
        
        scan_min = val_opt + 1.2*err_lower
        scan_max = val_opt + 1.2*err_upper
        
        if param.has_limits:
            p_lower = float(param.lower)
            p_upper = float(param.upper)
            epsilon = 1e-4
            
            scan_min = max(scan_min, p_lower + epsilon)
            scan_max = min(scan_max, p_upper - epsilon)
            
        scan_values = np.linspace(scan_min, scan_max, 50)
        nll_values = []
        
        # 4. Escaneo del NLL
        for val_scan in scan_values:
            for p in params_list:
                p.set_value(best_fit_values[p])
            
            param.set_value(val_scan)
            param.floating = False
            temp_result = minimizer.minimize(loss)
            nll_values.append(temp_result.fmin)


                
        param.floating = True
        nll_values = np.array(nll_values)
        delta_nll = nll_values - nll_min
        
        # ==========================================
        # CREACIÓN Y GUARDADO DEL GRÁFICO INDIVIDUAL
        # ==========================================
        # Asegúrate de haber importado mplhep antes
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(scan_values, delta_nll, '-', lw=1, color='blue', label='Profile Likelihood')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label=r'$\Delta$NLL = 0.5 (Errors $1\sigma$)')
        ax.axvline(val_opt, color='green', linestyle='-.', linewidth=1.5, label='Best fit')   

        ax.set_title(f'NLL Profile: {clean_name} ({bin_name})', loc='center', fontsize=14, fontweight='medium', y=1.05)
        ax.set_xlabel(f'{clean_name}', fontsize=16)
        ax.set_ylabel(r'$\Delta$ NLL', fontsize=16)
        ax.set_ylim(bottom=-0.1, top=0.52) 

        hep.cms.label(data=False, loc=0, ax=ax, rlabel="13 TeV", fontname="sans-serif", fontsize=16)

        ax.legend(frameon=False, fontsize=13, loc='upper right') # 'upper center' también suele funcionar bien en perfiles NLL
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in', top=True, right=True)

        file_path = f"/nll_profile_{clean_name}_{bin_name}.png"
        plt.savefig(bin_dir+file_path, dpi=300, bbox_inches='tight')
        plt.close()



def get_physical_array(r_array, obs_order):
    phys_dict = apply_transformation_equations(*r_array)
    return np.array([phys_dict.get(key, 0.0) for key in obs_order])


class DuplicarSalida:
    def __init__(self, archivo):
        self.terminal = sys.stdout
        self.log = open(archivo, "w", encoding="utf-8")

    def write(self, mensaje):
        self.terminal.write(mensaje)
        self.log.write(mensaje)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_jacobian(func, x, epsilon=1e-5):
    n_in = len(x)
    y_center = func(x)
    n_out = len(y_center)
    
    J = np.zeros((n_out, n_in))
    
    for i in range(n_in):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        y_plus = func(x_plus)
        y_minus = func(x_minus)
        
        J[:, i] = (y_plus - y_minus) / (2.0 * epsilon)
        
    return J