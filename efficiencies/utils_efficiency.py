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



def bernstein_1d(n, k, t):
    """Bernstein base polynomial B_{n,k}(t) on t in [0,1]."""
    return comb(n, k) * (t**k) * ((1.0 - t)**(n - k))


def bernstein2d_matrix(nx, ny, x, y):
    """
    Build the design matrix for Bernstein2D basis.
    Input: x, y in [-1,1] (arrays)
    Output: B matrix of size (Npoints, (nx+1)*(ny+1))
    """
    # map [-1,1] -> [0,1]
    tx = 0.5*(x + 1.0)
    ty = 0.5*(y + 1.0)

    B_list = []
    for i in range(nx+1):
        for j in range(ny+1):
            B_list.append(bernstein_1d(nx, i, tx) * bernstein_1d(ny, j, ty))
    B = np.vstack(B_list).T   # shape (Npoints, Ncoef)
    return B


# ======================================================
# 2) Fit Bernstein2D to a 2D efficiency map
# ======================================================
def fit_bernstein2d( xcenters, ycenters, eff2d, ngen2d, nx=8, ny=8, min_counts_mask=None,reg_lambda=1e-10,):
    """
    Fits a Bernstein2D polynomial to a 2D efficiency map using least squares + Tikhonov reg.

    Inputs:
        xcenters, ycenters   1D arrays (bin centers)
        eff2d                2D array with efficiency values (NaNs allowed)
        nx, ny               polynomial orders
        min_counts_mask      Boolean 2D mask (valid bins: True)
        reg_lambda           regularization parameter

    Returns:
        coef                 fitted coefficients
        eff_model            modeled efficiency on the grid

    
    Weighted fit of a Bernstein2D polynomial to a 2D efficiency map.

    The weights are derived from binomial uncertainties:
        sigma^2 = eff * (1 - eff) / ngen
    """

    XX, YY = np.meshgrid(xcenters, ycenters, indexing="ij")
    xflat = XX.ravel()
    yflat = YY.ravel()
    eff_flat = eff2d.ravel()
    ngen_flat = ngen2d.ravel()

    # Valid bins
    if min_counts_mask is None: use = (~np.isnan(eff_flat)) & (ngen_flat > 0)
    else:
        use = (min_counts_mask.ravel() & ~np.isnan(eff_flat)& (ngen_flat > 0))

    x_use = xflat[use]
    y_use = yflat[use]
    eff_use = eff_flat[use]
    ngen_use = ngen_flat[use]

    # Binomial uncertainty
    sigma2 = eff_use * (1.0 - eff_use) / ngen_use
    sigma2 = np.clip(sigma2, 1e-12, None)
    w = 1.0 / np.sqrt(sigma2)
    B = bernstein2d_matrix(nx, ny, x_use, y_use)

    # Apply weights
    Bw = B * w[:, None]
    yw = eff_use * w

    # Regularized weighted least squares
    BTB = Bw.T @ Bw + reg_lambda * np.eye(B.shape[1])
    BTy = Bw.T @ yw
    coef = np.linalg.solve(BTB, BTy)
    Bfull = bernstein2d_matrix(nx, ny, xflat, yflat)
    eff_model_flat = Bfull @ coef
    eff_model = eff_model_flat.reshape(eff2d.shape)
    return coef, eff_model


def build_efficiency_2d(gen_all_x, gen_all_y, gen_fid_x, gen_fid_y, reco_fid_x, reco_fid_y, reco_x, reco_y, 
                        weights_reco=None, nbx=20, nby=20, nxg=8, nyg=8, nxr=8, nyr=8, min_gen=0, reg_acc=1e-4, reg_reco=1e-4):
    xedges = np.linspace(-1, 1, nbx + 1)
    yedges = np.linspace(-1, 1, nby + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # Pesos 
    if weights_reco is None:
        weights_reco = np.ones(len(reco_x))

    # Histograms
    gen_allH, _, _ = np.histogram2d(gen_all_x, gen_all_y, bins=[xedges, yedges])
    gen_fidH, _, _ = np.histogram2d(gen_fid_x, gen_fid_y, bins=[xedges, yedges])
    # Denominador Reco
    reco_fidH, _, _ = np.histogram2d(reco_fid_x, reco_fid_y, bins=[xedges, yedges])    
    # Numerador Reco (CON PESOS APLICADOS)
    recoH, _, _ = np.histogram2d(reco_x, reco_y, bins=[xedges, yedges], weights=weights_reco)

    # ============================
    # Acceptance
    # ============================
    mask_gen = gen_allH > min_gen
    acc_gen = np.full_like(gen_allH, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_gen[mask_gen] = gen_fidH[mask_gen] / gen_allH[mask_gen]

    # ============================
    # Reconstruction efficiency
    # ============================
    eff_reco = np.full_like(gen_allH, np.nan)
    valid = mask_gen & (reco_fidH > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_reco[valid] = recoH[valid] / reco_fidH[valid]
    
    # ============================
    # Bernstein fits
    # ============================
    # Nota: Los fits también deben saber que recoH ahora tiene pesos (float), no solo counts (int).
    # Asegúrate de que fit_bernstein2d maneje arrays de floats en "data_hist" (recoH).
    coef_acc, acc_gen_model = fit_bernstein2d( xcenters, ycenters, acc_gen, gen_allH, nx=nxg, ny=nyg, min_counts_mask=mask_gen, reg_lambda=reg_acc)
    coef_reco, eff_reco_model = fit_bernstein2d(xcenters, ycenters, eff_reco, reco_fidH, nx=nxr, ny=nyr, min_counts_mask=valid, reg_lambda=reg_reco)

    return (xcenters, ycenters, acc_gen, acc_gen_model, coef_acc, eff_reco, eff_reco_model, coef_reco, mask_gen)


def build_efficiency_1d(gen_all, gen_fid, reco_fid, reco, weights_reco=None, nbins=30, n_poly=4, min_gen=10, reg_acc=1e-6, reg_reco=1e-5):

    limit_min, limit_max = -np.pi, np.pi
    edges = np.linspace(limit_min, limit_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers_norm = 2 * (centers - limit_min) / (limit_max - limit_min) - 1.0

    # Pesos
    if weights_reco is None:
        weights_reco = np.ones(len(reco))
    
    # Histograms
    gen_allH, _ = np.histogram(gen_all, bins=edges)
    gen_fidH, _ = np.histogram(gen_fid, bins=edges)
    reco_fidH, _ = np.histogram(reco_fid, bins=edges)
    
    # Numerador Reco CON PESOS
    recoH, _ = np.histogram(reco, bins=edges, weights=weights_reco)

    # Acceptance
    mask_gen = gen_allH > min_gen
    acc_gen = np.full_like(gen_allH, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_gen[mask_gen] = gen_fidH[mask_gen] / gen_allH[mask_gen]
    coef_acc, acc_model = fit_bernstein1d( centers_norm, acc_gen, gen_allH, n=n_poly, min_counts_mask=mask_gen, reg_lambda=reg_acc)
    
    # Efficiency reco
    eff_reco = np.full_like(gen_allH, np.nan, dtype=float)
    valid_reco = mask_gen & (reco_fidH > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_reco[valid_reco] = recoH[valid_reco] / reco_fidH[valid_reco]
    coef_reco, eff_reco_model = fit_bernstein1d(centers_norm, eff_reco, reco_fidH, n=n_poly, min_counts_mask=valid_reco, reg_lambda=reg_reco)

    return centers, acc_gen, acc_model, coef_acc, eff_reco, eff_reco_model, coef_reco, mask_gen


def bernstein2d_eval(x, y, model):
    """
    Evaluate a fitted Bernstein2D model.
    """
    nx = model["nx"]
    ny = model["ny"]
    coef = np.asarray(model["coef"])
    tx = 0.5 * (x + 1.0)
    ty = 0.5 * (y + 1.0)
    eff = np.zeros_like(tx, dtype=float)
    idx = 0
    for i in range(nx + 1):
        Bx = bernstein_1d(nx, i, tx)
        for j in range(ny + 1):
            By = bernstein_1d(ny, j, ty)
            eff += coef[idx] * Bx * By
            idx += 1

    return eff


# ======================================================
# Save / Load Bernstein models
# ======================================================
def save_bernstein2d_model(filename, coef, nx, ny):
    model = {"nx": nx, "ny": ny, "coef": coef.tolist(), "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(model, f, indent=2)


def load_bernstein_model(filename):
    with open(filename) as f:
        model = json.load(f)
    return (np.asarray(model["coef"], dtype=np.float64), model["nx"], model["ny"])


# ======================================================
#   plots
# ======================================================
def project_with_errors_x(data2d, mask):
    """Project 2D efficiency to x with diagnostic errors."""
    proj = []
    err = []
    for i in range(data2d.shape[0]):   # x bins
        vals = data2d[i, :][mask[i, :]]
        if len(vals) == 0:
            proj.append(np.nan)
            err.append(np.nan)
        else:
            proj.append(np.mean(vals))
            err.append(np.std(vals, ddof=0) / np.sqrt(len(vals)))
    return np.array(proj), np.array(err)


def project_with_errors_y(data2d, mask):
    """Project 2D efficiency to y with diagnostic errors."""
    proj = []
    err = []
    for j in range(data2d.shape[1]):   # y bins
        vals = data2d[:, j][mask[:, j]]
        if len(vals) == 0:
            proj.append(np.nan)
            err.append(np.nan)
        else:
            proj.append(np.mean(vals))
            err.append(np.std(vals, ddof=0) / np.sqrt(len(vals)))
    return np.array(proj), np.array(err)


# ======================================================
# CMS Style Plotting
# ======================================================
def _plot_cms_style(centers, data, data_err, model, xlabel, title, y_label="Efficiency", ylim=None, path_dir="plots"):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1.5], hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # --- AUTO-SCALING LOGIC ---
    if ylim is None:
        # Calculamos el punto más alto considerando el error superior
        # Usamos nanmax para ignorar posibles NaNs
        max_data = np.nanmax(data + data_err) if data_err is not None else np.nanmax(data)
        max_model = np.nanmax(model)
        global_max = max(max_data, max_model)
        
        # Si por alguna razón todo es 0 o NaN, ponemos un default
        if np.isnan(global_max) or global_max <= 0:
            global_max = 1.0
            
        # Definimos el límite: 0 abajo, y Max + 30% arriba para la leyenda/CMS label
        current_ylim = (0.0, global_max * 1.30)
    else:
        current_ylim = ylim
    # --------------------------

    # Plot principal
    ax0.plot(centers, model, '-', color='blue', linewidth=2.5, label="Bernstein Model")
    ax0.errorbar(centers, data, yerr=data_err, fmt='ks', markersize=5, elinewidth=1.5, capsize=2, label="Binned MC")
    ax0.set_ylabel(y_label, fontsize=16)
    ax0.set_title(title, loc='center', fontsize=14, fontweight='medium', y=1.05)
    
    # Aplicamos el límite calculado o el manual
    ax0.set_ylim(current_ylim)

    hep.cms.label(data=False, loc=0, ax=ax0, rlabel="13 TeV", fontname="sans-serif", fontsize=16)
    ax0.legend(frameon=False, fontsize=13, loc='upper right')
    ax0.grid(True, alpha=0.3)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # Pulls (El resto sigue igual)
    with np.errstate(divide='ignore', invalid='ignore'):
        pulls = (data - model) / data_err
    pulls[~np.isfinite(pulls)] = 0.0 

    width = centers[1] - centers[0]
    lower = centers[0] - width/2
    upper = centers[-1] + width/2

    ax1.errorbar(centers, pulls, yerr=1.0, xerr=0, fmt='ks',markersize=4, elinewidth=1.0,capsize=0)          
    ax1.axhline(0, color='black', linewidth=1.0, linestyle='-')
    ax1.axhline(3, color='gray', linestyle=':', linewidth=1, alpha=0.8) 
    ax1.axhline(-3, color='gray', linestyle=':', linewidth=1, alpha=0.8)    
    ax1.fill_between([lower, upper], -3, 3, color='gray', alpha=0.15, label=r'$3\sigma$') 
    ax1.set_xlabel(xlabel, fontsize=16)
    ax1.set_ylabel(r'Pull $(\sigma)$', fontsize=13)
    ax1.set_xlim(lower, upper)
    ax1.set_ylim(-4.9, 4.9)
    ax1.grid(True, alpha=0.3)
    ax0.tick_params(axis='both', which='major', labelsize=14, direction='in', top=True, right=True) 
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', top=True, right=True)
    
    # Guardado seguro
    save_path = os.path.join(path_dir, f"{title}.png")
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# Public Functions (Wrappers)
# ======================================================
def plot_projection_x_with_errors(xc, data2d, model2d, mask, title, ylim=None, path=None):
    """
    Proyección en X (cosThetaL)
    """

    data_proj, data_err = project_with_errors_x(data2d, mask)
    model_proj, _ = project_with_errors_x(model2d, mask)
    if np.nanmean(model_proj) != 0:
        scale = np.nanmean(data_proj) / np.nanmean(model_proj)
    else:
        scale = 1.0
    model_proj_scaled = model_proj * scale
    _plot_cms_style(centers=xc, data=data_proj, data_err=data_err, model=model_proj_scaled, xlabel=r"$\cos\theta_\ell$", title=title, ylim=ylim, path_dir=path)


def plot_projection_y_with_errors(yc, data2d, model2d, mask, title, ylim, path):
    """
    Proyección en Y (cosThetaK)
    """
    data_proj, data_err = project_with_errors_y(data2d, mask)
    model_proj, _ = project_with_errors_y(model2d, mask)
    
    if np.nanmean(model_proj) != 0:
        scale = np.nanmean(data_proj) / np.nanmean(model_proj)
    else:
        scale = 1.0
    model_proj_scaled = model_proj * scale

    _plot_cms_style(centers=yc, data=data_proj, data_err=data_err, model=model_proj_scaled, xlabel=r"$\cos\theta_K$", title=title, ylim=ylim, path_dir=path) 


def select_q2_bin(df, n_bin, cut):
    q2_bins = dict()
    q2_bins = { "bin0":[1.1,23.0],   "bin1":[1.1, 2.0],"bin2": [2.0, 4.0],"bin3":[4.0, 6.0],
                "bin4":[6.0, 7.0],   "bin5":[7.0, 8.0], "bin6": [8.0, 11.0],"bin7":[11.0, 12.5],
                "bin8":[12.5, 15.0], "bin9":[15.0, 17.0], "bin10":[17.0, 23.0]}
    df_ = df[(df[cut]>=q2_bins[n_bin][0]) & (df[cut] <= q2_bins[n_bin][1])].copy()
    return df_



# ======================================================
# PHI  1D Bernstei
# ======================================================

def save_bernstein1d_model(filename, coef, n):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    model = {"n": n, "coef": coef.tolist(), "range": [-np.pi, np.pi]}
    with open(filename, "w") as f:
        json.dump(model, f, indent=2)


def bernstein1d_matrix(n, x):
    """
    Build the design matrix for Bernstein 1D basis.
    Input: x in [-1, 1] (array)
    Output: B matrix of size (Npoints, n+1)
    """
    # map [-1,1] -> [0,1]
    t = 0.5 * (x + 1.0)
    B_list = []
    for i in range(n+1):
        B_list.append(bernstein_1d(n, i, t))
    B = np.vstack(B_list).T

    return B


def fit_bernstein1d(xcenters, eff1d, ngen1d, n=4, min_counts_mask=None, reg_lambda=1e-10):
    """
    Fits a Bernstein 1D polynomial to a 1D efficiency histogram.
    """
    # Filter NaNs and Zero Gen
    if min_counts_mask is None:
        use = (~np.isnan(eff1d)) & (ngen1d > 0)
    else:
        use = min_counts_mask & ~np.isnan(eff1d) & (ngen1d > 0)
        
    x_use = xcenters[use]
    eff_use = eff1d[use]
    ngen_use = ngen1d[use]
    # Binomial uncertainty weights
    sigma2 = eff_use * (1.0 - eff_use) / ngen_use
    sigma2 = np.clip(sigma2, 1e-12, None)
    w = 1.0 / np.sqrt(sigma2)
    
    B = bernstein1d_matrix(n, x_use)
    Bw = B * w[:, None]
    yw = eff_use * w
    BTB = Bw.T @ Bw + reg_lambda * np.eye(B.shape[1])
    BTy = Bw.T @ yw
    coef = np.linalg.solve(BTB, BTy)
    Bfull = bernstein1d_matrix(n, xcenters)
    eff_model = Bfull @ coef
    
    return coef, eff_model


def load_bernstein1d_model(filename):
    """Carga un modelo de Bernstein 1D desde un JSON."""

    with open(filename, 'r') as f:
        data = json.load(f)
    coef = np.asarray(data["coef"], dtype=np.float64)
    if "degree" in data:
        degree = data["degree"]
    else:
        degree = len(coef) - 1
        
    return coef, degree


def plot_1d_result(centers, data, model, mask, title, ylim, path):
    valid = mask
    dummy_err = np.zeros_like(data)
    dummy_err[valid] = 0.05 * data[valid]     
    _plot_cms_style(centers[valid], data[valid], dummy_err[valid], model[valid], xlabel=r"$\phi$", title=title,ylim=ylim, path_dir=path)


def run_fit(model, data):
    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    
    err = None
    try:
        err, _ = result.errors(name="minos", method="minuit_minos", cl=0.682)
    except Exception as e:
        print("MINOS failed:", e)
    return result, err

# =====================================================
# CODE FOR FIT INCLUDING EFFICIENCY
# =====================================================

def tf_bernstein_basis_vectorized(n, t):
    M = tf.shape(t)[0]
    k = tf.range(n + 1, dtype=tf.float64)
    n_float = tf.cast(n, tf.float64)
    log_binom = tf.math.lgamma(n_float + 1.0) - tf.math.lgamma(k + 1.0) - tf.math.lgamma(n_float - k + 1.0)
    binom = tf.exp(log_binom)
    t_col = tf.expand_dims(t, -1) 
    k_row = tf.expand_dims(k, 0)
    term1 = tf.pow(t_col, k_row)
    term2 = tf.pow(1.0 - t_col, n_float - k_row)
    basis = binom * term1 * term2 
    return basis


class Efficiency_Bernstein_Factorized(zfit.pdf.BasePDF):
    def __init__(self, obs,coef_acc_2d, coef_acc_phi, nx_acc, ny_acc, n_phi_acc, coef_reco_2d, coef_reco_phi, nx_reco, ny_reco, n_phi_reco,
                 name="Full_Efficiency_Model"):
        """
        Modelo Completo: Aceptancia * Eficiencia de Reconstrucción.
        Cada parte factorizada en 2D(cosL, cosK) * 1D(phi).
        """
        params = {
            'c_acc_2d': zfit.Parameter(f"c_a2d_{name}", tf.cast(coef_acc_2d, tf.float64), floating=False),
            'c_acc_phi': zfit.Parameter(f"c_aphi_{name}", tf.cast(coef_acc_phi, tf.float64), floating=False),
            'c_reco_2d': zfit.Parameter(f"c_r2d_{name}", tf.cast(coef_reco_2d, tf.float64), floating=False),
            'c_reco_phi': zfit.Parameter(f"c_rphi_{name}", tf.cast(coef_reco_phi, tf.float64), floating=False),
        }
        
        # Guardamos los grados de los polinomios
        self.nx_acc, self.ny_acc = nx_acc, ny_acc
        self.n_phi_acc = n_phi_acc
        self.nx_reco, self.ny_reco = nx_reco, ny_reco
        self.n_phi_reco = n_phi_reco
        
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x):
        vars_list = z.unstack_x(x)
        cos_l, cos_k, phi = vars_list[0], vars_list[1], vars_list[2]

        # [-1, 1] -> [0, 1] y [-pi, pi] -> [0, 1]
        tx = 0.5 * (cos_l + 1.0)
        ty = 0.5 * (cos_k + 1.0)
        t_phi = (phi + np.pi) / (2.0 * np.pi)
        # ======================================================
        # ACEPTANCIA
        # ======================================================
        # Bases
        Bx_acc = tf_bernstein_basis_vectorized(self.nx_acc, tx)
        By_acc = tf_bernstein_basis_vectorized(self.ny_acc, ty)
        Bphi_acc = tf_bernstein_basis_vectorized(self.n_phi_acc, t_phi)
        
        # 2D part
        c_acc_2d_mat = tf.reshape(self.params['c_acc_2d'], (self.nx_acc + 1, self.ny_acc + 1))
        acc_2d = tf.einsum('mi,mj,ij->m', Bx_acc, By_acc, c_acc_2d_mat)
        # 1D part
        acc_phi = tf.einsum('mk,k->m', Bphi_acc, self.params['c_acc_phi'])
        # Total Acceptance
        total_acc = acc_2d * acc_phi

        # ======================================================
        # EFICIENCIA DE RECONSTRUCCIÓN
        # ======================================================
        # Bases
        Bx_reco = tf_bernstein_basis_vectorized(self.nx_reco, tx)
        By_reco = tf_bernstein_basis_vectorized(self.ny_reco, ty)
        Bphi_reco = tf_bernstein_basis_vectorized(self.n_phi_reco, t_phi)
        
        # 2D part
        c_reco_2d_mat = tf.reshape(self.params['c_reco_2d'], (self.nx_reco + 1, self.ny_reco + 1))
        reco_2d = tf.einsum('mi,mj,ij->m', Bx_reco, By_reco, c_reco_2d_mat)
        # 1D part
        reco_phi = tf.einsum('mk,k->m', Bphi_reco, self.params['c_reco_phi'])
        # Total Reco Efficiency
        total_reco = reco_2d * reco_phi

        # ======================================================
        # EFICIENCIA FINAL
        # ======================================================
        return tf.maximum(total_acc * total_reco, 1e-15)
