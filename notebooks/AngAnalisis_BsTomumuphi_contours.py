#---------------#
#---------------#
#### IMPORTS ####
#---------------#
#---------------#

import os 
import sys
import pandas as pd
import numpy as np
import copy
import random

import matplotlib.pyplot as plt
import scipy
import math
import scipy.optimize
import json
#import uproot3 as uproot
import awkward as ak
import xgboost as xgb
import zfit

#sys.path.append('/cms/home/rreyes/Bphysics/btosll/analysis/scripts/')
sys.path.append('/home/jrggquantum/Desktop/Cinves/Tools/')

import customPDFs
import common_tools
import mass_models
import plot_tools
import tools
import selection_cuts
import ToyMC_tool_functions
import mplhep as hep 

from matplotlib import colors as pltcolors
hep.style.use('CMS')
plt.style.use(hep.style.CMS)
plt.rcParams['figure.figsize'] = [10,8]
plt.rcParams['font.size'] = 25

plt.figure()
plt.close()

plt.rcParams.update({'figure.figsize':[10,8]})
plt.rcParams.update({'font.size':25})

from hepstats.splot import compute_sweights

import tensorflow as tf
#from tensorflow.experimental import numpy as tnp
import zfit.z.numpy as znp

version =  zfit.__version__.split('.')
if int(version[1])>=5:
    from zfit import z
else:
    from zfit import ztf as z

from platform import python_version    

if '3.8' in python_version():
    py_v = 38
    import SLSQPv2 as SLSQP_zfit
elif '3.7' in python_version():
    py_v = 37
    import SLSQP_zfit
elif '3.12' in python_version():
    py_v = 38
    import SLSQPv3 as SLSQP_zfit

import zfit.minimizers.baseminimizer as bm
BaseMinimizer = bm.BaseMinimizer

from zfit.minimizers.baseminimizer import minimize_supports as zfit_minimize_supports
#print(minimize_supports)
#tf.compat.v1.disable_eager_execution()
from iminuit import Minuit
from zfit.core.integration import mc_integrate


#-----------------#
#-----------------#
#### MAIN CODE ####
#-----------------#
#-----------------#

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fit MC mass and Data mass")

    parser.add_argument("--era", 
                        default="2022_2023", type=str, 
                        help="era to run examples pre2023, post2023, 2022, 2023, 2022_2023")
    parser.add_argument("--output_dir", 
                        default="output", type=str, 
                        help="Directory to store results.")
    parser.add_argument("--input_dir", 
                        default="input", type=str, 
                        help="Directory to load some set.")
    parser.add_argument("--Data", 
                        default="True", type=str, 
                        help="If we load experimental data (True) or pseudo-data (False).")
    
    parser.add_argument("--json_yields_pre", 
                        help="Path of the json which have yields of resonant and nonresonant channels (preBPix 2023)")
    parser.add_argument("--json_yields_post", 
                        help="Path of the json which have yields of resonant and nonresonant channels (postBPix 2023)")

    parser.add_argument("--json_yields", 
                        help="Path of the json which have yields of resonant and nonresonant channels (2022)")
    
    parser.add_argument("--FL_value", 
                        default=0.7, type=float, 
                        help="value of F_L")
    parser.add_argument("--A6_value", 
                        default=0.0, type=float, 
                        help="value of A_6")
    parser.add_argument("--n_bin", 
                        default=1, type=int, 
                        help="bin used (1 - 6)")
    
    parser.add_argument("--short_pdf",
                        default="True", type=str,
                        help="This option say us if we use the simplified or complete pdf (True,False)")
    parser.add_argument("--qsq_all",
                        default="False", type=str,
                        help="This option works to calculate parametrer for all q square or for bin of q square (True,False)")
    parser.add_argument("--SLSQP",
                        default="False", type=str,
                        help="This option makes the minimization works with SLSQP or Minuit (True,False)")
    parser.add_argument("--tanh_trans",
                        default="True", type=str,
                        help="This option makes the minimization works with Minuit and use a tanh transformation (True,False)")
    parser.add_argument("--arctan_trans",
                        default="False", type=str,
                        help="This option makes the minimization works with Minuit and use a arctan transformation (True,False)")
    parser.add_argument("--tanh_ComposedParams",
                        default="False", type=str,
                        help="This option makes the minimization works with Minuit and use a tanh transformation and use composed parameters (True,False)")
    parser.add_argument("--ang_eff",
                        default="True", type=str,
                        help="This option say us if we multiplied the angular efficiencies with the angular pdf (True,False)")
    parser.add_argument("--ang_bkg",
                        default="True", type=str,
                        help="This option say us if we multiplied the angular backgrounds with the angular pdf (bkg) (True,False)")
    
    parser.add_argument("--allTriangle",
                        default="True", type=str,
                        help="This option say us if we print all physical space or we do a zoom into contour")
    
    
    
    args_ = parser.parse_args()
    argparse_dict = vars(args_)
    args = common_tools.dotdict(argparse_dict)

    ### Un-packing arguments!
    era            = args.era
    output_dir     = args.output_dir
    input_dir      = args.input_dir
    
    FL_value       = args.FL_value
    A6_value       = args.A6_value
    n_toys         = args.n_toys
    n_bin          = args.n_bin

    ### Definig list and paths usefulls
    bins = [[1.1, 4.0], [4.0, 6.0], [6.0, 8.0], [11.0, 12.5], [15.0, 17.0], [17.0, 23.0], [8.0, 11.0], [12.5, 15.0]]
    #bins = [[1.1, 4.0], [4.0, 6.0], [6.0, 8.0], [11.0, 12.5], [15.0, 17.0], [17.0, 23.0]]
    eras = ['2022', 'pre2023', 'post2023']

    ### In this point the main code begins
    FL_val = float(FL_value)
    A6_val = float(A6_value)
            
    print('Valor actual de FL:', FL_val, '\n')
    print('Valor actual de A6:', A6_val, '\n')

    print('Bin numero :', n_bin, '\n')

    # Lists to save parameters and their errors
    FL_min = list()
    A6_min = list()
    FL_min_error = list()
    A6_min_error = list()

    FL_min_trans = list()
    A6_min_trans = list()
    FL_min_error_trans = list()
    A6_min_error_trans = list()

    FL_contour_lower = list()
    FL_contour_upper = list()
    A6_contour_lower = list()
    A6_contour_upper = list()

    # Lists to save models and data sets
    models_list = list()
    data_list = list()

    # Observables of the pdf
    cos_k = zfit.Space(f'cos_k_{FL_val}_{A6_val}', limits=(-1, 1))
    cos_l = zfit.Space(f'cos_l_{FL_val}_{A6_val}', limits=(-1, 1))
    mass  = zfit.Space(f'mass_{FL_val}_{A6_val}', [5.1,5.7])
    obs = cos_k * cos_l  # The order of "product" is important
    complete_obs = mass * cos_k * cos_l

    # Parameters of simplified pdf (sample parameters and fit parameters)
    if args.SLSQP == 'True':
        FL = zfit.Parameter(f'FL_{FL_val}_{A6_val}', FL_val, 0, 1)
        A6 = zfit.Parameter(f'A6_{FL_val}_{A6_val}', A6_val, -1.0, 1.0)
    elif args.tanh_trans == 'True':
        # Transformation of FL and FL_hat using tanh
        FL_trans = math.atanh(2 * FL_val - 1.)
        A6_trans = math.atanh(A6_val/(1 - FL_val))

        FL = zfit.Parameter(f'FL_{FL_val}_{A6_val}', FL_trans)
        A6 = zfit.Parameter(f'A6_{FL_val}_{A6_val}', A6_trans)
    elif args.arctan_trans == 'True':
        # Transformation of FL and FL_hat using arc tan
        FL_trans = math.tan(math.pi * (FL_val - 0.5))
        A6_trans = math.tan((math.pi * A6_val)/(2 * (1 - FL_val)))

        FL = zfit.Parameter(f'FL_{FL_val}_{A6_val}', FL_trans)
        A6 = zfit.Parameter(f'A6_{FL_val}_{A6_val}', A6_trans)
    elif args.tanh_ComposedParams == 'True':
        fl = zfit.Parameter(f'fl_{FL_val}_{A6_val}', FL_val)
        a6 = zfit.Parameter(f'a6_{FL_val}_{A6_val}', A6_val)
        FL = zfit.param.ComposedParameter(f'FL_{FL_val}_{A6_val}', 
                                          lambda fl: tf.math.atanh(2*fl - 1.), params=[fl])
        A6 = zfit.param.ComposedParameter(f'A6_{FL_val}_{A6_val}', 
                                          lambda a6, fl: tf.math.atanh(a6/(1 - fl)), params=[a6,fl])

    for i in range(len(eras)):

        if eras[i] == '2022':
            if n_bin > 0 and n_bin <= 6:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularEffs2022/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularBkg2022/'
            elif n_bin == 7 or n_bin == 8:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngEff_RC/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngBkg_RC/'
            mass_fit_path = '/home/jrggquantum/Desktop/Cinves/Yields/Data2022_corrected/'
        elif eras[i] == 'pre2023':
            if n_bin > 0 and n_bin <= 6:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularEffspre2023/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularBkgpre2023/'
            elif n_bin == 7 or n_bin == 8:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngEff_RC/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngBkg_RC/'
            mass_fit_path = '/home/jrggquantum/Desktop/Cinves/Yields/Data2023/'
        elif eras[i] == 'post2023':
            if n_bin > 0 and n_bin <= 6:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularEffspost2023/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/Ang_Eff_Bkg/angularBkgpost2023/'
            elif n_bin == 7 or n_bin == 8:
                ang_eff_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngEff_RC/'
                ang_bkg_path = '/home/jrggquantum/Desktop/Cinves/AngularFit/AngBkg_RC/'
            mass_fit_path = '/home/jrggquantum/Desktop/Cinves/Yields/Data2023/'

        eff_list_k = ['angularEffTotal_scipy_costhetak_bin1.json','angularEffTotal_scipy_costhetak_bin2.json',
                      'angularEffTotal_scipy_costhetak_bin3.json','angularEffTotal_scipy_costhetak_bin5.json',
                      'angularEffTotal_scipy_costhetak_bin7.json','angularEffTotal_scipy_costhetak_bin8.json',
                      f'angularEff_model_total_costhetak_bin4_{eras[i]}.json',
                      f'angularEff_model_total_costhetak_bin6_{eras[i]}.json']
        eff_list_l = ['angularEffTotal_scipy_costhetal_bin1.json','angularEffTotal_scipy_costhetal_bin2.json',
                      'angularEffTotal_scipy_costhetal_bin3.json','angularEffTotal_scipy_costhetal_bin5.json',
                      'angularEffTotal_scipy_costhetal_bin7.json','angularEffTotal_scipy_costhetal_bin8.json',
                      f'angularEff_model_total_costhetal_bin4_{eras[i]}.json',
                      f'angularEff_model_total_costhetal_bin6_{eras[i]}.json']
        bkg_list_k = ['angularBkg_costhetak_bin1.json','angularBkg_costhetak_bin2.json',
                      'angularBkg_costhetak_bin3.json','angularBkg_costhetak_bin5.json',
                      'angularBkg_costhetak_bin7.json','angularBkg_costhetak_bin8.json',
                      f'PDFparams_AngBkg_costhetak_{eras[i]}_ResonantJpsi.json',
                      f'PDFparams_AngBkg_costhetak_{eras[i]}_ResonantPsi.json']
        bkg_list_l = ['angularBkg_costhetal_bin1.json','angularBkg_costhetal_bin2.json',
                      'angularBkg_costhetal_bin3.json','angularBkg_costhetal_bin5.json',
                      'angularBkg_costhetal_bin7.json','angularBkg_costhetal_bin8.json',
                      f'PDFparams_AngBkg_costhetal_{eras[i]}_ResonantJpsi.json',
                      f'PDFparams_AngBkg_costhetal_{eras[i]}_ResonantPsi.json']
        mass_params_list = [f'PDFparams_data_{eras[i]}_1.json',f'PDFparams_data_{eras[i]}_2.json',
                            f'PDFparams_data_{eras[i]}_3.json',f'PDFparams_data_{eras[i]}_4.json',
                            f'PDFparams_data_{eras[i]}_5.json',f'PDFparams_data_{eras[i]}_6.json',
                            f'PDFparams_data_{eras[i]}_JP.json',
                            f'PDFparams_data_{eras[i]}_P2.json']                      
                            
        # Construction of angular efficiency and background
        # Eff and Bkg of cos(theta_k)
        ang_json_k = tools.read_json(ang_eff_path + eff_list_k[n_bin - 1])
        params_ang_k = list()
        if n_bin > 0 and n_bin <= 6:
            for k in range(len(ang_json_k['parameters'])):
                par_k = ang_json_k['parameters'][f'c{k}']
                params_ang_k.append(par_k)
        elif n_bin == 7 or n_bin == 8:
            for k in range(len(ang_json_k['total']['parameters'])):
                par_k = ang_json_k['total']['parameters'][f'c{k}']
                params_ang_k.append(par_k)
        ang_eff_k = customPDFs.bernstein(coeffs=params_ang_k, obs=cos_k)

        bkg_json_k = tools.read_json(ang_bkg_path + bkg_list_k[n_bin - 1])
        params_bkg_k = list()
        for k in range(len(bkg_json_k['parameters'])):
            if n_bin > 0 and n_bin <= 6:
                par_k = bkg_json_k['parameters'][f'c_{k}']
            elif n_bin == 7 or n_bin == 8:
                par_k = bkg_json_k['parameters'][f'c{k}']
            params_bkg_k.append(par_k)            
        ang_bkg_k = customPDFs.bernstein(coeffs=params_bkg_k, obs=cos_k)

        # Eff and Bkg of cos(theta_l)
        ang_json_l = tools.read_json(ang_eff_path + eff_list_l[n_bin - 1])
        params_ang_l = list()
        if n_bin > 0 and n_bin <= 6:
            for k in range(len(ang_json_l['parameters'])):
                par_l = ang_json_l['parameters'][f'c{k}']
                params_ang_l.append(par_l)
        elif n_bin == 7 or n_bin == 8:
            for k in range(len(ang_json_l['total']['parameters'])):
                par_l = ang_json_l['total']['parameters'][f'c{k}']
                params_ang_l.append(par_l)
        ang_eff_l = customPDFs.bernstein(coeffs=params_ang_l, obs=cos_l)

        bkg_json_l = tools.read_json(ang_bkg_path + bkg_list_l[n_bin - 1])
        params_bkg_l = list()
        for k in range(len(bkg_json_l['parameters'])):
            if n_bin > 0 and n_bin <= 6:
                par_k = bkg_json_l['parameters'][f'c_{k}']
            elif n_bin == 7 or n_bin == 8:
                par_k = bkg_json_l['parameters'][f'c{k}']
            params_bkg_l.append(par_k)      
        ang_bkg_l = customPDFs.bernstein(coeffs=params_bkg_l, obs=cos_l)

        # Construction of mass model PDF
        mass_json = tools.read_json(mass_fit_path + mass_params_list[n_bin - 1])

        mass = zfit.Space(f'mass_{FL_val}_{A6_val}', [5.1, 5.7])
        mass_red = zfit.Space(f'mass_red_{FL_val}_{A6_val}', [5.2, 5.5])
        complete_obs_red = mass_red * cos_k * cos_l

        int_dom = zfit.Space(f'mass_{FL_val}_{A6_val}', [5.2, 5.5])

        sub_pdfs = ToyMC_tool_functions.list_of_pdfs(mass_json, create_params = True, obs = mass_red, era = eras[i], bin = n_bin, extra='red')
        #print(sub_pdfs[1][0].is_extended)
        sub_pdfs_complete = ToyMC_tool_functions.list_of_pdfs(mass_json, create_params = True, obs = mass, era = eras[i], bin = n_bin, extra='complete')
        if n_bin > 0 and n_bin <= 6:
            mass_signal_com, mass_bkg_com = ToyMC_tool_functions.build_signal_bkg(mass_json, sub_pdfs_complete, obs = mass, era = eras[i], bin = n_bin)
            mass_signal, mass_bkg, complete_mass_model, global_frac = ToyMC_tool_functions.build_signal_bkg_red(pdf_info=mass_json, sub_pdfs=sub_pdfs, 
                                                                                                                    sub_pdfs_complete=sub_pdfs_complete, 
                                                                                                                    signal_complete=mass_signal_com, 
                                                                                                                    bkg_complete=mass_bkg_com, 
                                                                                                                    int_dom=int_dom, 
                                                                                                                    obs_red=mass_red, obs_complete=mass, 
                                                                                                                    era = eras[i], bin = n_bin)
        elif n_bin == 7 or n_bin == 8:
            mass_signal_com, mass_bkg_com = ToyMC_tool_functions.build_signal_bkg_RC(mass_json, sub_pdfs_complete, obs = mass, era = eras[i], bin = n_bin)
            mass_signal, mass_bkg, complete_mass_model, global_frac = ToyMC_tool_functions.build_signal_bkg_RC_red(pdf_info=mass_json, sub_pdfs=sub_pdfs, 
                                                                                                                    sub_pdfs_complete=sub_pdfs_complete, 
                                                                                                                    signal_complete=mass_signal_com, 
                                                                                                                    bkg_complete=mass_bkg_com, 
                                                                                                                    int_dom=int_dom, 
                                                                                                                    obs_red=mass_red, obs_complete=mass, 
                                                                                                                    era = eras[i], bin = n_bin)

        # Product of efficiencies and backgrounds PDF's Built
        ang_eff = zfit.pdf.ProductPDF([ang_eff_k, ang_eff_l], obs=obs, name='Ang_eff')
        ang_bkg = zfit.pdf.ProductPDF([ang_bkg_k, ang_bkg_l], obs=obs, name='Ang_bkg')

        # Construction of complete PDF
        if args.SLSQP == 'True':
            pdf_theorical = ToyMC_tool_functions.DecayRate_BsTomumuphi_short(FL=FL, A6=A6, obs=obs)
        elif args.tanh_trans == 'True':
            pdf_theorical = ToyMC_tool_functions.DecayRate_BsTomumuphi_short_minuit_tanh(FL=FL, A6=A6, obs=obs)
        elif args.arctan_trans == 'True':
            pdf_theorical = ToyMC_tool_functions.DecayRate_BsTomumuphi_short_minuit_arctan(FL=FL, A6=A6, obs=obs)
        elif args.tanh_ComposedParams == 'True':
            pdf_theorical = ToyMC_tool_functions.DecayRate_BsTomumuphi_short(FL=FL, A6=A6, obs=obs)
                                
        signal_form = zfit.pdf.ProductPDF([mass_signal, pdf_theorical, ang_eff], obs=complete_obs_red)
        bkg_form = zfit.pdf.ProductPDF([mass_bkg, ang_bkg], obs=complete_obs_red)

        # Finally, we load the data of each era (or ToyMC)
        if args.Data == 'True':
            if n_bin > 0 and n_bin <= 6:
                data = pd.read_json(f'/home/jrggquantum/Desktop/Cinves/AngularFit/Filtered_Data/Angular_data_{eras[i]}_bin_{n_bin}.json', orient='records')
            elif n_bin == 7:
                data = pd.read_json(f'/home/jrggquantum/Desktop/Cinves/AngularFit/Filtered_Data/Angular_data_{eras[i]}_bin_Jpsi.json', orient='records')
            elif n_bin == 8:
                data = pd.read_json(f'/home/jrggquantum/Desktop/Cinves/AngularFit/Filtered_Data/Angular_data_{eras[i]}_bin_Psi.json', orient='records')
            data_df = data[['massB','costhetak','costhetal']]
            # Filter to use mass reduced space
            data_df = data_df[(data_df['massB']>(5.2)) & (data_df['massB']<(5.5))]
            n = len(data_df)
            data_df = data_df.rename(columns={
                                    "massB": f'mass_red_{FL_val}_{A6_val}',
                                    "costhetak": f'cos_k_{FL_val}_{A6_val}',
                                    "costhetal": f'cos_l_{FL_val}_{A6_val}'
                                    })
            zfit_data = zfit.Data.from_pandas(data_df, obs=complete_obs_red)
            data_list.append(zfit_data)
        else:
            # Needed resonant channel implementation
            path = input_dir
            sample_name = f"ToyMC_short_pdf_sample_Teff_bkg_{eras[i]}_bin_{n_bin}.json"
                        
            with open(os.path.join(path, sample_name), 'r') as f:
                sample_loaded = np.array(json.load(f)["sample"])

            sampler = zfit.Data.from_numpy(obs=complete_obs_red, array=sample_loaded)
            n = len(sample_loaded)
            data_list.append(sampler)

        # We extended the signal and bkg pdfs before joined their
        Ys = zfit.Parameter(f'SigYield_{FL_val}_{A6_val}_{i}', n*global_frac[0], floating = False)
        Yb = zfit.Parameter(f'BkgYield_{FL_val}_{A6_val}_{i}', n*global_frac[1], floating = False)
        #Ys = zfit.Parameter(f'SigYield_{FL_val}_{A6_val}_{i}', n*frac_sig, n*(frac_sig - 0.05), n*(frac_sig + 0.05), floating = True)
        #Yb = zfit.Parameter(f'BkgYield_{FL_val}_{A6_val}_{i}', n*frac_bkg, n*(frac_bkg - 0.05), n*(frac_bkg + 0.05), floating = True)
        #Ys = zfit.Parameter(f'SigYield_{FL_val}_{A6_val}_{i}', n*0.5, n*0.01, n*1., floating = True)
        #Yb = zfit.Parameter(f'BkgYield_{FL_val}_{A6_val}_{i}', n*0.5, n*0.01, n*1., floating = True)

        signal_ext = signal_form.create_extended(yield_=Ys, name=f"Ang_Signal_{FL_val}_{A6_val}_{i}")
        bkg_ext = bkg_form.create_extended(yield_=Yb, name=f"Ang_Background_{FL_val}_{A6_val}_{i}")

        pdf_model = zfit.pdf.SumPDF([signal_ext, bkg_ext], name=f'Ang_PDFComplete_{FL_val}_{A6_val}_{i}')
        models_list.append(pdf_model)

                        
    # Definition of minimizer (it means, defined the method of minimization)
    if args.SLSQP == 'True':
        constraint_slsqp = ToyMC_tool_functions.create_constraint_Bs_short(pdf_model, yield_index = 0 , n=n , a6_index=2, fl_index=1)
        minimizer = SLSQP_zfit.SLSQP(constraints=constraint_slsqp, verbosity = 6)
    else:
        minimizer = zfit.minimize.Minuit()

    print('Calculamos la funcion negative log likelihood')
    nll = zfit.loss.ExtendedUnbinnedNLL(model = models_list, data = data_list)

    if args.SLSQP == 'True':
        try:
            result = minimizer.minimize(nll)

            print('converged:', result.converged)
            print('validity:', result.valid)

            if not result.converged or not result.valid:
                print(f"Toy 1: minimización no válida.")
                                
            params = result.params
            FL_value = params[f'FL_{FL_val}_{A6_val}']['value']
            A6_value = params[f'A6_{FL_val}_{A6_val}']['value']
            N_value = params[f'yield_{FL_val}_{A6_val}']['value']

            if any(np.isnan(p) or np.isinf(p) for p in [FL_value, A6_value]):
                print(f"Toy 1: parámetros NaN o Inf detectados.")

        except Exception as e: 
            print(e)
    else:
        try:
            print('Minuit Branch')
            result = minimizer.minimize(nll)

            print('converged:', result.converged)
            print('validity:', result.valid)

            if result.converged == True:
                print(result.covariance())


            if not result.converged or not result.valid:
                print(f"Toy 1: minimización no válida.")

            results = result.hesse()
            params = result.params
            print(params)
            #print(dir(result))

            if args.tanh_ComposedParams == 'True':
                fl_value = params[f'fl_{FL_val}_{A6_val}']['value']
                a6_value = params[f'a6_{FL_val}_{A6_val}']['value']
                #N_value = params[f'yield_{FL_val}_{A6_val}_{i}']['value']

                if any(np.isnan(p) or np.isinf(p) for p in [fl_value, a6_value]):
                    print(f"Toy 1: parámetros NaN o Inf detectados.")
            else:
                FL_value = params[f'FL_{FL_val}_{A6_val}']['value']
                A6_value = params[f'A6_{FL_val}_{A6_val}']['value']
                #N_value = params[f'yield_{FL_val}_{A6_val}']['value']

                if any(np.isnan(p) or np.isinf(p) for p in [FL_value, A6_value]):
                    print(f"Toy 1: parámetros NaN o Inf detectados.")

            fl_value_current = FL.value()
            a6_value_current = A6.value()
                            
            def nll_wrap(fl_val, a6_val):
                FL.set_value(fl_val)
                A6.set_value(a6_val)
                                
                return nll.value()

            #m = Minuit(nll_wrap, a6_val=a6_value_current, fl_val=fl_value_current)
            m = Minuit(nll_wrap, fl_val=fl_value_current, a6_val=a6_value_current)
            m.errordef = 0.5
            m.migrad()
            m.hesse()

            # Conorno que encierra a un sigma
            contour = m.mncontour('fl_val', 'a6_val', cl=0.3935, size=50)

            # Errores MINOS por separado
            m.minos('fl_val', 'a6_val')

            # Acceso a los errores
            merror_FL = m.merrors['fl_val']
            merror_A6 = m.merrors['a6_val']
                                
        except Exception as e: 
            print(e)
                                     
    print('La minimizacion termino')

    # Json with params values of each era model
    for p in range(len(models_list)):
        tools.save_pdf_info_as_json(models_list[p], output_dir + f'PDFparams_{eras[p]}_bin_{n_bin}.json')

    if args.SLSQP == 'True':
        vec_params = [FL_value, A6_value]

    #Calculo numerico de la matriz hessiana
    print('Se calculara la Hessiana')
    if args.SLSQP == 'True':
        Hess = ToyMC_tool_functions.numerical_hessian(nll, vec_params, epsilon=1e-6)
        #print(Hess)
        try:
            inv_H = np.linalg.inv(Hess)
            #print(inv_H)
            errors = np.sqrt(np.diag(inv_H))

            FL_min.append(FL_value)
            A6_min.append(A6_value)
            FL_min_error.append(errors[1])
            A6_min_error.append(errors[2])
        except Exception as e: 
            print(e)
    else:
        if args.tanh_trans == 'True':
            # Errores MINOS de la minimizacion de iminuit
            FL_value = m.values['fl_val']
            FL_minos_upper = abs(merror_FL.upper)
            FL_minos_lower = abs(merror_FL.lower)
            A6_value = m.values['a6_val']
            A6_minos_upper = abs(merror_A6.upper)
            A6_minos_lower = abs(merror_A6.lower)

            A6_value_error = [[A6_minos_lower], [A6_minos_upper]]
            FL_value_error = [[FL_minos_lower], [FL_minos_upper]]

            # Inverse transformation (tanh) to obtain FL in base of FL_hat (considering asymmetric errors)
            FL_real = 0.5 + (math.tanh(FL_value))/2
            A6_real = (1 - FL_real) * (math.tanh(A6_value))
            FL_real_error_upper = abs((1/2) * (1 - math.tanh(FL_value)**2)) * FL_minos_upper
            FL_real_error_lower = abs((1/2) * (1 - math.tanh(FL_value)**2)) * FL_minos_lower
            A6_real_error_upper = (((-1/2) * ((1 - math.tanh(FL_value)**2) * (math.tanh(A6_value))) * FL_minos_upper)**2 + 
                                 ((1 - FL_real) * (1 - math.tanh(A6_value)**2) * A6_minos_upper)**2)**(1/2)
            A6_real_error_lower = (((-1/2) * ((1 - math.tanh(FL_value)**2) * (math.tanh(A6_value))) * FL_minos_lower)**2 + 
                                 ((1 - FL_real) * (1 - math.tanh(A6_value)**2) * A6_minos_lower)**2)**(1/2)
                            
            A6_real_error = [[A6_real_error_lower], [A6_real_error_upper]]
            FL_real_error = [[FL_real_error_lower], [FL_real_error_upper]]
        elif args.arctan_trans == 'True':
            FL_value = params[f'FL_{FL_val}_{A6_val}_{i}']['value']
            FL_value_error = params[f'FL_{FL_val}_{A6_val}_{i}']['hesse']['error']
            A6_value = params[f'A6_{FL_val}_{A6_val}_{i}']['value']
            A6_value_error = params[f'A6_{FL_val}_{A6_val}_{i}']['hesse']['error']

            # Inverse transformation (arc tan) to obtain FL in base of FL_hat
            FL_real = 0.5 + (math.atan(FL_value))/math.pi
            A6_real = 2 * (1 - FL_real) * (math.atan(A6_value)/math.pi)
            FL_real_error = abs((1)/(math.pi * (1 + FL_value * FL_value))) * FL_value_error
            A6_real_error = ((-2 * ((math.atan(A6_value))/((math.pi**2) * (1 + FL_value**2))) * FL_value_error)**2 + 
                             (2 * (1 - FL_real) * ((A6_value_error)/(math.pi * (1 + A6_value**2))))**2)**(1/2)
        elif args.tanh_ComposedParams == 'True':
            FL_real = params[f'fl_{FL_val}_{A6_val}_{i}']['value']                            
            FL_real_error = params[f'fl_{FL_val}_{A6_val}_{i}']['hesse']['error']
            A6_real = params[f'a6_{FL_val}_{A6_val}_{i}']['value']
            A6_real_error = params[f'a6_{FL_val}_{A6_val}_{i}']['hesse']['error']

    FL_min.append(FL_real)
    FL_min_error.append(FL_real_error)
    A6_min.append(A6_real)
    A6_min_error.append(A6_real_error)
    FL_min_trans.append(FL_value)
    A6_min_trans.append(A6_value)
    FL_min_error_trans.append(FL_value_error)
    A6_min_error_trans.append(A6_value_error)

    # Contours using Minuit
    y_contour, x_contour = zip(*contour)
    fl_contour = ToyMC_tool_functions.Fl_from_x(y_contour)
    a6_contour = ToyMC_tool_functions.A6_from_xy(y_contour, x_contour)

    x_max = max(x_contour)
    x_min = min(x_contour)
    y_max = max(y_contour)
    y_min = min(y_contour)

    fl_max = max(fl_contour)
    fl_min = min(fl_contour)
    a6_max = max(a6_contour)
    a6_min = min(a6_contour)

    # Crear el espacio transformado (x, y)
    #x_vals = np.linspace(-5, 5, 200)
    #y_vals = np.linspace(-5, 5, 200)
    x_vals = np.linspace(-12, 12, 500)
    y_vals = np.linspace(-12, 12, 500)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Transformaciones
    Fl = ToyMC_tool_functions.Fl_from_x(X)
    A6 = ToyMC_tool_functions.A6_from_xy(X,Y)

    # Crear figura con dos subgráficas
    x_flat = X.ravel()
    y_flat = Y.ravel()
    Fl_flat = Fl.ravel()
    A6_flat = A6.ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    if n_bin > 0 and n_bin <= 6:
        plt.suptitle(f'Bin {n_bin}')
    elif n_bin == 7:
        plt.suptitle(r'Bin $J/\psi$')
    elif n_bin == 8:
        plt.suptitle(r'Bin $\psi (2S)$')
    plt.tight_layout() #Esta linea ajusta los titulos para que no exista un traslape

    # Espacio físico (Fl, A6)
    ax1.scatter(A6, Fl,  c='lightgray', s=0.5, label="Puntos válidos")
    ax1.plot( a6_contour, fl_contour, 'r', lw=2, label="Contorno Fisico")

    ax1.scatter(A6_real, FL_real, c='blue', s=22, label = 'Punto minimizado')
    
    if args.allTriangle != 'True':
        ax1.text(A6_real - (A6_real-min(a6_contour))/2, FL_real + (max(fl_contour)-FL_real)/2,
                f'$F_L = {round(FL_real,4)} \:\: (+/-) \:\: {round(max(fl_contour)-FL_real,4)}/{round(FL_real-min(fl_contour),4)}$', 
                fontsize=12, color='blue')
        ax1.text(A6_real - (A6_real-min(a6_contour))/2, FL_real - (FL_real-min(fl_contour))/2,
                f'$A_6 = {round(A6_real,4)} \:\: (+/-) \:\: {round(max(a6_contour)-A6_real,4)}/{round(A6_real-min(a6_contour),4)}$', 
                fontsize=12, color='blue')

    ax1.axvline(x= min(a6_contour), color = 'red', linestyle='--')
    ax1.axvline(x= max(a6_contour), color = 'red', linestyle='--')
    ax1.axhline(y= min(fl_contour), color = 'red', linestyle='--')
    ax1.axhline(y= max(fl_contour), color = 'red', linestyle='--')

    FL_contour_lower.append(fl_min)
    FL_contour_upper.append(fl_max)
    A6_contour_lower.append(a6_min)
    A6_contour_upper.append(a6_max)

    ax1.set_title("Physical Space ($A_6$, $F_L$)")
    ax1.set_xlabel("$A_6$", fontsize=18)
    ax1.set_ylabel("$F_L$", fontsize=18)
    fl_line = np.linspace(0.01, 0.99, 300)
    a6_bound = 1 - fl_line
    ax1.plot(a6_bound,fl_line,  'k--', lw=1)
    ax1.plot( -a6_bound,fl_line, 'k--', lw=1)
    #ax1.legend()

    if args.allTriangle == 'True':
        ax1.set_xlim(-1.05,1.05)
        ax1.set_ylim(-0.05,1.05)
    else:
        ax1.set_xlim( a6_min - (a6_max-a6_min)/4 , a6_max + (a6_max-a6_min)/4 )
        ax1.set_ylim( fl_min - (fl_max-fl_min)/4 , fl_max + (fl_max-fl_min)/4 )

    #fl_max_error = max(FL_plus_error)
    #fl_min_error = min(FL_minus_error)
    #a6_max_error = max(A6_plus_error)
    #a6_min_error = min(A6_minus_error)
    #ax1.set_xlim( a6_min_error - (a6_max_error - a6_min_error)/4 , a6_max_error + (a6_max_error - a6_min_error)/4 )
    #ax1.set_ylim( fl_min_error - (fl_max_error - fl_min_error)/4 , fl_max_error + (fl_max_error - fl_min_error)/4 )

    # Espacio transformado (x, y)
    ax2.scatter(X, Y, c='lightgray', s=0.5, label="Puntos válidos")
    ax2.plot(x_contour, y_contour, 'r', lw=2, label="Contorno Transformado")

    ax2.scatter(A6_value, FL_value, c='blue', s=22, label = 'Punto minimizado')
    ax2.errorbar(A6_value, FL_value, xerr=A6_value_error, yerr=FL_value_error, ls='none', capsize=10, color='blue', label='Error punto minimizado')

    ax2.set_title("Transformed Space ($\\hat{A}_{6}$, $\\hat{F}_{L}$)")
    ax2.set_xlabel("$\\hat{A}_{6}$", fontsize=18)
    ax2.set_ylabel("$\\hat{F}_{L}$", fontsize=18)
    ax2.tick_params(axis='y', which='major', labelsize = 17)
    #ax2.legend()
                    
    ax2.set_xlim( x_min - (x_max-x_min)/4 , x_max + (x_max-x_min)/4 )
    ax2.set_ylim( y_min - (y_max-y_min)/4 , y_max + (y_max-y_min)/4 )

    ax2.axvline(x= m.values['a6_val'] + merror_A6.lower, color = 'red', linestyle='--')
    ax2.axvline(x= m.values['a6_val'] + merror_A6.upper, color = 'red', linestyle='--')
    ax2.axhline(y= m.values['fl_val'] + merror_FL.lower, color = 'red', linestyle='--')
    ax2.axhline(y= m.values['fl_val'] + merror_FL.upper, color = 'red', linestyle='--')

    #ax2.axvline(x= A6_value - A6_value_error, color = 'purple', linestyle='--')
    #ax2.axvline(x= A6_value + A6_value_error, color = 'purple', linestyle='--')
    #ax2.axhline(y= FL_value - FL_value_error, color = 'purple', linestyle='--')
    #ax2.axhline(y= FL_value + FL_value_error, color = 'purple', linestyle='--')

    #plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_bin_{i+1}.png"), bbox_inches='tight')
    #plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_zfit.png"), bbox_inches='tight')
    #plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_minuit_both.png"), bbox_inches='tight')
    #plt.show()

    #plt.savefig(os.path.join(output_dir, f"Contours_spaces_eff_{FL_val}_{A6_val}_bin_{i+1}_allTriangle.png"), bbox_inches='tight')

    if args.Data == 'True':
        if args.allTriangle == 'True':
            if n_bin > 0 and n_bin <= 6:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}_allTriangle.png"), bbox_inches='tight')
            elif n_bin == 7:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_Jpsi_allTriangle.png"), bbox_inches='tight')
            elif n_bin == 8:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_Psi_allTriangle.png"), bbox_inches='tight')
        else:
            if n_bin > 0 and n_bin <= 6:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}.png"), bbox_inches='tight')
            elif n_bin == 7:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_Jpsi.png"), bbox_inches='tight')
            elif n_bin == 8:
                plt.savefig(os.path.join(output_dir, f"Contours_spaces_{FL_val}_{A6_val}_simultaneous_bin_Psi.png"), bbox_inches='tight')
    else:
        if args.allTriangle == 'True':
            plt.savefig(os.path.join(output_dir, f"ContoursToyMC_spaces_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}_allTriangle.png"), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(output_dir, f"ContoursToyMC_spaces_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}.png"), bbox_inches='tight')
    
    dict_0 = dict()
    dict_0['F_L'] = FL_real
    dict_0['A_6'] = A6_real
    dict_0['F_L_contour_error'] = [abs(fl_min - FL_real), abs(FL_real - fl_max)]
    dict_0['A_6_contour_error'] = [abs(a6_min - A6_real), abs(A6_real - a6_max)]

    dict_0['F_L_trans'] = FL_value
    dict_0['A_6_trans'] = A6_value
    dict_0['F_L_error_trans'] = [FL_minos_lower, FL_minos_upper]
    dict_0['A_6_error_trans'] = [A6_minos_lower, A6_minos_upper]

    #print(dict_0)

    path = output_dir
    if args.Data == 'True':
        if n_bin > 0 and n_bin <= 6:
            file_name = f"Data_short_pdf_values_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}_contours.json"
        elif n_bin == 7:
            file_name = f"Data_short_pdf_values_{FL_val}_{A6_val}_simultaneous_bin_Jpsi_contours.json"
        elif n_bin == 8:
            file_name = f"Data_short_pdf_values_{FL_val}_{A6_val}_simultaneous_bin_Psi_contours.json"
    else:
        file_name = f"ToyMC_short_pdf_values_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}_contours.json"

    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(dict_0, file)

    dict_p = dict()
    dict_p['F_L'] = list(fl_contour)
    dict_p['A_6'] = list(a6_contour)
    dict_p['F_L_trans'] = list(y_contour)
    dict_p['A_6_trans'] = list(x_contour)   

    if args.Data == 'True':
        if n_bin > 0 and n_bin <= 6:
            file_name = f"Data_short_pdf_contour_points_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}.json"
        elif n_bin == 7:
            file_name = f"Data_short_pdf_contour_points_{FL_val}_{A6_val}_simultaneous_bin_Jpsi.json"
        elif n_bin == 8:
            file_name = f"Data_short_pdf_contour_points_{FL_val}_{A6_val}_simultaneous_bin_Psi.json"
    else:
        file_name = f"ToyMC_short_pdf_contour_points_{FL_val}_{A6_val}_simultaneous_bin_{n_bin}.json"

    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(dict_p, file)

    print('Finished!!!')