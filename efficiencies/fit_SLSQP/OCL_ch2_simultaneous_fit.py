import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import uuid
import math
import json

#sys.path.insert(0,'/home/noe-tepec/PhD/Feldman_Cousins/Scripts')
import customPDFs
import plot_tools
import tools
import zfit
import SLSQPv2 as SLSQP_zfit
#import SLSQP_zfit
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
import tensorflow.math as tfmath
import tensorflow as tf
from timeit import default_timer as timer

from zfit import z

class DecayRate_BsTomumuphi_short(zfit.pdf.BasePDF):
    def __init__(self, FL, A6, obs, name="angular_Phi_integrated"):
        params = {'FL': FL, 'A6': A6}
        super().__init__(obs=obs, params=params, name=name)

    def _pdf(self, x, norm_range):
        cos_k, cos_l = z.unstack_x(x) #El orden de cos_l, cos_k es importante, debe ser igual a como se define en obs
        FL = self.params['FL']
        A6 = self.params['A6']

        cos2_k = tf.square(cos_k)
        cos2_l = tf.square(cos_l)

        term1 = (9. / 32.) * (1. - FL) * (1. - cos2_k) * (1. + cos2_l)
        term2 = (9. / 8.) * FL * cos2_k * (1. - cos2_l)
        term3 = (9. / 16.) * A6 * (1. - cos2_k) * cos_l

        pdf = term1 + term2 + term3
        tf.debugging.assert_rank(pdf, 1, message="PDF debe ser 1D")
        pdf = tf.where(tf.math.is_nan(pdf), tf.zeros_like(pdf), pdf)
        tf.debugging.check_numerics(pdf, message="NaN/Inf en la PDF del modelo Bs->mumu phi")

        return pdf

# Definimos las constricciones que existe (restricciones a los parametros A6 y FL)

def Constraint_BsTomumuphi_short():
    
    F_L = FL.value()
    A_6 = A6.value()

    lower_bound = F_L - 1
    upper_bound = 1 - F_L

    # Penalización si A_6 está fuera del rango permitido
    #penalty_0 = tf.constant(0.0, dtype=tf.float64)
    #if A_6 < lower_bound:
    #    penalty_1 = tf.square((A_6 - lower_bound) / 0.1)
    #else:
    #    penalty_1 = tf.constant(0.0, dtype=tf.float64)

    #if A_6 > upper_bound:
    #    penalty_2 = tf.square((upper_bound - A_6) / 0.1)
    #else:
    #    penalty_2 = tf.constant(0.0, dtype=tf.float64)

    penalty_1 = tf.where(A_6 < lower_bound,
                         tf.square((A_6 - lower_bound) / 0.1),
                         0.0)

    penalty_2 = tf.where(A_6 > upper_bound,
                         tf.square((upper_bound - A_6) / 0.1),
                         0.0)

    penalty = penalty_1 + penalty_2
    
    return penalty


def create_constraint_Bs_short(model, a6_index=False, fl_index=False):

    #First look for the indices of the POIs
    if (type(a6_index)!=int and a6_index==False) or (type(fl_index)!=int and fl_index==False):
        #afb_index = False
        #fh_index  = False
        for i,p in enumerate(model.get_params()):
            if 'a6' in p.name.lower() or 'a_6' in p.name.lower():  a6_index = i
            if 'fl' in p.name.lower() or 'f_l' in p.name.lower():  fl_index = i

        if str(a6_index)=='False' or str(fl_index)=='False':
            print('I was not able to find the indices, please fix it here:\n ../scripts/SLSQP_zfit.py')
            raise NotImplementedError
    #Now define the "simple" constraints give the found index
    constAngParams = (
                 {'type': 'ineq', 'fun': lambda x:  1 - x[fl_index]},
                 {'type': 'ineq', 'fun': lambda x:  x[fl_index]},
                 {'type': 'ineq', 'fun': lambda x:  1 - (x[fl_index] + x[a6_index])},
                 {'type': 'ineq', 'fun': lambda x:  1 + (x[a6_index] - x[fl_index])}
                )
    print(a6_index, fl_index)

    return constAngParams

solutions = dict()
solutions[0] = dict(d0 = "c0")
solutions[1] = dict(d0 = "(c0*diff1 - c1*sum1 + c1*sum2)/diff1",
                    d1 = "c1*diff2/diff1")
solutions[2] = dict(d0 = "(c0*diff1**2 - c1*diff1*sum1 + c1*diff1*sum2 - c2*diff1**2 + c2*diff2**2 + 2*c2*sum1**2 - 4*c2*sum1*sum2 + 2*c2*sum2**2)/diff1**2",
                    d1 = "(c1*diff1*diff2 - 4*c2*diff2*sum1 + 4*c2*diff2*sum2)/diff1**2",
                    d2 = "c2*diff2**2/diff1**2")
solutions[3] = dict(d0 = "(c0*diff1**3 - c1*diff1**2*sum1 + c1*diff1**2*sum2 - c2*diff1**3 + c2*diff1*diff2**2 + 2*c2*diff1*sum1**2 - 4*c2*diff1*sum1*sum2 + 2*c2*diff1*sum2**2 + 3*c3*diff1**2*sum1 - 3*c3*diff1**2*sum2 - 6*c3*diff2**2*sum1 + 6*c3*diff2**2*sum2 - 4*c3*sum1**3 + 12*c3*sum1**2*sum2 - 12*c3*sum1*sum2**2 + 4*c3*sum2**3)/diff1**3",
                    d1 = "(c1*diff1**2*diff2 - 4*c2*diff1*diff2*sum1 + 4*c2*diff1*diff2*sum2 - 3*c3*diff1**2*diff2 + 3*c3*diff2**3 + 12*c3*diff2*sum1**2 - 24*c3*diff2*sum1*sum2 + 12*c3*diff2*sum2**2)/diff1**3",
                    d2 = "(c2*diff1*diff2**2 - 6*c3*diff2**2*sum1 + 6*c3*diff2**2*sum2)/diff1**3",
                    d3 = "c3*diff2**3/diff1**3")
solutions[4] = dict(d0 = "(c0*diff1**4 - c1*diff1**3*sum1 + c1*diff1**3*sum2 - c2*diff1**4 + c2*diff1**2*diff2**2 + 2*c2*diff1**2*sum1**2 - 4*c2*diff1**2*sum1*sum2 + 2*c2*diff1**2*sum2**2 + 3*c3*diff1**3*sum1 - 3*c3*diff1**3*sum2 - 6*c3*diff1*diff2**2*sum1 + 6*c3*diff1*diff2**2*sum2 - 4*c3*diff1*sum1**3 + 12*c3*diff1*sum1**2*sum2 - 12*c3*diff1*sum1*sum2**2 + 4*c3*diff1*sum2**3 + c4*diff1**4 - 4*c4*diff1**2*diff2**2 - 8*c4*diff1**2*sum1**2 + 16*c4*diff1**2*sum1*sum2 - 8*c4*diff1**2*sum2**2 + 3*c4*diff2**4 + 24*c4*diff2**2*sum1**2 - 48*c4*diff2**2*sum1*sum2 + 24*c4*diff2**2*sum2**2 + 8*c4*sum1**4 - 32*c4*sum1**3*sum2 + 48*c4*sum1**2*sum2**2 - 32*c4*sum1*sum2**3 + 8*c4*sum2**4)/diff1**4",
                    d1 = "(c1*diff1**3*diff2 - 4*c2*diff1**2*diff2*sum1 + 4*c2*diff1**2*diff2*sum2 - 3*c3*diff1**3*diff2 + 3*c3*diff1*diff2**3 + 12*c3*diff1*diff2*sum1**2 - 24*c3*diff1*diff2*sum1*sum2 + 12*c3*diff1*diff2*sum2**2 + 16*c4*diff1**2*diff2*sum1 - 16*c4*diff1**2*diff2*sum2 - 24*c4*diff2**3*sum1 + 24*c4*diff2**3*sum2 - 32*c4*diff2*sum1**3 + 96*c4*diff2*sum1**2*sum2 - 96*c4*diff2*sum1*sum2**2 + 32*c4*diff2*sum2**3)/diff1**4",
                    d2 = "(c2*diff1**2*diff2**2 - 6*c3*diff1*diff2**2*sum1 + 6*c3*diff1*diff2**2*sum2 - 4*c4*diff1**2*diff2**2 + 4*c4*diff2**4 + 24*c4*diff2**2*sum1**2 - 48*c4*diff2**2*sum1*sum2 + 24*c4*diff2**2*sum2**2)/diff1**4",
                    d3 = "(c3*diff1*diff2**3 - 8*c4*diff2**3*sum1 + 8*c4*diff2**3*sum2)/diff1**4",
                    d4 = "c4*diff2**4/diff1**4")  
solutions[5] = dict(d0 = "(c0*diff1**5 - c1*diff1**4*sum1 + c1*diff1**4*sum2 - c2*diff1**5 + c2*diff1**3*diff2**2 + 2*c2*diff1**3*sum1**2 - 4*c2*diff1**3*sum1*sum2 + 2*c2*diff1**3*sum2**2 + 3*c3*diff1**4*sum1 - 3*c3*diff1**4*sum2 - 6*c3*diff1**2*diff2**2*sum1 + 6*c3*diff1**2*diff2**2*sum2 - 4*c3*diff1**2*sum1**3 + 12*c3*diff1**2*sum1**2*sum2 - 12*c3*diff1**2*sum1*sum2**2 + 4*c3*diff1**2*sum2**3 + c4*diff1**5 - 4*c4*diff1**3*diff2**2 - 8*c4*diff1**3*sum1**2 + 16*c4*diff1**3*sum1*sum2 - 8*c4*diff1**3*sum2**2 + 3*c4*diff1*diff2**4 + 24*c4*diff1*diff2**2*sum1**2 - 48*c4*diff1*diff2**2*sum1*sum2 + 24*c4*diff1*diff2**2*sum2**2 + 8*c4*diff1*sum1**4 - 32*c4*diff1*sum1**3*sum2 + 48*c4*diff1*sum1**2*sum2**2 - 32*c4*diff1*sum1*sum2**3 + 8*c4*diff1*sum2**4 - 5*c5*diff1**4*sum1 + 5*c5*diff1**4*sum2 + 30*c5*diff1**2*diff2**2*sum1 - 30*c5*diff1**2*diff2**2*sum2 + 20*c5*diff1**2*sum1**3 - 60*c5*diff1**2*sum1**2*sum2 + 60*c5*diff1**2*sum1*sum2**2 - 20*c5*diff1**2*sum2**3 - 30*c5*diff2**4*sum1 + 30*c5*diff2**4*sum2 - 80*c5*diff2**2*sum1**3 + 240*c5*diff2**2*sum1**2*sum2 - 240*c5*diff2**2*sum1*sum2**2 + 80*c5*diff2**2*sum2**3 - 16*c5*sum1**5 + 80*c5*sum1**4*sum2 - 160*c5*sum1**3*sum2**2 + 160*c5*sum1**2*sum2**3 - 80*c5*sum1*sum2**4 + 16*c5*sum2**5)/diff1**5",
                    d1 = "(c1*diff1**4*diff2 - 4*c2*diff1**3*diff2*sum1 + 4*c2*diff1**3*diff2*sum2 - 3*c3*diff1**4*diff2 + 3*c3*diff1**2*diff2**3 + 12*c3*diff1**2*diff2*sum1**2 - 24*c3*diff1**2*diff2*sum1*sum2 + 12*c3*diff1**2*diff2*sum2**2 + 16*c4*diff1**3*diff2*sum1 - 16*c4*diff1**3*diff2*sum2 - 24*c4*diff1*diff2**3*sum1 + 24*c4*diff1*diff2**3*sum2 - 32*c4*diff1*diff2*sum1**3 + 96*c4*diff1*diff2*sum1**2*sum2 - 96*c4*diff1*diff2*sum1*sum2**2 + 32*c4*diff1*diff2*sum2**3 + 5*c5*diff1**4*diff2 - 15*c5*diff1**2*diff2**3 - 60*c5*diff1**2*diff2*sum1**2 + 120*c5*diff1**2*diff2*sum1*sum2 - 60*c5*diff1**2*diff2*sum2**2 + 10*c5*diff2**5 + 120*c5*diff2**3*sum1**2 - 240*c5*diff2**3*sum1*sum2 + 120*c5*diff2**3*sum2**2 + 80*c5*diff2*sum1**4 - 320*c5*diff2*sum1**3*sum2 + 480*c5*diff2*sum1**2*sum2**2 - 320*c5*diff2*sum1*sum2**3 + 80*c5*diff2*sum2**4)/diff1**5",
                    d2 = "(c2*diff1**3*diff2**2 - 6*c3*diff1**2*diff2**2*sum1 + 6*c3*diff1**2*diff2**2*sum2 - 4*c4*diff1**3*diff2**2 + 4*c4*diff1*diff2**4 + 24*c4*diff1*diff2**2*sum1**2 - 48*c4*diff1*diff2**2*sum1*sum2 + 24*c4*diff1*diff2**2*sum2**2 + 30*c5*diff1**2*diff2**2*sum1 - 30*c5*diff1**2*diff2**2*sum2 - 40*c5*diff2**4*sum1 + 40*c5*diff2**4*sum2 - 80*c5*diff2**2*sum1**3 + 240*c5*diff2**2*sum1**2*sum2 - 240*c5*diff2**2*sum1*sum2**2 + 80*c5*diff2**2*sum2**3)/diff1**5",
                    d3 = "(c3*diff1**2*diff2**3 - 8*c4*diff1*diff2**3*sum1 + 8*c4*diff1*diff2**3*sum2 - 5*c5*diff1**2*diff2**3 + 5*c5*diff2**5 + 40*c5*diff2**3*sum1**2 - 80*c5*diff2**3*sum1*sum2 + 40*c5*diff2**3*sum2**2)/diff1**5",
                    d4 = "(c4*diff1*diff2**4 - 10*c5*diff2**4*sum1 + 10*c5*diff2**4*sum2)/diff1**5",
                    d5 = "c5*diff2**5/diff1**5")


def get_params_fromdict_new_range(coefs, old_space, new_space):
    if len(coefs)>len(solutions): 
        raise NotImplementedError(f'Have not included more solutions for Chebyshev degree > {len(solutions)}.\nEither implement a gloabl solver or use this colab notebook to obtain for the other degrees:\n-> https://colab.research.google.com/drive/1h5xDvMwx99Aqpath_ijTU9PW2oeXOulZbYIm?usp=sharing')
    
    #Define the distnaces needed for the transformations
    diff1 = old_space.limits[1][0][0]-old_space.limits[0][0][0]
    sum1 = old_space.limits[1][0][0]+old_space.limits[0][0][0]

    diff2 = new_space.limits[1][0][0]-new_space.limits[0][0][0]
    sum2 = new_space.limits[1][0][0]+new_space.limits[0][0][0]

    #Extract the set of solutions given the degree of the Chebyshev polynomial
    sols = solutions[len(coefs)-1]

    #To get the value of the coeficients, first define local variables named ci (Original value for the coeficients)
    for name, value in coefs.items():   
        locals()[name.replace('_', '')] = value

    #To get the new value of the coeficients, use `eval` to evaluate the string as a mathematical function
    new_vals = dict()
    for name in coefs:        
        new_vals[name] = eval(sols[name.replace('c_', 'd')])
    return new_vals

def set_params_chebyshev(model, old_space):
    values = dict()
    for name, param in model.params.items():
        values[name] = param.value().numpy()

    new_vals = get_params_fromdict_new_range(values, old_space, model.space)
    
    for name, param in model.params.items():
        print(f'{name:<3}', f'{values[name]:<4.3f} --> {new_vals[name]:.3f}')
        param.set_value(new_vals[name])

def create_single_pdf(pdf_info, create_params=True, obs=None, era = '2022', bin = '1', extra=''):
    type_ = pdf_info['type']
    #print(type_)
    #name  = pdf_info['name']
    if not obs:
        obs = zfit.Space(pdf_info['Space']['obs'][0],
                        pdf_info['Space']['limits'])

    if type_=='Chebyshev':
        coeffs_names = [k for k in pdf_info['parameters'].keys()]
        coeffs_names.sort(key=lambda x: int(x.split('_')[1]))
        if create_params:
            try:
                coeffs = [zfit.Parameter(f"{pdf_info['name']}_{c}_{uuid.uuid4().hex[:6]}", pdf_info['parameters'][c])\
                                            for c in coeffs_names]
            except zfit.exception.NameAlreadyTakenError:
                rand_name = str(np.random.randint(0, 100))
                coeffs = [zfit.Parameter(f"{pdf_info['name']}-{rand_name}_{c}", pdf_info['parameters'][c])\
                                            for c in coeffs_names]
        else: coeffs = [pdf_info['parameters'][c] for c in coeffs_names]

        return zfit.pdf.Chebyshev(obs, coeffs[1:], coeff0=coeffs[0], name=pdf_info['name'])


    params_zfit = dict()
    for name_, value in pdf_info['parameters'].items():
        if create_params: params_zfit[name_] = zfit.Parameter(f"{pdf_info['name']}_{name_}_{uuid.uuid4().hex[:6]}", value, floating=False)
        else: params_zfit[name_] = value

    if type_=='MyErf_PDF':
        return customPDFs.MyErf_PDF(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=='DoubleCB':
        return zfit.pdf.DoubleCB(**params_zfit, obs=obs, name=pdf_info['name']+f'_{extra}')
    elif type_=="MyArctanPDF":
        return customPDFs.MyArctanPDF(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=="MyNSST_PDF":
        return customPDFs.MyNSST_PDF(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=='Gauss':
        return zfit.pdf.Gauss(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=="JohnsonSU":
        return customPDFs.JohnsonSU(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=="Exponential":
        if 'lambda' in  params_zfit and 'lam' in params_zfit:
            params_zfit.pop('lambda')
        return zfit.pdf.Exponential(**params_zfit, obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=='bernstein':
        return customPDFs.bernstein(**params_zfit,obs=obs, name=pdf_info['name'] + f'_{extra}')
    elif type_=='errf':
        return customPDFs.errf(**params_zfit,obs=obs, name=pdf_info['name'] + f'_{extra}')

def extend_model(pdf, yield_, create_param=False):
    """
    Extend the model by creating an extended PDF.

    Parameters:
    - pdf: The PDF object.
    - yield_: The yield value.
    - create_param: Whether to create a zfit.Parameter for yield.

    Returns:
    - The extended PDF object.
    """
    if create_param:
        yield_ = zfit.Parameter(f"Y_{pdf.name}_{uuid.uuid4().hex[:6]}", yield_)
    return pdf.create_extended(yield_)

def build_sum_pdf(pdf_info, sub_pdfs, fracs, create_params):
    """
    Build a SumPDF from sub-PDFs and fractions.

    Parameters:
    - pdf_info: Dictionary containing PDF information.
    - sub_pdfs: List of sub-PDF objects.
    - fracs: List of fractions for the SumPDF.
    - create_params: Boolean indicating if parameters should be created.

    Returns:
    - The created SumPDF object.
    """
    if len(sub_pdfs) == 1:
        return sub_pdfs[0]
    
    if all([sub_pdf.is_extended for sub_pdf in sub_pdfs]):
        return zfit.pdf.SumPDF(sub_pdfs,  name=pdf_info['name'])    
    
    if create_params:
        fracs = [zfit.Parameter(f"{key}_{pdf_info['name']}_{uuid.uuid4().hex[:6]}", frac) for key, frac in pdf_info['parameters'].items() if 'frac' in key]
    return zfit.pdf.SumPDF(sub_pdfs, fracs, name=pdf_info['name'])

def build_pdfs_bottom_up(pdf_info, create_params=True, obs=None):
    """
    Build PDFs from bottom to top based on the provided information.

    Parameters:
    - pdf_info: Dictionary containing PDF information.
    - create_params: Boolean indicating if parameters should be created.
    - obs: Observation space.

    Returns:
    - The constructed PDF object.
    """
    if not obs:
        obs = zfit.Space(pdf_info['Space']['obs'][0], pdf_info['Space']['limits'])
    
    sub_pdfs_info = pdf_info.get('sub_pdfs', None)
    if sub_pdfs_info:
        sub_pdfs_ = [build_pdfs_bottom_up(sub_pdf_info, create_params=create_params, obs=obs) for sub_pdf_info in sub_pdfs_info]
        sub_pdfs = [pdf for pdf in sub_pdfs_ if pdf]
        
        if pdf_info['type'] == 'SumPDF':
            # Get the fracs of the SUM
            fracs = [f for indx,(k, f) in enumerate(pdf_info['parameters'].items()) if 'frac' in k and sub_pdfs_[indx]]
            # Sum the components
            pdf = build_sum_pdf(pdf_info, sub_pdfs, fracs, create_params)            
            # In case the model is expected to be extended, but is not yet done
            if pdf_info['is_extended'] and not pdf.is_extended:
                yield_ = pdf_info['yield']
                # This can happen when the pdf in the observable (possible reduced region)
                # vanishes and therefore is removed from the complete PDF to avoid crashes
                if len(fracs) == 1:
                    # One has to modify the yield since we are removing a component
                    yield_ *= fracs[0]
                pdf = extend_model(pdf, yield_, create_param=create_params)
        else:
            raise NotImplementedError

        return pdf
    
    else:
        pdf = create_single_pdf(pdf_info, create_params=create_params, obs=obs)
        # It could happen that the model is partically zero in the reduced region
        # It creates problems in the fit if added, 
        # The integral over the observable is a nan
        #if np.isnan(pdf.integrate(obs).numpy()[0]):
        if np.isnan(pdf.integrate(obs).numpy()):
            return None
        if pdf_info['is_extended']:
            pdf = extend_model(pdf, pdf_info['yield'], create_param=create_params)
        return pdf

def set_new_fraction_2pdfs(pdf, original_obs, reduced_obs):
    # Set normalization range for both components of the PDF to the original space (observable)
    try:
        # Calculate integrals of the PDF components over the reduced space (observable)
        # This should be teh ratio of the total new area over the old area
        # In this case the new area is 1, since it is normalized in the new range 
        _Int0 = 1/pdf.pdfs[0].integrate(original_obs)
        _Int1 = 1/pdf.pdfs[1].integrate(original_obs)
        
        # Get the fraction of the first component as a numpy array
        ff = pdf.fracs[0].numpy()
        print('Original Fraction : ' , ff)
        
        # Calculate the new fraction for the first component based on integrals and existing fractions
        newff = _Int0 * ff / (_Int0 * ff + _Int1 * (1 - ff))
        print('New Fraction      : ' , newff)
        
        # Set the new fraction value for the first component in the PDF
        pdf.fracs[0].set_value(newff.numpy()[0])
    
    except AttributeError:
        print('Single model, no fraction to change')

def save_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)

def setup_mass_model(mass_model_json, mass, mass_red):
    nominalComplete = build_pdfs_bottom_up(mass_model_json, create_params=True, obs=mass_red)

    signal = nominalComplete.pdfs[0]
    backgr = nominalComplete.pdfs[1]

    try:
        signal_pdfs = signal.pdfs
        fractions = signal.fracs
        signal_unex = zfit.pdf.SumPDF(signal_pdfs, fractions[:-1])
    except AttributeError:
        signal_unex = signal

    try:
        backgr_pdfs = backgr.pdfs
        fractions = backgr.fracs
        backgr_unex = zfit.pdf.SumPDF(backgr_pdfs, fractions[:-1])
        for bkg_subpdf in backgr_pdfs:
            if 'Chebyshev' in str(type(bkg_subpdf)):
                print('Changing params in model', bkg_subpdf.name)
                set_params_chebyshev(bkg_subpdf, mass)
    except AttributeError:
        backgr_unex = backgr
        if 'Chebyshev' in str(type(backgr_unex)):
            print('Changing params in model', backgr_unex.name)
            set_params_chebyshev(backgr_unex, mass)

    set_new_fraction_2pdfs(backgr, mass, mass_red)
    set_new_fraction_2pdfs(signal, mass, mass_red)

    Yb_red = backgr.get_yield()
    Ys_red = signal.get_yield()

    Yb_ini = Yb_red.value().numpy()
    Ys_ini = Ys_red.value().numpy()

    Yb_red.set_value(Yb_ini / backgr.integrate(mass).numpy()[0])
    Ys_red.set_value(Ys_ini / signal.integrate(mass).numpy()[0])

    return {
        'complete': nominalComplete,
        'signal': signal,
        'signal_unex': signal_unex,
        'backgr': backgr,
        'backgr_unex': backgr_unex,
        'Yb': Yb_red,
        'Ys': Ys_red
    }

def list_of_pdfs(pdf_info, create_params=True, obs=None, era = '2022', bin = '1', extra=''):
    """
    Build PDFs from bottom to top based on the provided information.

    Parameters:
    - pdf_info: Dictionary containing PDF information.
    - create_params: Boolean indicating if parameters should be created.
    - obs: Observation space.

    Returns:
    - List of PDF's that are part of a global PDF (signal, background).
    """
    if not obs:
        obs = zfit.Space(pdf_info['Space']['obs'][0], pdf_info['Space']['limits'])

    sub_pdfs_info = pdf_info.get('sub_pdfs', None)
    if sub_pdfs_info:
        sub_pdfs_ = [list_of_pdfs(sub_pdf_info, create_params=create_params, obs=obs, era = era, bin = bin, extra=extra) for sub_pdf_info in sub_pdfs_info]
        sub_pdfs = [pdf for pdf in sub_pdfs_ if pdf]

        return sub_pdfs

    else:
        pdf = create_single_pdf(pdf_info, create_params=create_params, obs=obs, era = era, bin = bin, extra=extra)
        # It could happen that the model is partically zero in the reduced region
        # It creates problems in the fit if added, 
        # The integral over the observable is a nan
        #if np.isnan(pdf.integrate(obs).numpy()[0]):
        #    return None
        #if pdf_info['is_extended']:
        #    pdf = extend_model(pdf, pdf_info['yield'], create_param=create_params)
        return pdf
    
def build_signal_bkg(pdf_info, sub_pdfs, create_params=True, obs=None, era = '2022', bin = '1'):

    if not obs:
        obs = zfit.Space(pdf_info['Space']['obs'][0], pdf_info['Space']['limits'])

    signal_params = pdf_info['sub_pdfs'][0]
    bkg_params = pdf_info['sub_pdfs'][1]

    fracs_signal = [signal_params['parameters']['frac_0'],signal_params['parameters']['frac_1']]
    #fracs_bkg = [f for indx,(k, f) in enumerate(pdf_info['sub_pdfs']['1']['parameters'].items())]
    
    signal = zfit.pdf.SumPDF(sub_pdfs[0], fracs_signal, name = signal_params['name'] + f'_{era}_{bin}')
    bkg = sub_pdfs[-1]

    return signal, bkg

def build_signal_bkg_red(pdf_info, sub_pdfs, 
                        sub_pdfs_complete, signal_complete, bkg_complete, 
                        int_dom, 
                        obs_red=None, obs_complete=None, 
                        era = '2022', bin = '1', create_params=True):

    if not obs_complete:
        obs_complete = zfit.Space(pdf_info['Space']['obs'][0], pdf_info['Space']['limits'])

    signal_params = pdf_info['sub_pdfs'][0]
    bkg_params = pdf_info['sub_pdfs'][1]
    
    global_fracs = [pdf_info['parameters']['frac_0'],pdf_info['parameters']['frac_1']]
    signal_fracs = [signal_params['parameters']['frac_0'],signal_params['parameters']['frac_1']]

    ###SIGNAL_FRACS
    sub_func_0 = sub_pdfs_complete[0][0]
    sub_func_1 = sub_pdfs_complete[0][1]

    int_func_0 = sub_func_0.integrate(limits=int_dom)
    int_func_1 = sub_func_1.integrate(limits=int_dom)

    new_frac_0 = ((signal_fracs[0] * int_func_0)/(signal_fracs[0] * int_func_0 + signal_fracs[1] * int_func_1))
    new_frac_1 = 1 - new_frac_0

    new_frac_0 = new_frac_0.numpy()
    new_frac_1 = new_frac_1.numpy()

    new_signal_fracs = [new_frac_0[0],new_frac_1[0]]

    #print(signal_fracs,' --> ',new_signal_fracs)

    ###GLOBAL_FRACS
    int_func_0 = signal_complete.integrate(limits=int_dom)
    int_func_1 = bkg_complete.integrate(limits=int_dom)

    new_frac_0 = ((global_fracs[0] * int_func_0)/(global_fracs[0] * int_func_0 + global_fracs[1] * int_func_1))
    new_frac_1 = 1 - new_frac_0

    new_frac_0 = new_frac_0.numpy()
    new_frac_1 = new_frac_1.numpy()

    new_global_fracs = [new_frac_0[0],new_frac_1[0]]

    #print(global_fracs,' --> ',new_global_fracs)

    
    bkg = sub_pdfs[1]

    sig = zfit.pdf.SumPDF(sub_pdfs[0],new_signal_fracs,obs=obs_red, name = signal_params['name'] + f'_{era}_{bin}_red')

    complete_mass_model = zfit.pdf.SumPDF([sig,bkg], new_global_fracs, obs=obs_red, name = f'complete_mass_model_{era}_{bin}_red')

    return sig, bkg, complete_mass_model, new_global_fracs


def complete_PDF_new(FL, A6, n_list, flabel=None):

    complete = []

    for i in range(len(YearsLabels)):

        if YearsLabels[i] == '2022':
            path_i = path_jsons_2022
            n = n_list[0]
        elif YearsLabels[i] == 'pre2023':
            path_i = path_jsons_20231
            n = n_list[1]
        elif YearsLabels[i] == 'post2023':
            path_i = path_jsons_20232
            n = n_list[2]
        else:
            print('Incorrect era, check your code.')

        # == Angular efficiency and background ==

        bkg_json_k = tools.read_json(path_i + f'angularBkg_costhetak_bin{nBin}.json')
        params_bkg_k = list()
        for k in range(len(bkg_json_k['parameters'])):
            par_k = bkg_json_k['parameters'][f'c_{k}']
            params_bkg_k.append(par_k)             
        bkg_k = customPDFs.bernstein(coeffs=params_bkg_k, obs=cos_k)

        bkg_json_l = tools.read_json(path_i + f'angularBkg_costhetal_bin{nBin}.json')
        params_bkg_l = list()
        for k in range(len(bkg_json_l['parameters'])):
            par_k = bkg_json_l['parameters'][f'c_{k}']
            params_bkg_l.append(par_k)      
        bkg_l = customPDFs.bernstein(coeffs=params_bkg_l, obs=cos_l)


        eff_json_k = tools.read_json(path_i + f'angularEffTotal_scipy_costhetak_bin{nBin}.json')
        params_eff_k = list()
        for k in range(len(eff_json_k['parameters'])):
            par_k = eff_json_k['parameters'][f'c{k}']
            params_eff_k.append(par_k)             
        eff_k = customPDFs.bernstein(coeffs=params_eff_k, obs=cos_k)

        eff_json_l = tools.read_json(path_i + f'angularEffTotal_scipy_costhetal_bin{nBin}.json')
        params_eff_l = list()
        for k in range(len(eff_json_l['parameters'])):
            par_l = eff_json_l['parameters'][f'c{k}']
            params_eff_l.append(par_l)
        eff_l = customPDFs.bernstein(coeffs=params_eff_l, obs=cos_l)

        # == Mass PDF ==
        mass_json = tools.read_json(path_i + f'PDFparams_data_{YearsLabels[i]}_{nBin}.json')

        #mass_red = zfit.Space('mass_red', [5.2, 5.5])
        #complete_obs_red = mass_red * cos_k * cos_l
        #int_dom = zfit.Space(f'mass', [5.2, 5.5])

        sub_pdfs = list_of_pdfs(mass_json, create_params = True, obs = mass_red, era = YearsLabels[i], bin = nBin, extra='red')
        sub_pdfs_complete = list_of_pdfs(mass_json, create_params = True, obs = mass, era = YearsLabels[i], bin = nBin, extra='complete')

        mass_signal_com, mass_bkg_com = build_signal_bkg(mass_json, sub_pdfs_complete, obs = mass, era = YearsLabels[i], bin = nBin)
        mass_signal, mass_bkg, complete_mass_model, global_frac = build_signal_bkg_red(pdf_info=mass_json, sub_pdfs=sub_pdfs, 
                                                                                                                    sub_pdfs_complete=sub_pdfs_complete, 
                                                                                                                    signal_complete=mass_signal_com, 
                                                                                                                    bkg_complete=mass_bkg_com, 
                                                                                                                    int_dom=int_dom, 
                                                                                                                    obs_red=mass_red, obs_complete=mass, 
                                                                                                                    era = YearsLabels[i], bin = nBin)


        ang_eff = zfit.pdf.ProductPDF([eff_k, eff_l], obs=obs, name='Ang_eff')
        ang_bkg = zfit.pdf.ProductPDF([bkg_k, bkg_l], obs=obs, name='Ang_bkg')

        pdf_theory = DecayRate_BsTomumuphi_short(FL=FL, A6=A6, obs=obs)
        
        signal_form = zfit.pdf.ProductPDF([mass_signal, pdf_theory, ang_eff], obs=complete_obs_red)
        bkg_form = zfit.pdf.ProductPDF([mass_bkg, ang_bkg], obs=complete_obs_red)

        Ys = zfit.Parameter(f'SigYield_{YearsLabels[i]}_{flabel}', n*global_frac[0], floating = False)
        Yb = zfit.Parameter(f'BkgYield_{YearsLabels[i]}_{flabel}', n*global_frac[1], floating = False)

        signal_ext = signal_form.create_extended(yield_=Ys)
        bkg_ext = bkg_form.create_extended(yield_=Yb)

        pdf_model = zfit.pdf.SumPDF([signal_ext, bkg_ext], name='CompletePDF')
        complete.append(pdf_model)
        

    return complete

def read_data(file):

    if file.endswith('.csv'):
        data = pd.read_csv(file)

    elif file.endswith('.json'):
        with open(file, 'r') as f:
            sample_loaded = np.array(json.load(f)['sample'])

            data = pd.DataFrame({'mass_red': sample_loaded[:, 0], 'cos_k': sample_loaded[:, 1], 'cos_l': sample_loaded[:, 2]})

            data = data[(data['mass_red']>(5.2)) & (data['mass_red']<(5.5))]
        
    else:
        raise NotImplementedError('We have not added a way to read this type of file.')
    
    return data


def MC_loop(N_MC, fl_i, a6_i, n_list):

    Dchis = []

    zfit.core.parameter.Parameter._existing_params.pop('FL0', None)
    zfit.core.parameter.Parameter._existing_params.pop('A60', None)

    # ==== PDF construction and FH definition ====

    FL0   = zfit.Parameter('FL0', fl_i)
    A60   = zfit.Parameter('A60', a6_i)

    # ==== MC generation ====
    MC = complete_PDF_new(FL=FL0, A6=A60, n_list=n_list, flabel='MC')
    MC_model = MC

    constraint_slsqp = create_constraint_Bs_short(MC_model, a6_index=1, fl_index=0)
    SLSQP = SLSQP_zfit.SLSQP(constraints=constraint_slsqp)
    SLSQP_PROF = SLSQP_zfit.SLSQP()

    print('MC PDF created')

    MCsampler1 = MC[0].create_sampler(n_list[0], fixed_params=True)
    MCsampler2 = MC[1].create_sampler(n_list[1], fixed_params=True)
    MCsampler3 = MC[2].create_sampler(n_list[2], fixed_params=True)

    MCsampler1.resample()
    MCsampler2.resample()
    MCsampler3.resample()
    print('MC sampler completed')

    MCsampler = [MCsampler1, MCsampler2, MCsampler3]

    nll_best_mc = zfit.loss.ExtendedUnbinnedNLL(model = MC_model, data = MCsampler)
    nll_prof_mc = zfit.loss.ExtendedUnbinnedNLL(model = MC_model, data = MCsampler)

    # MC proper loop that computes Dchi2 for each toy
    for i in range(N_MC):

        MCsampler1.resample()
        MCsampler2.resample()
        MCsampler3.resample()
        FL0.set_value(fl_i)
        A60.set_value(a6_i)

        free_params_mc = [FL0, A60]

        result_bmc = SLSQP.minimize(nll_best_mc, params=free_params_mc)

        best_L_mc = nll_best_mc.value().numpy()

        FL0.set_value(fl_i)
        A60.set_value(a6_i)

        prof_L_mc = nll_prof_mc.value().numpy()

        Dchi2_mc = prof_L_mc - best_L_mc
        Dchis.append(Dchi2_mc)

        print(f'Profile Likelihood: {prof_L_mc}')
        print(f'Best Likelihood: {best_L_mc}')
        print(f'New Dchi2 for MC sample: {Dchi2_mc}')
      
    return Dchis



# =======================================
# =======================================
# ============= Main code ===============
# =======================================
# =======================================




if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser(prog='Script that computes one point of the 1-CL curve.',
                                        description='This script automatically does everything to obtain 1-CL points for the channel B0s --> mumuphi.')
    

    my_parser.add_argument('--nBin', 
                           type=int,
                           default=1,
                           help='Number of bin.')
    my_parser.add_argument('--fl',
                           type = float,
                           default=0.18,
                           help='Value of FL.')
    my_parser.add_argument('--a6',
                           type = float,
                           default=0.05,
                           help='Value of A6')
    my_parser.add_argument('--N_MC',
                           type=int,
                           default=50, 
                           help='Number of toy MC samples') 
    my_parser.add_argument('--unblinded',
                           type=int,
                           default=0, 
                           help='This flag determines, whether the analysis will be blinded or not: 0 for unblinded, 1 for blinded.') 
    my_parser.add_argument('--it_label',
                           type=int,
                           default=0, 
                           help='Point number of the 1-CL curve.') 

    args  = my_parser.parse_args()

    N_MC = args.N_MC
    nBin = args.nBin
    ilabel = args.it_label

    fl_i = args.fl
    a6_i = args.a6

    # ====== Directories ======

    path_2022 = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Data/B0stomumuphi/2022/Bin{nBin}/'
    path_20231 = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Data/B0stomumuphi/2023/Era1/Bin{nBin}/'
    path_20232 = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Data/B0stomumuphi/2023/Era2/Bin{nBin}/'

    path_jsons_2022 = f'/tmp/Tools/Jsons/2022/Bin{nBin}/'
    path_jsons_20231 = f'/tmp/Tools/Jsons/2023/Era1/Bin{nBin}/'
    path_jsons_20232 = f'/tmp/Tools/Jsons/2023/Era2/Bin{nBin}/'

    # ====== Data reading ======

    if args.unblinded == 0:

        RD_file_2022 = f'RD_Bin{nBin}.csv'
        RD_file_20231 = f'RD_Bin{nBin}.csv'
        RD_file_20232 = f'RD_Bin{nBin}.csv'

    elif args.unblinded == 1:

        RD_file_2022 = f'ToyMC_short_pdf_sample_Teff_bkg_2022_bin_{nBin}.json'
        RD_file_20231 = f'ToyMC_short_pdf_sample_Teff_bkg_pre2023_bin_{nBin}.json'
        RD_file_20232 = f'ToyMC_short_pdf_sample_Teff_bkg_post2023_bin_{nBin}.json'

    else:
        print('Mistake on the .txt file. Check again.')
        print('Only 0 or 1 are valid values for the --unblinded parameter.')

    rf_2022 = path_2022 + RD_file_2022
    real_df_2022 = read_data(rf_2022)
    print(real_df_2022)
    real_sampler_2022 = zfit.Data.from_pandas(real_df_2022)

    rf_20231 = path_20231 + RD_file_20231
    real_df_20231 = read_data(rf_20231)
    real_sampler_20231 = zfit.Data.from_pandas(real_df_20231)

    rf_20232 = path_20232 + RD_file_20232
    real_df_20232 = read_data(rf_20232)
    real_sampler_20232 = zfit.Data.from_pandas(real_df_20232)

    n1 = len(real_df_2022)
    n2 = len(real_df_20231)
    n3 = len(real_df_20232)

    n_list = [n1, n2, n3]

    dataZ_red=[]
    dataZ_red.append(real_sampler_2022)
    dataZ_red.append(real_sampler_20231)
    dataZ_red.append(real_sampler_20232)

    # ====== PDF construction ======
    #  == Observables ==

    YearsLabels = ['2022', 'pre2023', 'post2023']
    cos_k = zfit.Space(f'cos_k', limits=(-1, 1))
    cos_l = zfit.Space(f'cos_l', limits=(-1, 1))
    mass  = zfit.Space(f'mass', [5.1,5.7])
    obs = cos_k * cos_l  
    complete_obs = mass * cos_k * cos_l

    mass_red = zfit.Space('mass_red', [5.2, 5.5])
    complete_obs_red = mass_red * cos_k * cos_l
    int_dom = zfit.Space(f'mass', [5.2, 5.5])

    # == POIs initialization ==
    FL = zfit.Parameter('FL', 0.5, 0, 1)
    A6 = zfit.Parameter('A6', 0.1, -1.0, 1.0) # Real values

    complete = complete_PDF_new(FL=FL, A6=A6, n_list=n_list, flabel='RD')
    RD_model = complete

    constraint_slsqp = create_constraint_Bs_short(RD_model, a6_index=1, fl_index=0)
    SLSQP = SLSQP_zfit.SLSQP(constraints=constraint_slsqp, verbosity = 6)
    SLSQP_PROF = SLSQP_zfit.SLSQP()

    print('PDF created')

    nll_best_data = zfit.loss.ExtendedUnbinnedNLL(model=RD_model, data = dataZ_red)
    nll_prof_data = zfit.loss.ExtendedUnbinnedNLL(model=RD_model, data = dataZ_red)

    # ====== Starting fitting procedures ======

    # == Best likelihood ==
    print(f'Starting for FL = {fl_i} and A6 = {a6_i}')
    start = timer()

    FL.set_value(fl_i)
    A6.set_value(a6_i)

    free_params = [FL, A6]

    result_b_data = SLSQP.minimize(nll_best_data, params=free_params)

    best_L_data = nll_best_data.value().numpy()

    print('best L data:',best_L_data)

    flbest = result_b_data.params[FL]['value']
    a6best = result_b_data.params[A6]['value']
    print('The best FH value for data is: ', flbest)
    print('The best A6 value for data is: ', a6best)

    # == Profile likelihood ==
    FL.set_value(fl_i)
    A6.set_value(a6_i)

    prof_L_data = nll_prof_data.value().numpy()

    print('prof L data: ', prof_L_data)

    Dchi2_data = prof_L_data - best_L_data

    print(f'Dchi data: {Dchi2_data}')

    # ===== Toy MC generation and Dchi2 calculation =====
    dchis = MC_loop(N_MC=N_MC, fl_i=fl_i, a6_i=a6_i, n_list=n_list)

    N_toys = 0
    for j in range(len(dchis)):
        if dchis[j] >= Dchi2_data:
            N_toys+=1
        else:
            pass

    OneminusCL = N_toys/N_MC

    end = timer()

    if args.unblinded == 1:
        output_dir = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Results/B0stomumuphi/v0/Blinded/Bin{nBin}/'
    else:
        output_dir = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Results/B0stomumuphi/v0/Simultaneous_fit/Bin{nBin}/'
        
    if N_MC != 100:
        output_dir = f'/eos/user/n/ntepecti/PhD/Feldman_Cousins/Results/B0stomumuphi/v0/Simultaneous_fit/Bin{nBin}/{N_MC}MCs/'
    else:
        pass

    with open(output_dir + f'New_channel_OCLpoints_Bin{nBin}_point{ilabel}.txt', 'w') as sf:
        sf.write(f'{a6_i}, {fl_i}, {a6best}, {flbest}, {OneminusCL}\n')


    print(f'It took {(end - start):.4f}')