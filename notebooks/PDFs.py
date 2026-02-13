import zfit
from zfit import z
import tensorflow as tf
import numpy as np
import pandas as pd

# PDF física completa
class FullAngular_Physical_PDF(zfit.pdf.BasePDF):
    def __init__(self, obs, FL, S3, S9, AFB, S4, S7, S5, S8, name="FullAngular_Physical_PDF"):
        params = {
            'FL': FL, 'S3': S3, 'S9': S9, 'AFB': AFB,
            'S4': S4, 'S7': S7, 'S5': S5, 'S8': S8
        }
        super().__init__(obs, params, name=name)
    
    def _unnormalized_pdf(self, x):
        vars_list = z.unstack_x(x)
        cos_l = vars_list[0]
        cos_k = vars_list[1]
        phi   = vars_list[2]
        
        # sin_k = tf.sqrt(1.0 - cos_k**2)
        # sin_l = tf.sqrt(1.0 - cos_l**2)
        # # Versión segura
        sin_k = tf.sqrt(tf.maximum(1.0 - cos_k**2, 0.0))
        sin_l = tf.sqrt(tf.maximum(1.0 - cos_l**2, 0.0))
        sin2_k = sin_k**2
        cos2_k = cos_k**2
        sin2_l = sin_l**2
        
        cos2l_term = 2.0 * cos_l**2 - 1.0
        sin2l_term = 2.0 * sin_l * cos_l
        sin2k_term = 2.0 * sin_k * cos_k
        
        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)
        cos2_phi = tf.cos(2.0 * phi)
        sin2_phi = tf.sin(2.0 * phi)

        FL = self.params['FL']
        S3 = self.params['S3']
        S9 = self.params['S9']
        AFB = self.params['AFB']
        S4 = self.params['S4']
        S7 = self.params['S7']
        S5 = self.params['S5']
        S8 = self.params['S8']
        
        term1 = 0.75 * (1.0 - FL) * sin2_k
        term2 = FL * cos2_k
        term3 = 0.25 * (1.0 - FL) * sin2_k * cos2l_term
        term4 = -1.0 * FL * cos2_k * cos2l_term
        term5 = S3 * sin2_k * sin2_l * cos2_phi
        term6 = S4 * sin2k_term * sin2l_term * cos_phi
        term7 = S5 * sin2k_term * sin_l * cos_phi
        term8 = (4.0/3.0) * AFB * sin2_k * cos_l
        term9 = S7 * sin2k_term * sin_l * sin_phi
        term10 = S8 * sin2k_term * sin2l_term * sin_phi
        term11 = S9 * sin2_k * sin2_l * sin2_phi
        
        pdf = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11
        return pdf

    @zfit.supports(norm=True)
    def _integrate(self, limits, norm_range, options=None):
        """
        Integral analítica constante.
        Se añade 'options=None' para compatibilidad con las llamadas internas de zfit.
        """
        integral_value = 32.0 * np.pi / 9.0
        return tf.constant([integral_value], dtype=self.dtype)


# PDF trasnformada
class FullAngular_Transformed_PDF(zfit.pdf.BasePDF):
    def __init__(self, obs, raw_FL, raw_S3, raw_S9, raw_AFB, raw_S4, raw_S7, raw_S5, raw_S8, name="FullAngular_Transformed_PDF"):
        params = {
            'rFL': raw_FL, 'rS3': raw_S3, 'rS9': raw_S9, 'rAFB': raw_AFB,
            'rS4': raw_S4, 'rS7': raw_S7, 'rS5': raw_S5, 'rS8': raw_S8
        }
        super().__init__(obs, params, name=name)

    def _get_physical_coeffs(self):

        FL = 0.5 * (1.0 + tf.math.tanh(self.params['rFL']))
        FT = 1.0 - FL
        limit_S3 = 0.5 * FT
        S3 = limit_S3 * tf.math.tanh(self.params['rS3'])
        
        R2_trans_total = 0.25 * tf.square(FT)
        R2_available = R2_trans_total - tf.square(S3)
        R2_available = tf.maximum(R2_available, 1e-18) 
        R_trans = tf.sqrt(R2_available)
        S9 = R_trans * tf.math.tanh(self.params['rS9'])
        
        R2_rem_AFB = R2_available - tf.square(S9)
        R2_rem_AFB = tf.maximum(R2_rem_AFB, 1e-18)
        limit_AFB = 1.5 * tf.sqrt(R2_rem_AFB)
        AFB = limit_AFB * tf.math.tanh(self.params['rAFB'])
        
        bound_mix1 = FL * (FT - 2.0 * S3)
        bound_mix1 = tf.maximum(bound_mix1, 1e-18)
        limit_S4 = 0.5 * tf.sqrt(bound_mix1)
        S4 = limit_S4 * tf.math.tanh(self.params['rS4'])
        
        R2_S7 = bound_mix1 - 4.0 * tf.square(S4)
        R2_S7 = tf.maximum(R2_S7, 1e-18)
        S7 = tf.sqrt(R2_S7) * tf.math.tanh(self.params['rS7'])
        
        bound_mix2 = FL * (FT + 2.0 * S3)
        bound_mix2 = tf.maximum(bound_mix2, 1e-18)
        limit_S5 = tf.sqrt(bound_mix2)
        S5 = limit_S5 * tf.math.tanh(self.params['rS5'])
        
        R2_S8 = bound_mix2 - tf.square(S5)
        R2_S8 = tf.maximum(R2_S8, 1e-18)
        limit_S8 = 0.5 * tf.sqrt(R2_S8)
        S8 = limit_S8 * tf.math.tanh(self.params['rS8'])
        
        return FL, S3, S9, AFB, S4, S7, S5, S8

    def _unnormalized_pdf(self, x):
        vars_list = z.unstack_x(x)
        cos_l = vars_list[0]
        cos_k = vars_list[1]
        phi   = vars_list[2]
        
        sin_k = tf.sqrt(1.0 - cos_k**2)
        sin_l = tf.sqrt(1.0 - cos_l**2)
        sin2_k = sin_k**2
        cos2_k = cos_k**2
        sin2_l = sin_l**2
        cos2l_term = 2.0 * cos_l**2 - 1.0
        sin2l_term = 2.0 * sin_l * cos_l
        sin2k_term = 2.0 * sin_k * cos_k
        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)
        cos2_phi = tf.cos(2.0 * phi)
        sin2_phi = tf.sin(2.0 * phi)

        FL, S3, S9, AFB, S4, S7, S5, S8 = self._get_physical_coeffs()
        
        term1 = 0.75 * (1.0 - FL) * sin2_k
        term2 = FL * cos2_k
        term3 = 0.25 * (1.0 - FL) * sin2_k * cos2l_term
        term4 = -1.0 * FL * cos2_k * cos2l_term
        term5 = S3 * sin2_k * sin2_l * cos2_phi
        term6 = S4 * sin2k_term * sin2l_term * cos_phi
        term7 = S5 * sin2k_term * sin_l * cos_phi
        term8 = (4.0/3.0) * AFB * sin2_k * cos_l
        term9 = S7 * sin2k_term * sin_l * sin_phi
        term10 = S8 * sin2k_term * sin2l_term * sin_phi
        term11 = S9 * sin2_k * sin2_l * sin2_phi
        
        pdf = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11
        return pdf

    @zfit.supports(norm=True)
    def _integrate(self, limits, norm_range, options=None):
            """
            Integral analítica constante.
            Se añade 'options=None' para compatibilidad con las llamadas internas de zfit.
            """
            integral_value = 32.0 * np.pi / 9.0
            return tf.constant([integral_value], dtype=self.dtype)
    


def get_inverse_values(phys_params):
    FL, S3, S9, AFB, S4, S7, S5, S8 = phys_params
    def atanh(x): return np.arctanh(np.clip(x, -0.9999, 0.9999))
    
    rFL = atanh(2.0 * FL - 1.0)
    FT = 1.0 - FL
    rS3 = atanh(S3 / (0.5 * FT))
    
    R2_trans = 0.25 * FT**2 - S3**2
    R_trans = np.sqrt(max(1e-15, R2_trans))
    rS9 = atanh(S9 / R_trans)
    
    R2_rem_AFB = R2_trans - S9**2
    limit_AFB = 1.5 * np.sqrt(max(1e-15, R2_rem_AFB))
    rAFB = atanh(AFB / limit_AFB)
    
    bound_mix1 = FL * (FT - 2.0 * S3)
    limit_S4 = 0.5 * np.sqrt(max(1e-15, bound_mix1))
    rS4 = atanh(S4 / limit_S4)
    
    R2_S7 = bound_mix1 - 4.0 * S4**2
    limit_S7 = np.sqrt(max(1e-15, R2_S7))
    rS7 = atanh(S7 / limit_S7)
    
    bound_mix2 = FL * (FT + 2.0 * S3)
    limit_S5 = np.sqrt(max(1e-15, bound_mix2))
    rS5 = atanh(S5 / limit_S5)
    
    R2_S8 = bound_mix2 - S5**2
    limit_S8 = 0.5 * np.sqrt(max(1e-15, R2_S8))
    rS8 = atanh(S8 / limit_S8)
    
    return [rFL, rS3, rS9, rAFB, rS4, rS7, rS5, rS8]


def apply_transformation_equations(rFL, rS3, rS9, rAFB, rS4, rS7, rS5, rS8):
    """
    Aplica las ecuaciones de transformación de espacio TRANSFORMADO a FISICO.
    """
    FL = 0.5 * (1.0 + np.tanh(rFL))
    FT = 1.0 - FL
    
    limit_S3 = 0.5 * FT
    S3 = limit_S3 * np.tanh(rS3)
    
    R2_trans = np.maximum(0.25 * FT**2 - S3**2, 1e-15)
    R_trans = np.sqrt(R2_trans)
    S9 = R_trans * np.tanh(rS9)
    
    R2_rem_AFB = np.maximum(R2_trans - S9**2, 1e-15)
    limit_AFB = 1.5 * np.sqrt(R2_rem_AFB)
    AFB = limit_AFB * np.tanh(rAFB)
    
    bound_mix1 = np.maximum(FL * (FT - 2.0 * S3), 1e-15)
    limit_S4 = 0.5 * np.sqrt(bound_mix1)
    S4 = limit_S4 * np.tanh(rS4)
    
    R2_S7 = np.maximum(bound_mix1 - 4.0 * S4**2, 1e-15)
    S7 = np.sqrt(R2_S7) * np.tanh(rS7)
    
    bound_mix2 = np.maximum(FL * (FT + 2.0 * S3), 1e-15)
    limit_S5 = np.sqrt(bound_mix2)
    S5 = limit_S5 * np.tanh(rS5)
    
    R2_S8 = np.maximum(bound_mix2 - S5**2, 1e-15)
    limit_S8 = 0.5 * np.sqrt(R2_S8)
    S8 = limit_S8 * np.tanh(rS8)
    
    return {'FL': FL, 'S3': S3, 'S9': S9, 'AFB': AFB, 'S4': S4, 'S7': S7, 'S5': S5, 'S8': S8}

def get_physical_region_scan(n_points=100000):
    """Genera la nube de puntos de la región física."""
    r_vals = {
        'rFL': np.random.uniform(-5, 5, n_points),
        'rS3': np.random.uniform(-5, 5, n_points),
        'rS9': np.random.uniform(-5, 5, n_points),
        'rAFB': np.random.uniform(-5, 5, n_points),
        'rS4': np.random.uniform(-5, 5, n_points),
        'rS7': np.random.uniform(-5, 5, n_points),
        'rS5': np.random.uniform(-5, 5, n_points),
        'rS8': np.random.uniform(-5, 5, n_points)
    }
    # transforma a físico
    phys_dict = apply_transformation_equations(**r_vals)
    return pd.DataFrame(phys_dict)