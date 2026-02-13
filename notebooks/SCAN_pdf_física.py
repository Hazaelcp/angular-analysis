

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
























import zfit
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zfit.z as z
# Asegúrate de tener tu archivo PDFs.py en la misma carpeta o define la clase aquí
# from PDFs import FullAngular_Physical_PDF 

# --- DEFINICIÓN DE LA CLASE PDF (Incluida para que el script sea autónomo) ---
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
        
        # Versión segura para evitar NaN en raices cuadradas
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
        
        # Términos de la PDF
        term1 = 0.75 * (1.0 - FL) * sin2_k
        term2 = FL * cos2_k
        term3 = 0.25 * (1.0 - FL) * sin2_k * cos2l_term
        term4 = -1.0 * FL * cos2_k * cos2l_term
        term5 = S3 * sin2_k * sin2_l * cos2_phi
        term6 = S4 * sin2k_term * sin2l_term * cos_phi
        term7 = S5 * sin2k_term * sin_l * cos_phi
        term8 = (4.0/3.0) * AFB * sin2_k * cos_l # Signo positivo según paper LHCb
        term9 = S7 * sin2k_term * sin_l * sin_phi
        term10 = S8 * sin2k_term * sin2l_term * sin_phi
        term11 = S9 * sin2_k * sin2_l * sin2_phi
        
        pdf = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11
        return pdf

# --- DATOS EXPERIMENTALES (LHCb) ---
experimental_data = [
    {
        'label': '0.10 < q² < 0.98',
        'params': {
            'FL':  (0.255, 0.032, 0.007), 'S3':  (0.034, 0.044, 0.003),
            'S4':  (0.059, 0.050, 0.004), 'S5':  (0.227, 0.041, 0.008),
            'AFB': (-0.004, 0.040, 0.004), 'S7':  (0.006, 0.042, 0.002),
            'S8':  (-0.003, 0.051, 0.001), 'S9':  (-0.055, 0.041, 0.002)
        }
    },
    {
        'label': '1.1 < q² < 2.5',
        'params': {
            'FL':  (0.655, 0.046, 0.017), 'S3':  (-0.107, 0.052, 0.003),
            'S4':  (-0.038, 0.070, 0.011), 'S5':  (0.174, 0.060, 0.007),
            'AFB': (-0.229, 0.046, 0.009), 'S7':  (-0.107, 0.063, 0.004),
            'S8':  (-0.174, 0.075, 0.002), 'S9':  (-0.112, 0.054, 0.005)
        }
    },
    {
        'label': '2.5 < q² < 4.0',
        'params': {
            'FL':  (0.756, 0.047, 0.023), 'S3':  (0.020, 0.053, 0.002),
            'S4':  (-0.187, 0.074, 0.008), 'S5':  (-0.064, 0.068, 0.010),
            'AFB': (-0.070, 0.043, 0.006), 'S7':  (-0.066, 0.065, 0.004),
            'S8':  (0.016, 0.074, 0.002), 'S9':  (-0.012, 0.055, 0.003)
        }
    },
    {
        'label': '4.0 < q² < 6.0',
        'params': {
            'FL':  (0.684, 0.035, 0.015), 'S3':  (0.014, 0.038, 0.003),
            'S4':  (-0.145, 0.057, 0.004), 'S5':  (-0.204, 0.051, 0.013),
            'AFB': (0.050, 0.033, 0.002), 'S7':  (-0.136, 0.053, 0.002),
            'S8':  (0.077, 0.062, 0.001), 'S9':  (0.029, 0.045, 0.002)
        }
    },
    {
        'label': '6.0 < q² < 8.0',
        'params': {
            'FL':  (0.645, 0.030, 0.011), 'S3':  (-0.013, 0.038, 0.004),
            'S4':  (-0.275, 0.045, 0.006), 'S5':  (-0.279, 0.043, 0.013),
            'AFB': (0.110, 0.027, 0.005), 'S7':  (-0.074, 0.046, 0.003),
            'S8':  (-0.062, 0.047, 0.001), 'S9':  (0.024, 0.035, 0.002)
        }
    },
    {
        'label': '11.0 < q² < 12.5',
        'params': {
            'FL':  (0.461, 0.031, 0.010), 'S3':  (-0.124, 0.037, 0.003),
            'S4':  (-0.245, 0.047, 0.007), 'S5':  (-0.310, 0.043, 0.011),
            'AFB': (0.333, 0.030, 0.008), 'S7':  (-0.096, 0.050, 0.003),
            'S8':  (0.009, 0.049, 0.001), 'S9':  (0.042, 0.040, 0.003)
        }
    },
    {
        'label': '15.0 < q² < 17.0',
        'params': {
            'FL':  (0.352, 0.026, 0.009), 'S3':  (-0.166, 0.034, 0.007),
            'S4':  (-0.299, 0.033, 0.008), 'S5':  (-0.341, 0.034, 0.009),
            'AFB': (0.385, 0.024, 0.007), 'S7':  (0.029, 0.039, 0.001),
            'S8':  (0.003, 0.042, 0.002), 'S9':  (0.000, 0.037, 0.002)
        }
    }
]

# --- CONFIGURACIÓN ZFIT ---
cos_l = zfit.Space('cos_l', limits=(-1, 1))
cos_k = zfit.Space('cos_k', limits=(-1, 1))
phi   = zfit.Space('phi', limits=(-np.pi, np.pi))
obs = cos_l * cos_k * phi

p_map = {name: zfit.Parameter(name, 0.0) for name in ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']}
pdf_model = FullAngular_Physical_PDF(obs, **p_map)

# --- GENERACIÓN DE DATOS (ALTA ESTADÍSTICA) ---
# Usamos 100,000 puntos para ver bien las colas de las distribuciones
n_points = 100000 
print(f"Generando {n_points} puntos de espacio de fase uniforme...")
data_np = np.stack([
    np.random.uniform(-1, 1, n_points),
    np.random.uniform(-1, 1, n_points),
    np.random.uniform(-np.pi, np.pi, n_points)
], axis=-1)
data_zfit = zfit.Data.from_numpy(obs=obs, array=data_np)

# --- CONFIGURACIÓN DE GRÁFICOS ---
num_bins = len(experimental_data)
cols = 2
rows = (num_bins + 1) // cols 
fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes_flat = axes.flatten()

print("Comenzando evaluación de histogramas...")

# --- BUCLE DE PLOTEO ---
for i, case in enumerate(experimental_data):
    ax = axes_flat[i]
    ax.set_title(f"Bin: {case['label']}")
    
    # Definir escenarios
    scenarios = {
        'Central': {'color': 'green', 'factor': 0},
        'Lower (-1σ)': {'color': 'skyblue', 'factor': -1},
        'Upper (+1σ)': {'color': 'orange', 'factor': 1}
    }
    
    for label, props in scenarios.items():
        # Actualizar parámetros
        factor = props['factor']
        for param_name, values in case['params'].items():
            val, err_stat, err_syst = values
            err_tot = np.sqrt(err_stat**2 + err_syst**2)
            set_val = val + (factor * err_tot)
            p_map[param_name].set_value(set_val)
        
        # Calcular PDF
        probs = pdf_model.pdf(data_zfit, norm=False).numpy()
        
        # Plotear Histograma Transparente
        # Usamos 'stepfilled' con alpha para ver superposiciones
        ax.hist(probs, bins=50, range=(np.min(probs), np.percentile(probs, 99)), 
                alpha=0.4, label=label, color=props['color'], density=True)

    # Línea Roja en Cero (Límite Físico)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Límite Físico (0)')
    
    # Estética
    ax.set_xlabel("Valor de la PDF")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize='small')
    ax.grid(alpha=0.3)
    
    # Zoom automático cerca de cero si hay valores negativos
    # Esto ayuda a ver si la cola negativa es grande o pequeña
    xlims = ax.get_xlim()
    if xlims[0] < -0.1:
        ax.set_xlim(left=-0.1) # No mostrar demasiado espacio vacío negativo si hay outliers extremos

# Ocultar subplots vacíos si los hay
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

plt.tight_layout()
plt.show()






















################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################











# import zfit
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import zfit.z as z
# from PDFs import FullAngular_Physical_PDF 
# experimental_data = [
#     {
#         'label': '0.10 < q² < 0.98',
#         'params': {
#             'FL':  (0.255, 0.032, 0.007),
#             'S3':  (0.034, 0.044, 0.003),
#             'S4':  (0.059, 0.050, 0.004),
#             'S5':  (0.227, 0.041, 0.008),
#             'AFB': (-0.004, 0.040, 0.004),
#             'S7':  (0.006, 0.042, 0.002),
#             'S8':  (-0.003, 0.051, 0.001),
#             'S9':  (-0.055, 0.041, 0.002)
#         }
#     },
#     {
#         'label': '1.1 < q² < 2.5',
#         'params': {
#             'FL':  (0.655, 0.046, 0.017),
#             'S3':  (-0.107, 0.052, 0.003),
#             'S4':  (-0.038, 0.070, 0.011),
#             'S5':  (0.174, 0.060, 0.007),
#             'AFB': (-0.229, 0.046, 0.009),
#             'S7':  (-0.107, 0.063, 0.004),
#             'S8':  (-0.174, 0.075, 0.002),
#             'S9':  (-0.112, 0.054, 0.005)
#         }
#     },
#     {
#         'label': '2.5 < q² < 4.0',
#         'params': {
#             'FL':  (0.756, 0.047, 0.023),
#             'S3':  (0.020, 0.053, 0.002),
#             'S4':  (-0.187, 0.074, 0.008),
#             'S5':  (-0.064, 0.068, 0.010),
#             'AFB': (-0.070, 0.043, 0.006),
#             'S7':  (-0.066, 0.065, 0.004),
#             'S8':  (0.016, 0.074, 0.002),
#             'S9':  (-0.012, 0.055, 0.003)
#         }
#     },
#     {
#         'label': '4.0 < q² < 6.0',
#         'params': {
#             'FL':  (0.684, 0.035, 0.015),
#             'S3':  (0.014, 0.038, 0.003),
#             'S4':  (-0.145, 0.057, 0.004),
#             'S5':  (-0.204, 0.051, 0.013),
#             'AFB': (0.050, 0.033, 0.002),
#             'S7':  (-0.136, 0.053, 0.002),
#             'S8':  (0.077, 0.062, 0.001),
#             'S9':  (0.029, 0.045, 0.002)
#         }
#     },
#     {
#         'label': '6.0 < q² < 8.0',
#         'params': {
#             'FL':  (0.645, 0.030, 0.011),
#             'S3':  (-0.013, 0.038, 0.004),
#             'S4':  (-0.275, 0.045, 0.006),
#             'S5':  (-0.279, 0.043, 0.013),
#             'AFB': (0.110, 0.027, 0.005),
#             'S7':  (-0.074, 0.046, 0.003),
#             'S8':  (-0.062, 0.047, 0.001),
#             'S9':  (0.024, 0.035, 0.002)
#         }
#     },
#     {
#         'label': '11.0 < q² < 12.5',
#         'params': {
#             'FL':  (0.461, 0.031, 0.010),
#             'S3':  (-0.124, 0.037, 0.003),
#             'S4':  (-0.245, 0.047, 0.007),
#             'S5':  (-0.310, 0.043, 0.011),
#             'AFB': (0.333, 0.030, 0.008),
#             'S7':  (-0.096, 0.050, 0.003),
#             'S8':  (0.009, 0.049, 0.001),
#             'S9':  (0.042, 0.040, 0.003)
#         }
#     },
#     {
#         'label': '15.0 < q² < 17.0',
#         'params': {
#             'FL':  (0.352, 0.026, 0.009),
#             'S3':  (-0.166, 0.034, 0.007),
#             'S4':  (-0.299, 0.033, 0.008),
#             'S5':  (-0.341, 0.034, 0.009),
#             'AFB': (0.385, 0.024, 0.007),
#             'S7':  (0.029, 0.039, 0.001),
#             'S8':  (0.003, 0.042, 0.002),
#             'S9':  (0.000, 0.037, 0.002)
#         }
#     }
# ]

# cos_l = zfit.Space('cos_l', limits=(-1, 1))
# cos_k = zfit.Space('cos_k', limits=(-1, 1))
# phi   = zfit.Space('phi', limits=(-np.pi, np.pi))
# obs = cos_l * cos_k * phi

# p_map = {name: zfit.Parameter(name, 0.0) for name in ['FL', 'S3', 'S9', 'AFB', 'S4', 'S7', 'S5', 'S8']}
# pdf_model = FullAngular_Physical_PDF(obs, **p_map)

# # Generar datos de prueba
# n_points = 2000 
# data_np = np.stack([
#     np.random.uniform(-1, 1, n_points),
#     np.random.uniform(-1, 1, n_points),
#     np.random.uniform(-np.pi, np.pi, n_points)
# ], axis=-1)
# data_zfit = zfit.Data.from_numpy(obs=obs, array=data_np)

# results = {
#     'bin': [], 'central': [], 'upper': [], 'lower': []
# }

# print(f"{'BIN':<15} | {'TIPO':<10} | {'Mínimo PDF':<12} | {'Estado':<10}")
# print("-" * 60)

# for case in experimental_data:
#     results['bin'].append(case['label'])    
#     scenarios = ['central', 'upper', 'lower']
    
#     for sc in scenarios:
#         for param_name, values in case['params'].items():
#             val, err_stat, err_syst = values
#             err_tot = np.sqrt(err_stat**2 + err_syst**2)
            
#             if sc == 'central':
#                 set_val = val
#             elif sc == 'upper':
#                 set_val = val + err_tot
#             elif sc == 'lower':
#                 set_val = val - err_tot
            
#             p_map[param_name].set_value(set_val)
        
#         probs = pdf_model.pdf(data_zfit, norm=False).numpy()
#         min_val = np.min(probs)
        
#         results[sc].append(min_val)
        
#         # Imprimir
#         status = "OK" if min_val >= -1e-1 else "FAIL NEG"
#         print(f"{case['label']:<15} | {sc:<10} | {min_val:.5e}  | {status}")
#     print("-" * 60)

# x = np.arange(len(results['bin']))
# width = 0.25

# plt.figure(figsize=(12, 7))

# # Barras para cada escenario
# bars1 = plt.bar(x - width, results['lower'], width, label='Lower (-1σ)', color='skyblue', edgecolor='black')
# bars2 = plt.bar(x, results['central'], width, label='Central', color='limegreen', edgecolor='black')
# bars3 = plt.bar(x + width, results['upper'], width, label='Upper (+1σ)', color='orange', edgecolor='black')
# plt.axhline(0, color='red', linewidth=1.5, linestyle='--')

# plt.ylabel('Mínimo de la PDF')
# plt.xticks(x, results['bin'], rotation=45, ha='right')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# for i, vals in enumerate(zip(results['lower'], results['central'], results['upper'])):
#     min_of_bin = min(vals)
#     if min_of_bin < -0.05:
#         plt.text(i, min_of_bin - 0.05, "FAIL", ha='center', fontsize=12)

# plt.tight_layout()
# plt.show()





################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################




# import zfit
# from zfit import z
# import tensorflow as tf
# import numpy as np
# import pandas as pd 
# from PDFs import FullAngular_Physical_PDF 

# cos_l = zfit.Space('cos_l', limits=(-1, 1))
# cos_k = zfit.Space('cos_k', limits=(-1, 1))
# phi   = zfit.Space('phi',   limits=(-np.pi, np.pi))
# obs   = cos_l * cos_k * phi

# def get_physical_parameters():
#     """Genera parámetros que cumplen SOLO las condiciones 2x2 desiguales o constrainst"""
#     while True:
#         fl_val = np.random.uniform(0.3, 0.7) 
#         s3_val  = np.random.uniform(-0.3, 0.3)         
#         s9_val  = np.random.uniform(-0.2, 0.2); afb_val = np.random.uniform(-0.2, 0.2)
#         s4_val  = np.random.uniform(-0.2, 0.2); s7_val  = np.random.uniform(-0.2, 0.2)
#         s5_val  = np.random.uniform(-0.2, 0.2); s8_val  = np.random.uniform(-0.2, 0.2)

#         if not (0 <= fl_val <= 1): continue
#         if abs(s3_val) > 0.5 * (1 - fl_val): continue
        
#         #########################################
#         if s3_val**2 + (4.0/9.0)*afb_val**2 + s9_val**2 > 0.25 * (1 - fl_val)**2: continue
        
#         ##########################################
#         term_pos = (1 - fl_val + 2 * s3_val)
#         term_neg = (1 - fl_val - 2 * s3_val)
        
#         if term_neg < 1e-9 or term_pos < 1e-9: continue 

#         if 4 * s4_val**2 + s7_val**2 > fl_val * term_neg: continue
#         if s5_val**2 + 4 * s8_val**2 > fl_val * term_pos: continue

#         return {
#             'FL': fl_val, 'S3': s3_val, 'S9': s9_val, 'AFB': afb_val,
#             'S4': s4_val, 'S7': s7_val, 'S5': s5_val, 'S8': s8_val
#         }

# n_grid = 2000 # eventos del toy
# uniform_data_np = np.stack([
#     np.random.uniform(-1, 1, n_grid),
#     np.random.uniform(-1, 1, n_grid),
#     np.random.uniform(-np.pi, np.pi, n_grid)
# ], axis=-1)
# uniform_data = zfit.Data.from_numpy(obs=obs, array=uniform_data_np)

# # Inicialización de parámetros zfit
# params_container = {
#     'FL': zfit.Parameter('FL', 0.5), 'S3': zfit.Parameter('S3', 0.0),
#     'S9': zfit.Parameter('S9', 0.0), 'AFB': zfit.Parameter('AFB', 0.0),
#     'S4': zfit.Parameter('S4', 0.0), 'S7': zfit.Parameter('S7', 0.0),
#     'S5': zfit.Parameter('S5', 0.0), 'S8': zfit.Parameter('S8', 0.0)
# }

# pdf = FullAngular_Physical_PDF(obs, **params_container)

# failed_parameters_list = []
# TOLERANCIA = 1e-3 

# print(f"\n{'ITER':<5} | {'STATUS':<10} | {'MIN VALUE':<12} | {'NEGATIVES':<10}")
# print("-" * 55)

# n_iterations = 2000

# for i in range(n_iterations):
#     p = get_physical_parameters()
#     for key, val in p.items():
#         params_container[key].set_value(val)
        
#     probs = pdf.pdf(uniform_data).numpy()
#     min_val = np.min(probs)
    
#     negatives = np.sum(probs < -TOLERANCIA) 
#     status = "OK" if negatives == 0 else "FAIL"
    
#     if status == "FAIL":
#         print(f"\033[91m{i+1:<5} | {status:<10} | {min_val:.6e}  | {negatives:<10}\033[0m")
#         print("   >>> ¡PARAMETROS ERRONEOS ENCONTRADOS!")
#         print(f"   >>> FL: {p['FL']:.4f}, S3: {p['S3']:.4f}, AFB: {p['AFB']:.4f}, S9: {p['S9']:.4f}")
#         print(f"   >>> S4: {p['S4']:.4f}, S5: {p['S5']:.4f}, S7: {p['S7']:.4f}, S8: {p['S8']:.4f}")
#         print(f"   >>> Min Value: {min_val}")

#         p_fail = p.copy()
#         p_fail['min_value'] = min_val
#         p_fail['num_negatives'] = negatives
#         failed_parameters_list.append(p_fail)
        
#     else:
#         if i % 50 == 0:
#             print(f"{i+1:<5} | {status:<10} | {min_val:.6e}  | {negatives:<10}")

# if len(failed_parameters_list) > 0:
#     df_failed = pd.DataFrame(failed_parameters_list)
#     filename = "failed_params_physical_check.csv"
#     df_failed.to_csv(filename, index=False)
#     print(f"Se encontraron {len(failed_parameters_list)} configuraciones inválidas.")
#     print(f"\n\n parámetros guardados: {filename}")
# else:
#     print("\n\n[DONE] No se encontraron fallos.")