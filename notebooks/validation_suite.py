import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zfit
from zfit import z


from PDFs import FullAngular_Physical_PDF, FullAngular_Transformed_PDF

def plot_slice_3d(pdf, obs, param_name, slice_vars={'cosK': 0, 'phi': 0}, axis=None, color='blue', label=''):
    """
    Función auxiliar para graficar un 'slice' (corte) de la PDF 3D.
    Mantiene fijas 2 variables y varía la tercera.
    """
    if axis is None:
        fig, axis = plt.subplots()

    # Identificar cuál es la variable que vamos a graficar (la que no está en slice_vars)
    # obs tiene orden: cosL, cosK, phi
    
    # Crear un linspace para la variable de eje
    x_val = np.linspace(-1, 1, 1000)
    if 'phi' not in slice_vars: # Si graficamos phi, el rango es -pi a pi
         x_val = np.linspace(-np.pi, np.pi, 1000)
            
    # Construir los datos para las 3 dimensiones
    # Estructura: [cosL, cosK, phi]
    
    # Caso 1: Graficando cosThetaL (cosK y phi fijos)
    if 'cosK' in slice_vars and 'phi' in slice_vars:
        # data shape: (1000, 3) -> col0: var, col1: fijo, col2: fijo
        cosK_fix = np.full_like(x_val, slice_vars['cosK'])
        phi_fix  = np.full_like(x_val, slice_vars['phi'])
        data_np = np.stack([x_val, cosK_fix, phi_fix], axis=1)
        xlabel = r'$\cos\theta_l$'
        
    # Caso 2: Graficando cosThetaK (cosL y phi fijos)
    elif 'cosL' in slice_vars and 'phi' in slice_vars:
        cosL_fix = np.full_like(x_val, slice_vars['cosL'])
        phi_fix  = np.full_like(x_val, slice_vars['phi'])
        data_np = np.stack([cosL_fix, x_val, phi_fix], axis=1)
        xlabel = r'$\cos\theta_K$'
        
    # Caso 3: Graficando Phi (cosL y cosK fijos)
    elif 'cosL' in slice_vars and 'cosK' in slice_vars:
        cosL_fix = np.full_like(x_val, slice_vars['cosL'])
        cosK_fix = np.full_like(x_val, slice_vars['cosK'])
        data_np = np.stack([cosL_fix, cosK_fix, x_val], axis=1)
        xlabel = r'$\phi$'

    # Evaluar PDF
    # Convertir a tensor para zfit
    probs = pdf.pdf(data_np).numpy()
    
    axis.plot(x_val, probs, color=color, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel('Densidad de Probabilidad (Slice)')

def main_validation():
    # 1. Definir el Espacio Observable 3D
    # Orden crítico: cosThetaL, cosThetaK, phi
    cosL = zfit.Space('cosThetaL', limits=(-1, 1))
    cosK = zfit.Space('cosThetaK', limits=(-1, 1))
    phi  = zfit.Space('phi', limits=(-np.pi, np.pi))
    
    obs = cosL * cosK * phi

    # 2. Definir Parámetros Físicos (Valores del Modelo Estándar aprox)
    # Estos se usarán para la PDF Física "Verdadera"
    FL_gen  = zfit.Parameter('FL_gen', 0.7)
    S3_gen  = zfit.Parameter('S3_gen', -0.1)
    S9_gen  = zfit.Parameter('S9_gen', 0.0) # Pequeña violación CP o cero
    AFB_gen = zfit.Parameter('AFB_gen', 0.1)
    S4_gen  = zfit.Parameter('S4_gen', 0.15)
    S7_gen  = zfit.Parameter('S7_gen', 0.02)
    S5_gen  = zfit.Parameter('S5_gen', 0.2)
    S8_gen  = zfit.Parameter('S8_gen', -0.05)

    pdf_phys = FullAngular_Physical_PDF(obs, FL_gen, S3_gen, S9_gen, AFB_gen, S4_gen, S7_gen, S5_gen, S8_gen)

    # 3. Definir Parámetros Transformados (Para pruebas de estrés)
    # Inicializamos en 0 (que es el centro de la tanh)
    rFL = zfit.Parameter('rFL', 0.0)
    rS3 = zfit.Parameter('rS3', 0.0)
    rS9 = zfit.Parameter('rS9', 0.0)
    rAFB= zfit.Parameter('rAFB', 0.0)
    rS4 = zfit.Parameter('rS4', 0.0)
    rS7 = zfit.Parameter('rS7', 0.0)
    rS5 = zfit.Parameter('rS5', 0.0)
    rS8 = zfit.Parameter('rS8', 0.0)

    pdf_trans = FullAngular_Transformed_PDF(obs, rFL, rS3, rS9, rAFB, rS4, rS7, rS5, rS8)

    # ---------------------------------------------------------
    # PRUEBA 1: Comparación Visual (Slices)
    # ---------------------------------------------------------
    print("Generando gráficos de validación...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Slice 1: Ver cosThetaL (fijando los otros a 0)
    plot_slice_3d(pdf_phys, obs, 'cosL', slice_vars={'cosK':0, 'phi':0}, axis=axes[0], color='black', label='Física (SM)')
    plot_slice_3d(pdf_trans, obs, 'cosL', slice_vars={'cosK':0, 'phi':0}, axis=axes[0], color='red', label='Transformada (Raw=0)')
    axes[0].set_title('Proyección cosThetaL')
    axes[0].legend()

    # Slice 2: Ver cosThetaK
    plot_slice_3d(pdf_phys, obs, 'cosK', slice_vars={'cosL':0, 'phi':0}, axis=axes[1], color='black')
    plot_slice_3d(pdf_trans, obs, 'cosK', slice_vars={'cosL':0, 'phi':0}, axis=axes[1], color='red')
    axes[1].set_title('Proyección cosThetaK')

    # Slice 3: Ver Phi
    plot_slice_3d(pdf_phys, obs, 'phi', slice_vars={'cosL':0, 'cosK':0}, axis=axes[2], color='black')
    plot_slice_3d(pdf_trans, obs, 'phi', slice_vars={'cosL':0, 'cosK':0}, axis=axes[2], color='red')
    axes[2].set_title('Proyección Phi')
    
    plt.tight_layout()
    plt.savefig('Validacion_Fisica_vs_Trans.png')
    plt.close()

    # ---------------------------------------------------------
    # PRUEBA 2: Estrés de Parámetros (Como tu compañero)
    # ---------------------------------------------------------
    # Vamos a probar valores "locos" en la PDF transformada para asegurar que no explota
    print("Iniciando prueba de estrés de parámetros...")
    
    stress_values = [
        {'val': 0.0,  'color': 'green', 'label': 'Centro (0)'},
        {'val': 2.0,  'color': 'orange','label': 'Alto (2.0)'},
        {'val': -5.0, 'color': 'blue',  'label': 'Extremo (-5.0)'}, 
        {'val': 100.0,'color': 'red',   'label': 'Saturación (100)'}
    ]
    
    fig_stress, ax_stress = plt.subplots(figsize=(8, 6))
    
    for setup in stress_values:
        val = setup['val']
        # Ponemos TODOS los parámetros raw en este valor para estresar el sistema
        rFL.set_value(val)
        rAFB.set_value(val) 
        rS5.set_value(val) # Probamos mover S5 que es clave para P5'
        
        # Graficamos un slice (ej. cosThetaL)
        plot_slice_3d(pdf_trans, obs, 'cosL', slice_vars={'cosK':0.5, 'phi':0.5}, 
                      axis=ax_stress, color=setup['color'], label=setup['label'])
        
    ax_stress.set_title('Prueba de Estabilidad: Variando parámetros Raw')
    ax_stress.legend()
    plt.savefig('Validacion_Stress_Test.png')
    plt.close()
    
    print("¡Validación completada! Revisa las imágenes PNG generadas.")

if __name__ == "__main__":
    main_validation()