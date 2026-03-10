#!/usr/bin/env python3
"""
compute_detG.py

Lee archivos JSON de resultados de fit en:
    fit_results/gen_phy/bin{b}/fit_results_gen_physical_bin{b}.json

para b en [1,2,3,4,5,7,9,10], extrae los valores centrales de:
    FL, S3, S9, AFB, S4, S7, S5, S8

y evalúa la expresión (determinante del Gram) dada por:

  detG =
    1/4 * F_L * (1 - F_L)**2
    - F_L * S_3**2
    - F_L * (4/9 * A_FB**2 + S_9**2)
    - 1/4 * (1 - F_L + 2 S_3) * (4 S_4**2 + S_7**2)
    - 1/4 * (1 - F_L - 2 S_3) * (S_5**2 + 4 S_8**2)
    - [ (2/3) A_FB (2 S_4 S_5 + 2 S_7 S_8) - S_9 (4 S_4 S_8 - S_5 S_7) ]

Salida:
 - imprime detG por bin con precisión
 - guarda CSV "detG_by_bin.csv"
"""
import json
import os
import math
import csv

BINS = [1, 2, 3, 4, 5, 7, 9, 10]
BASE_PATH = "fit_results/gen_phy_slsqp"
OUT_CSV = "detG_by_bin.csv"

# parámetros esperados (nombres base para buscar en JSON)
PARAM_NAMES = ["FL", "S3", "S9", "AFB", "S4", "S7", "S5", "S8"]

def find_param_key(parameters_dict, base_name, bin_index):
    """
    Busca la clave dentro de parameters_dict que corresponda al parámetro base_name
    para el bin dado. Ejemplos de claves esperadas: 'FL_bin1', 'S3_bin1', 'AFB_bin1', ...
    La búsqueda es tolerante (case-insensitive y busca contains/startwith).
    """
    target_frag = f"{base_name}_bin{bin_index}".lower()
    # primera pasada: coincidencia exacta (lower)
    for k in parameters_dict.keys():
        if k.lower() == target_frag:
            return k
    # segunda pasada: startswith
    for k in parameters_dict.keys():
        if k.lower().startswith(base_name.lower()) and f"bin{bin_index}" in k.lower():
            return k
    # tercera pasada: contains both tokens (base and bin)
    for k in parameters_dict.keys():
        kl = k.lower()
        if base_name.lower() in kl and f"bin{bin_index}" in kl:
            return k
    # cuarta: contains base name anywhere (fallback)
    for k in parameters_dict.keys():
        if base_name.lower() in k.lower():
            return k
    return None

def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_central_values(data, bin_index):
    """
    Retorna dict con valores centrales para PARAM_NAMES si existen.
    Si falta alguno, lo deja como None y devuelve lista de misses.
    """
    params = data.get("parameters", {})
    values = {}
    missing = []
    for pname in PARAM_NAMES:
        key = find_param_key(params, pname, bin_index)
        if key is None:
            values[pname] = None
            missing.append(pname)
            continue
        entry = params.get(key)
        if entry is None:
            values[pname] = None
            missing.append(pname)
            continue
        # asumimos el campo "value" existe
        values[pname] = entry.get("value")
        if values[pname] is None:
            missing.append(pname)
    return values, missing

def compute_detG(vals):
    """
    Evalúa la expresión del determinante dada en la conversación.
    Se espera vals sea un dict con FL,S3,S9,AFB,S4,S7,S5,S8
    """
    FL = vals["FL"]
    S3 = vals["S3"]
    S9 = vals["S9"]
    AFB = vals["AFB"]
    S4 = vals["S4"]
    S7 = vals["S7"]
    S5 = vals["S5"]
    S8 = vals["S8"]

    term1 = 0.25 * FL * (1.0 - FL)**2
    term2 = - FL * (S3**2)
    term3 = - FL * ( (4.0/9.0) * (AFB**2) + (S9**2) )
    term4 = - 0.25 * (1.0 - FL + 2.0*S3) * (4.0*(S4**2) + (S7**2))
    term5 = - 0.25 * (1.0 - FL - 2.0*S3) * ((S5**2) + 4.0*(S8**2))
    term6 = - ( (2.0/3.0)*AFB*(2.0*S4*S5 + 2.0*S7*S8) - S9*(4.0*S4*S8 - S5*S7) )

    detG = term1 + term2 + term3 + term4 + term5 + term6
    return detG

def main():
    results = []
    for b in BINS:
        file_path = os.path.join(BASE_PATH, f"bin{b}", f"fit_results_gen_physical_slsqp_bin{b}.json")
        if not os.path.exists(file_path):
            print(f"[WARN] Bin {b}: archivo no encontrado: {file_path}")
            results.append((b, None, "file_not_found"))
            continue

        try:
            data = load_json_file(file_path)
        except Exception as e:
            print(f"[ERROR] Bin {b}: error leyendo JSON: {e}")
            results.append((b, None, "read_error"))
            continue

        vals, missing = extract_central_values(data, b)
        if missing:
            print(f"[WARN] Bin {b}: faltan parámetros: {missing}. Se omite este bin.")
            results.append((b, None, f"missing:{','.join(missing)}"))
            continue

        # aseguremos que todos son floats
        try:
            for k in PARAM_NAMES:
                vals[k] = float(vals[k])
        except Exception as e:
            print(f"[ERROR] Bin {b}: valor no convertible a float: {e}")
            results.append((b, None, "bad_value"))
            continue

        detG = compute_detG(vals)
        results.append((b, detG, "ok"))
        print(f"Bin {b:2d} : detG = {detG:.6e}")


    print(f"\nResultados guardados en: {OUT_CSV}")

if __name__ == "__main__":
    main()