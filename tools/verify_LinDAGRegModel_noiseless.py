import os
import sys

import numpy as np
import networkx as nx
import time
import torch
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models import LinDAGRegModel
import src.utils as utils
import src.dag_utils as dagu

# 1. Configuración idéntica al experimento (pero sin ruido)
M = 2000
d_p = {
    'n_tries': 1, # Solo una prueba para verificar
    'p': 0.2,
    'N': 100,
    'M': M,
    'M_train': int(0.7 * M),
    'M_val': int(0.2 * M),
    'M_test': int(0.1 * M),
    'src_t': 'constant',
    'max_src_node': 25,
    'n_sources': 5,
    'n_p_x': 0.0, # SIN RUIDO
    'n_p_y': 0.0, # SIN RUIDO
    'max_GSO': 100,
    'min_GSO': 50,
    'n_GSOs': 25
}

# 2. Generar datos
print(f"Generando datos para N={d_p['N']} nodos, {d_p['n_GSOs']} GSOs reales, {d_p['n_sources']} fuentes...")
Adj, W, GSOs, Psi = utils.get_graph_data(d_p, get_Psi=True)

# get_signals usa internamente d_p['n_GSOs'] para elegir cuáles GSOs son activos
# Devuelve:
# - sel_GSOs: Los GSOs que realmente se usaron (shape: [25, N, N])
# - gsos_idx: Los índices de esos GSOs (shape: [25])
# Ah, `get_signals` NO está en utils.py, está definida en el notebook! 
# Voy a reimplementarla aquí rápidamente basándome en lo que leí del notebook.

def get_signals_local(d_p, GSOs, norm_y='l2_norm'):
    range_GSO = np.arange(d_p['min_GSO'], d_p['max_GSO'])
    gsos_idx = np.random.choice(range_GSO, size=d_p['n_GSOs'], replace=False)
    sel_GSOs = GSOs[gsos_idx]
    # create_diff_data está en dag_utils
    Yn_t, X_t, Y_t = dagu.create_diff_data(d_p['M'], sel_GSOs, d_p['max_src_node'], d_p['n_p_x'], d_p['n_p_y'],
                                           d_p['n_sources'], src_t=d_p['src_t'], norm_y=norm_y, torch_tensor=True, verb=False)
    
    X_data = {'train': X_t[:d_p['M_train']], 'val': X_t[d_p['M_train']:-d_p['M_test']], 'test': X_t[-d_p['M_test']:]}
    Y_data = {'train': Yn_t[:d_p['M_train']], 'val': Yn_t[d_p['M_train']:-d_p['M_test']],
              'test': Y_t[-d_p['M_test']:]}
        
    return X_data, Y_data, sel_GSOs, gsos_idx

# Usamos la función local
X_data, Y_data, sel_GSOs, sel_GSOs_idx = get_signals_local(d_p, GSOs)

print("\n--- Datos Generados (Con Normalización l2_norm) ---")
print(f"X train shape: {X_data['train'].shape}")
print(f"Y train shape: {Y_data['train'].shape}")
print(f"Índices de GSOs verdaderos ({len(sel_GSOs_idx)}): {np.sort(sel_GSOs_idx)}")

# --- ANÁLISIS DE NORMALIZACIÓN (Pedido por el usuario) ---
print("\n--- Análisis de Normalización ---")
# Recuperamos Y sin procesar para comparar
_, Y_raw_data, _, _ = get_signals_local(d_p, GSOs, norm_y=None)
Y_raw = Y_raw_data['train'].numpy().squeeze()
Y_norm = Y_data['train'].numpy().squeeze()

# 1. ¿Más grande o más pequeño?
raw_norms = np.linalg.norm(Y_raw, axis=1) # Normas por muestra (M,)
print(f"Norma L2 promedio de Y (crudo): {raw_norms.mean():.4f} (std: {raw_norms.std():.4f})")
print(f"  -> Rango de normas: [{raw_norms.min():.4f}, {raw_norms.max():.4f}]")

if raw_norms.mean() > 1:
    print(">> RESULTADO: La normalización hace los valores más PEQUEÑOS (divide por ~{:.2f})".format(raw_norms.mean()))
else:
    print(">> RESULTADO: La normalización hace los valores más GRANDES.")

# 2. ¿Por qué afecta tanto? (Varianza de factores de escala)
scaling_factors = 1.0 / raw_norms
print(f"\nFactores de escala (1/||y||): promedio={scaling_factors.mean():.4f}, std={scaling_factors.std():.4f}")
var_coefficient = scaling_factors.std() / scaling_factors.mean()
print(f"Coeficiente de variación de los factores: {var_coefficient*100:.2f}%")

if var_coefficient > 0.01:
    print(">> EXPLICACIÓN CRÍTICA: Cada muestra se escala por un número MUY DIFERENTE.")
    print(">> Esto rompe la linealidad global. Y_norm = H * X * D, donde D es diagonal pero variable con X.")
    print(">> El modelo lineal intenta aprender un H_promedio, pero no puede ajustar todas las muestras a la vez.")
else:
    print(">> La variación es pequeña, el efecto no debería ser tan drástico.")



# 3. Escenario 1: Modelo "Tramposo" (Sabe exactamente qué GSOs usar)
print("\n--- Escenario 1: Modelo con GSOs correctos (Oracle) ---")
# Seleccionamos solo las columnas de Psi correspondientes a los GSOs verdaderos
Psi_oracle = Psi[:, sel_GSOs_idx] 
print(f"Psi oracle shape: {Psi_oracle.shape}")

model_oracle = LinDAGRegModel(W, Psi_oracle)
model_oracle.fit(X_data['train'], Y_data['train'])
err_oracle, std_oracle = model_oracle.test(X_data['test'], Y_data['test'])
print(f"Error Test (Oracle, con norm): {err_oracle:.10f}")

# Chequeamos invertibilidad interna manualmente replicando la lógica de fit
X_np = X_data['train'].numpy()
X_freq = model_oracle.W_inv @ X_np.squeeze().T
M_samples = X_np.shape[0]
Zm = np.array([model_oracle.W @ np.diag(X_freq[:,m]) @ model_oracle.Psi for m in range(M_samples)])
ZZ_oracle = (Zm.transpose(0, 2, 1) @ Zm).sum(axis=0)
rank_oracle = np.linalg.matrix_rank(ZZ_oracle)
print(f"Rank de ZZ (Oracle): {rank_oracle}/{ZZ_oracle.shape[0]}")


# 4. Escenario 2: Modelo Real (Usa TODOS los GSOs, como en el experimento)
print("\n--- Escenario 2: Modelo Real (Todos los GSOs) ---")
Psi_all = Psi # Usa todos los 100 GSOs
print(f"Psi all shape: {Psi_all.shape}")

model_real = LinDAGRegModel(W, Psi_all)
model_real.fit(X_data['train'], Y_data['train'])
err_real, std_real = model_real.test(X_data['test'], Y_data['test'])
print(f"Error Test (Real, con norm): {err_real:.10f}")

# Chequeamos invertibilidad
Zm_real = np.array([model_real.W @ np.diag(X_freq[:,m]) @ model_real.Psi for m in range(M_samples)])
ZZ_real = (Zm_real.transpose(0, 2, 1) @ Zm_real).sum(axis=0)
rank_real = np.linalg.matrix_rank(ZZ_real)
print(f"Rank de ZZ (Real): {rank_real}/{ZZ_real.shape[0]}")

if rank_real < ZZ_real.shape[0]:
    print(">> CONFIRMADO: La matriz ZZ es singular (rank deficient).")
    print(">> El modelo está usando pseudo-inversa y encontrando una solución de mínima norma.")
    print(">> Esto explica por qué el error no es cero: hay infinitas soluciones que encajan en train, pero la de mínima norma no es la verdadera.")
else:
    print(">> La matriz ZZ es invertible. El error debería ser numérico.")

# Visualización de coeficientes aprendidos vs reales
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Coeficientes Reales (simulados)")
# No tenemos acceso directo a los coefs reales h aquí porque create_DAG_filter los genera dentro,
# pero sabemos que son no nulos en sel_GSOs_idx y 0 en el resto.
# Podríamos intentar recuperarlos si create_DAG_filter devolviera h, pero asumamos uniformes.
plt.stem(sel_GSOs_idx, np.ones_like(sel_GSOs_idx)) # Placeholder visual
plt.xlim(0, 100)

plt.subplot(1, 2, 2)
plt.title("Coeficientes Aprendidos (Modelo Real)")
plt.stem(np.arange(100), model_real.h)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('coefs_comparison.png')
print("\nGráfica guardada en coefs_comparison.png")

# 5. Escenario 3: Oracle SIN Normalización (Debería ser ~0)
print("\n--- Escenario 3: Oracle SIN Normalización ---")
# Usamos nuevos datos random porque create_diff_data no permite inyectar X determinista fácilmente
X_data_nn, Y_data_nn, sel_GSOs_nn, sel_GSOs_idx_nn = get_signals_local(d_p, GSOs, norm_y=None)

# Seleccionamos psi para los nuevos GSOs seleccionados
Psi_oracle_nn = Psi[:, sel_GSOs_idx_nn]

model_oracle_nn = LinDAGRegModel(W, Psi_oracle_nn) 
model_oracle_nn.fit(X_data_nn['train'], Y_data_nn['train'])
err_oracle_nn, _ = model_oracle_nn.test(X_data_nn['test'], Y_data_nn['test'])
print(f"Error Test (Oracle, SIN norm): {err_oracle_nn:.30f}")

if err_oracle_nn < 1e-10:
    print(">> CONFIRMADO: Sin normalización no lineal (y con GSOs correctos), el error es numéricamente cero.")

# --- NUEVOS ESCENARIOS: GRAFO TRANSPUESTO ---
print("\n--- Generando Grafo Transpuesto y sus GSOs ---")
# Adj es triangular inferior (A del paper), así que nx.from_numpy_array(Adj) crea el grafo con flechas 'al revés' (A^T)
# Esto es consistente con lo que hace el notebook.
dag_T = nx.from_numpy_array(Adj, create_using=nx.DiGraph())
print(f"Calculando Psi para el grafo transpuesto (N={d_p['N']})...")
Psi_T = np.array([dagu.compute_Dq(dag_T, i, d_p['N']) for i in range(d_p['N'])]).T
print(f"Psi_T shape: {Psi_T.shape}")


# 6. Escenario 4: Oracle Transpuesto (Norm Y)
# Usamos los índices verdaderos (del grafo original) pero sobre la base transpuesta.
# Esto debería ser un desastre.
print("\n--- Escenario 4: Oracle Transpuesto (Norm Y) ---")
Psi_oracle_T = Psi_T[:, sel_GSOs_idx]
print(f"Psi oracle T shape: {Psi_oracle_T.shape}")

model_oracle_T = LinDAGRegModel(W, Psi_oracle_T)
model_oracle_T.fit(X_data['train'], Y_data['train'])
err_oracle_T, _ = model_oracle_T.test(X_data['test'], Y_data['test'])
print(f"Error Test (Oracle Transpuesto, con norm): {err_oracle_T:.10f}")


# 7. Escenario 5: Real Transpuesto (Norm Y)
# Usamos TODOS los GSOs de la base transpuesta.
# Debería dar el MISMO error que el Escenario 2, confirmando invarianza.
print("\n--- Escenario 5: Real Transpuesto (Todos los GSOs) ---")
Psi_all_T = Psi_T
# Nota: LinDAGRegModel usa W para "filtrar" X e Y. W depende de Adj. 
# Si el modelo "transpuesto" usa la misma W (que viene del grafo original Adj), 
# entonces solamente cambiamos la base Psi.
model_real_T = LinDAGRegModel(W, Psi_all_T)
model_real_T.fit(X_data['train'], Y_data['train'])
err_real_T, _ = model_real_T.test(X_data['test'], Y_data['test'])
print(f"Error Test (Real Transpuesto, con norm): {err_real_T:.10f}")

if np.isclose(err_real, err_real_T, atol=1e-8):
    print(">> CONFIRMADO: El error del modelo Real Transpuesto es IDÉNTICO al del Real Original.")
else:
    print(f">> WARNING: Los errores difieren! Original: {err_real}, Transpuesto: {err_real_T}")


# 8. Escenario 6: Oracle Transpuesto SIN Normalización
# Igual que el 4, pero sin la capa de normalización que metía ruido en le Oracle original.
# Si la base transpuesta es 'incorrecta' para el proceso físico, esto NO debería dar 0.
print("\n--- Escenario 6: Oracle Transpuesto SIN Normalización ---")
Psi_oracle_T_nn = Psi_T[:, sel_GSOs_idx_nn]

model_oracle_T_nn = LinDAGRegModel(W, Psi_oracle_T_nn)
model_oracle_T_nn.fit(X_data_nn['train'], Y_data_nn['train'])
err_oracle_T_nn, _ = model_oracle_T_nn.test(X_data_nn['test'], Y_data_nn['test'])
print(f"Error Test (Oracle Transpuesto, SIN norm): {err_oracle_T_nn:.30f}")

if err_oracle_T_nn > 1e-5:
    print(">> CONFIRMADO: Incluso sin ruido/norm, usar la base transpuesta restringida da error alto.")
    print(">> Esto demuestra que la 'invarianza' solo existe si usas la base COMPLETA.")
else:
    print(">> Aún hay error. Algo más está pasando.")
