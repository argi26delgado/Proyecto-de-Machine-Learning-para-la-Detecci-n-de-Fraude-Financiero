import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Carga del dataset
df = pd.read_csv(r"C:\Users\delga\OneDrive\Escritorio\Data eng. & IA\Proyecto 1\CreditcardDataset.csv")

# Seleccionamos 2 variables para poder visualizar bien la red en 2D (ej. V1 y V2)
# O puedes usar todo el dataset si solo te interesa el cálculo numérico.
X_raw = df[['V1', 'V2']].values 

# 2. Escalado de datos
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 3. Hiperparámetros e Inicialización de Neuronas (W)
n_neuronas = 5
dim_entradas = X.shape[1]

# Inicializamos W eligiendo 5 puntos aleatorios de los datos
np.random.seed(42)
indices_iniciales = np.random.choice(len(X), n_neuronas, replace=False)
W = X[indices_iniciales].copy()
W_inicial = W.copy()  # Guardamos copia para la gráfica

epochs = 10
eta_inicial = 0.1

# 4. Bucle de Entrenamiento (Aprendizaje Competitivo)
for epoch in range(epochs):
    eta = eta_inicial * (1 - epoch / epochs)
    
    # Barajar datos en cada época
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for idx in indices:
        observacion = X[idx]
        
        # Distancia euclídea a cada neurona
        distancias = np.linalg.norm(W - observacion, axis=1)
        
        # Neurona ganadora (BMU)
        ganadora = np.argmin(distancias)
        
        # Actualización de la ganadora
        W[ganadora] += eta * (observacion - W[ganadora])

print("Entrenamiento completado.")

# 5. Visualización del Mapa de Pesos y Distribución
plt.figure(figsize=(10, 8))
sns.kdeplot(x=X[:, 0], y=X[:, 1], fill=True, cmap="Blues", thresh=0, levels=50, alpha=0.6)
plt.scatter(X[::10, 0], X[::10, 1], alpha=0.2, c="gray", label="Casos legales/bancarios (Muestra)")

plt.scatter(W_inicial[:, 0], W_inicial[:, 1], c='red', marker='x', s=100, linewidths=3, label="Pesos Iniciales")
plt.scatter(W[:, 0], W[:, 1], c='blue', marker='o', s=100, label="Pesos Finales")

colores = ['red', 'blue', 'green', 'orange', 'purple']
perfiles = ["Riesgo Bajo", "Riesgo Medio", "Fraude", "Empresa", "Particular"]

for j in range(len(W)):
    plt.annotate(perfiles[j], xy=(W[j, 0], W[j, 1]), xytext=(W_inicial[j, 0], W_inicial[j, 1]),
                 arrowprops=dict(arrowstyle="->", color=colores[j], lw=2, alpha=0.8))

plt.title(f"Resultado después de {epochs} épocas de entrenamiento")
plt.xlabel("Monto / Variable bancaria (Escalada)")
plt.ylabel("Riesgo / Variable Legal (Escalada)")
plt.legend()
plt.grid(True)
plt.show()

# 6. Detección de Anomalías / Score
def score_anomalia(x, W_matrix):
    return np.min(np.linalg.norm(W_matrix - x, axis=1))

nueva_transaccion = np.array([0.9, 0.95])
# Escalar la nueva transacción con el mismo scaler
nueva_transaccion_scaled = scaler.transform(nueva_transaccion.reshape(1, -1))[0]
score = score_anomalia(nueva_transaccion_scaled, W)

print(f"Score de anomalía: {score:.4f}")
if score > 0.3:
    print("⚠️ Posible fraude detectado")

# 7. Asignación de Perfiles y Gráfico de Barras por Neurona
labels = []
for x in X:
    ganadora = np.argmin(np.linalg.norm(W - x, axis=1))
    labels.append(ganadora)
labels = np.array(labels)

riesgo_promedio = np.zeros(n_neuronas)
conteo = np.zeros(n_neuronas)

for i, x in enumerate(X):
    bmu = labels[i]
    riesgo_promedio[bmu] += x[1]  # Variable Y (Riesgo Legal)
    conteo[bmu] += 1

riesgo_promedio /= (conteo + 1e-6)

plt.figure(figsize=(8, 5))
colores_mapa = plt.cm.coolwarm((riesgo_promedio - riesgo_promedio.min()) / (riesgo_promedio.max() - riesgo_promedio.min() + 1e-6))

plt.bar(range(n_neuronas), riesgo_promedio, color=colores_mapa, edgecolor='black')
plt.xticks(range(n_neuronas), [f"{perfiles[i]}\n({int(c)} casos)" for i, c in enumerate(conteo)])
plt.ylabel("Nivel de Riesgo Promedio")
plt.title("Riesgo Legal Identificado por Cada Neurona")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

