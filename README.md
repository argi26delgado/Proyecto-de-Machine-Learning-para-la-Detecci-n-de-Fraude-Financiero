# Detección de Fraude con Aprendizaje Competitivo (SOM simplificado)

Proyecto que implementa una red de **aprendizaje competitivo** (una versión simplificada de un
Mapa Autoorganizado / SOM) para agrupar transacciones bancarias y detectar posibles anomalías
o fraudes, usando el dataset de tarjetas de crédito.

## 📋 Descripción

El script:

1. Carga y escala 2 variables del dataset (`V1`, `V2`).
2. Entrena 5 neuronas mediante aprendizaje competitivo (winner-takes-all) durante 10 épocas.
3. Visualiza cómo se mueven los pesos de las neuronas desde su posición inicial hasta la final,
junto con la densidad de los datos.
4. Calcula un **score de anomalía** para una nueva transacción (distancia a la neurona más cercana).
5. Asigna cada transacción a un "perfil" (neurona ganadora) y grafica el riesgo promedio por perfil.

## 📁 Estructura del proyecto

```
proyecto-som-fraude/
├── data/
│   └── CreditcardDataset.csv   # (no incluido en el repo, ver abajo)
├── src/
│   └── som\_fraude.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Instalación

```bash
git clone https://github.com/TU\_USUARIO/proyecto-som-fraude.git
cd proyecto-som-fraude
pip install -r requirements.txt
```

## 📊 Datos

El dataset (`CreditcardDataset.csv`) **no está incluido** en el repositorio por su tamaño/sensibilidad.
Debes colocarlo manualmente en la carpeta `data/`:

```
data/CreditcardDataset.csv
```

Debe contener, como mínimo, las columnas `V1` y `V2`.

## ▶️ Uso

```bash
python src/som\_fraude.py
```

Esto generará dos imágenes en la raíz del proyecto:

* `resultado\_mapa\_pesos.png`
* `riesgo\_por\_neurona.png`

## 🧠 Perfiles de neuronas

|Neurona|Perfil|
|-|-|
|0|Riesgo Bajo|
|1|Riesgo Medio|
|2|Fraude|
|3|Empresa|
|4|Particular|

## 🛠️ Tecnologías

* Python 3
* NumPy / Pandas
* Matplotlib / Seaborn
* Scikit-learn (`StandardScaler`)

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT (ver `LICENSE`).

