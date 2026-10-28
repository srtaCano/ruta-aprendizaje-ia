# 02 · Predicción del precio de renta de vivienda en Mumbai

**¿Cuánto cuesta arrendar una vivienda en Mumbai según su tamaño, ubicación y número de baños?** Comparo tres enfoques muy distintos para resolver el mismo problema de regresión: una red neuronal, un ensamble de árboles y un algoritmo genético programado desde cero.

## Fuente de datos

Indian Housing – Mumbai (Kaggle): 5.000 anuncios de arriendo con tamaño, ubicación, coordenadas, número de baños y balcones, precio en rupias (INR), entre otras variables.


## Herramientas

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn

## Proceso

### 1. Limpieza y análisis exploratorio
- ElimSe eliminarion registros duplicados y convertí el tamaño de texto (`"1,200 sq ft"`) a número.
- Imputacion de nulos con la mediana y descarté columnas vacías o irrelevantes (`priceSqFt` estaba 100 % vacía).
- Con la matriz de correlación se seleccionaron 4 variables: `house_size` (0,73), `numBathrooms` (0,58), `latitude` y `longitude`.
- Deteccion de outliers con el método IQR.
- El precio tenía una asimetría muy fuerte (skewness de 5,58), así que se aplico una **transformación logarítmica** que la redujo a 0,21.
- Datos finales: 4.021 viviendas, divididas en 88 % entrenamiento y 12 % prueba.

### 2. Red neuronal MLP
Arquitectura 4 → 64 → 32 → 1 con ReLU, optimizador Adam, pérdida MSE y `EarlyStopping` para detener el entrenamiento cuando la validación deja de mejorar.

### 3. Ensamble con Gradient Boosting
Comparacion de un árbol de decisión individual (baseline) contra `GradientBoostingRegressor`, que entrena árboles en secuencia corrigiendo los errores del anterior.

### 4. Algoritmo genético desde cero
Implementacion de las cinco etapas sin librerías especializadas: inicialización de la población, evaluación del fitness (MSE), selección por elitismo, cruce y mutación. El algoritmo busca los 5 coeficientes de una regresión lineal. Se usó como referencia la solución por mínimos cuadrados para saber cuál era el mejor resultado alcanzable y probé tres escenarios (100 a 1.000 generaciones, poblaciones de 100 a 300) y cinco tamaños de mutación.

## Resultados

Métricas sobre el conjunto de prueba (483 viviendas), en escala logarítmica del precio:

| Modelo | MSE | R² | Error medio en rupias |
|---|---|---|---|
| **Gradient Boosting** | **0,081** | **0,90** | **15.130 INR** |
| Red neuronal MLP | 0,114 | 0,86 | 15.548 INR |
| Algoritmo genético | 0,457 | 0,45 | 35.859 INR |

- **Gradient Boosting fue el mejor modelo** y además el más rápido: entrenó en 0,5 segundos frente a unos 16 segundos de la red neuronal y del algoritmo genético.
- **El algoritmo genético cumplió su objetivo:** llegó a un MSE de entrenamiento de 0,462, prácticamente igual al óptimo de la regresión lineal (0,459). Su R² es menor porque busca una recta, y la relación entre las variables y el precio no es lineal; eso explica por qué los modelos no lineales lo superan.
- El experimento con la mutación mostró el equilibrio entre **exploración y explotación**: saltos muy pequeños no alcanzan a recorrer el espacio de búsqueda, y saltos muy grandes no logran afinar la solución.

<!-- ## Capturas - despues

![Comparación de modelos](img/comparacion_mse.png)
![Real vs. predicho](img/real_vs_predicho.png)
![Curva de entrenamiento MLP](img/curva_mse_mlp.png) -->


## Limitaciones y próximos pasos

- Solo se usaron 4 variables. Codificar la ubicación (barrio) o el tipo de vivienda probablemente mejoraría las predicciones.
- Los hiperparámetros del Gradient Boosting y de la red quedaron fijos; una búsqueda con validación cruzada podría mejorarlos.


## Autores

Valentina Cano Meneses · Juan José Flores · Simón Tobón Ospina
