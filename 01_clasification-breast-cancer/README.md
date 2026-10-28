# 01 · Clasificación de tumores de mama con KNN, SVM y MLP

Mi primer proyecto de machine learning: **¿se puede predecir si un tumor es maligno o benigno a partir de sus características físicas?** Comparo tres algoritmos clásicos de clasificación sobre el mismo conjunto de datos.

## Fuente de datos

[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), incluido en scikit-learn. Tiene 569 tumores descritos por 30 características (radio, textura, perímetro, etc.) y dos clases: maligno y benigno.

## Herramientas

Python · scikit-learn · Matplotlib · Seaborn

## Proceso

1. Separación los datos en 80 % entrenamiento y 20 % prueba (114 tumores).
2. Estandarización de las características con `StandardScaler`, ajustándolo solo con los datos de entrenamiento, porque KNN, SVM y las redes neuronales son sensibles a la escala.
3. Entrenamiento de tres modelos:
   - **KNN** con 5 vecinos.
   - **SVM** con kernel lineal.
   - **MLP** con una capa oculta de 100 neuronas, activación ReLU y optimizador Adam.
4. Evaluacion de cada uno con accuracy, reporte de clasificación y matriz de confusión.
5. Documentación en el notebook los conceptos clave: generalización, subajuste y sobreajuste, y cómo leer precision, recall y F1.

## Resultados

| Modelo | Accuracy | Recall en tumores malignos |
|---|---|---|
| **MLP** | **97,4 %** | 95 % |
| SVM (lineal) | 95,6 % | 95 % |
| KNN (k = 5) | 94,7 % | 93 % |

En un problema médico, el error más grave es clasificar un tumor maligno como benigno. Por eso, además del accuracy, me fijé en el **recall de la clase maligna**: el MLP y el SVM detectan 41 de los 43 tumores malignos del conjunto de prueba.

<!-- ## Capturas - añadir luego

![Comparativa de modelos](img/comparativa_modelos.png)
![Matriz de confusión MLP](img/matriz_confusion_mlp.png) -->

## Cómo ejecutarlo

Abre `notebooks/clasificacion_breast_cancer.ipynb` en Google Colab o Jupyter. No necesita descargar datos porque el dataset viene con scikit-learn.

## Limitaciones y próximos pasos

- La evaluación usa una sola división de los datos; con validación cruzada los resultados serían más confiables.
- Los hiperparámetros (número de vecinos, kernel, capas) quedaron en valores por defecto. Probaría `GridSearchCV` para optimizarlos.


## Autores

Valentina Cano Meneses · Juan José Flores · Simón Tobón Ospina
