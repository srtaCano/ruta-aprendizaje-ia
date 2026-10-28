# Ruta de aprendizaje en inteligencia artificial

Proyectos que he desarrollado mientras aprendo inteligencia artificial en la Institución Universitaria de Envigado (IUE), ordenados de lo más básico a lo más avanzado. Cada carpeta es un proyecto independiente con su notebook, resultados y explicación.

## Proyectos

| # | Proyecto | Tipo de problema | Técnicas | Resultado principal |
|---|---|---|---|---|
| 01 | [Clasificación de tumores de mama](01-clasificacion-breast-cancer/) | Clasificación binaria | KNN, SVM, MLP | 97,4 % de accuracy (MLP) |
| 02 | [Predicción de renta de vivienda en Mumbai](02-prediccion-renta-mumbai/) | Regresión | Red neuronal, Gradient Boosting, algoritmo genético desde cero | R² de 0,90 (Gradient Boosting) |
| 03 | [Clasificación de imágenes de maravillas del mundo](03-clasificacion-imagenes-maravillas/) | Visión por computador, 12 clases | CNN, data augmentation, regularización | 78,1 % de accuracy en prueba |

## Qué muestra esta ruta

- **01:** los fundamentos del aprendizaje supervisado: separar entrenamiento y prueba, escalar datos y leer una matriz de confusión.
- **02:** un flujo completo con datos reales y sucios: limpieza, análisis exploratorio, transformación de la variable objetivo y comparación de tres enfoques muy distintos.
- **03:** trabajo con imágenes, redes convolucionales y experimentos controlados para entender y reducir el sobreajuste.

## Herramientas

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn · Google Colab

## Cómo ejecutar los proyectos

Cada notebook se puede abrir en Google Colab sin instalar nada. Para correrlos localmente:

```bash
git clone https://github.com/srtaCano/ruta-aprendizaje-ia.git
cd ruta-aprendizaje-ia
pip install -r requirements.txt
```

Las instrucciones para descargar los datos de cada proyecto están en su propia carpeta.

## Créditos

Los proyectos se desarrollaron en equipo como parte de los cursos de la universidad:

**Valentina Cano Meneses** · [GitHub](https://github.com/srtaCano) · [LinkedIn](www.linkedin.com/in/valentina-cano-meneses)

**Juan José Flores** 


**Simón Tobón Ospina** 


Este repositorio seguirá creciendo con nuevos proyectos a medida que avance en mi formación.
