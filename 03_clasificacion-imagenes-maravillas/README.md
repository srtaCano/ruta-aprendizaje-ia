# 03 · Clasificación de imágenes de las maravillas del mundo con CNN

**¿Puede una red neuronal convolucional reconocer qué monumento aparece en una foto?** Entrené una CNN desde cero para clasificar 12 maravillas del mundo y realicé experimentos para entender y controlar el sobreajuste.

## Fuente de datos

[Wonders of the World Image Classification (Kaggle)](https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification): 3.846 imágenes de 12 clases, entre ellas Machu Picchu, Torre Eiffel, Taj Mahal, Coliseo romano y Salto Ángel.

El notebook descarga los datos automáticamente con `kagglehub`.

## Herramientas

Python · TensorFlow/Keras · scikit-learn · NumPy · Pillow · Matplotlib · Seaborn

## Proceso

### 1. Preparación de las imágenes
- Revisé que no hubiera archivos corruptos y analicé los tamaños: iban de 99 a 5.000 píxeles de ancho.
- Corregí la rotación de las fotos tomadas con celular (`exif_transpose`).
- Redimensioné todas a 128 × 128 **sin deformarlas**, rellenando con bordes negros (`ImageOps.pad`).
- Dividí en 88 % entrenamiento y 12 % prueba, de forma estratificada para mantener la proporción de cada clase, y normalicé los píxeles a [0, 1].
- Verifiqué con `assert` que los conteos por clase coincidieran antes y después del procesamiento.

### 2. Modelo base
CNN con 3 bloques convolucionales (32, 64 y 128 filtros) con MaxPooling, una capa densa de 128 neuronas, Dropout de 0,5 y salida softmax de 12 clases. Se hizo el entrenamiento con Adam y los callbacks `EarlyStopping`, `ModelCheckpoint` y `ReduceLROnPlateau`.

### 3. Experimentos contra el sobreajuste
Las clases estaban desbalanceadas (Taj Mahal tenía 158 imágenes y Coliseo 394), así que se generaron imágenes nuevas con **data augmentation** (rotación, volteo, zoom, desplazamiento y brillo) hasta llevar cada clase a 300 en entrenamiento. Luego se compararon 10 configuraciones de regularización:

| Configuración | Acc. entrenamiento | Acc. validación | Brecha | ¿Sobreajuste? |
|---|---|---|---|---|
| Sin regularización | 0,999 | 0,837 | 0,163 | Sí |
| **Base (Dropout 0,5 + EarlyStopping)** | 0,888 | 0,841 | 0,047 | No |
| Augmentation + Dropout 0,5 + ES | 0,860 | 0,839 | 0,021 | No |
| BatchNorm + ES | 1,000 | 0,856 | 0,143 | Sí |
| Combinado (Aug + Dropout + L2 + BN + ES) | 0,750 | 0,852 | −0,102 | No |

*Se muestran 5 de las 10 configuraciones; la tabla completa está en el notebook.*

## Resultados

- **Accuracy en prueba: 78,1 %** sobre 462 imágenes que el modelo nunca vio (macro F1 de 0,77).
- Mejores clases: Pirámides de Giza (F1 0,88) y Coliseo romano (F1 0,87).
- Clase más difícil: Estatua de la Libertad, con recall de 0,52; el modelo la confunde con otras clases en casi la mitad de los casos.
- Sin regularización el modelo memoriza los datos (99,9 % en entrenamiento) pero no generaliza mejor. Dropout y EarlyStopping redujeron la brecha de 16 a menos de 5 puntos.

<!-- ## Capturas -  despues

![Curvas de entrenamiento](img/curvas_loss_accuracy.png)
![Matriz de confusión](img/matriz_confusion.png)
![Ejemplos de data augmentation](img/data_augmentation.png) -->

## Cómo ejecutarlo

Abre `notebooks/clasificacion_maravillas.ipynb` en Google Colab con GPU activada (Entorno de ejecución → Cambiar tipo de entorno → GPU). El dataset se descarga solo.

## Limitaciones y próximos pasos

- Hay una diferencia entre validación (84 %) y prueba (78 %), lo que indica que la validación de un solo bloque fue algo optimista.
- Una CNN entrenada desde cero tiene un techo con pocos datos. El siguiente paso sería **transfer learning** con un modelo preentrenado como MobileNetV2 o EfficientNet.
- Probar tamaños de imagen mayores (160 o 180 píxeles) para ver si mejora la precisión.

## Autores

Valentina Cano Meneses · Juan José Flores · Simón Tobón Ospina
