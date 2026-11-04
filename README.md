# Pokédex CNN

Este proyecto implementa un sistema de clasificación de imágenes de Pokémon mediante aprendizaje profundo con TensorFlow y Keras. Se utiliza una CNN propia como modelo base y una arquitectura ResNet50 con transfer learning y fine-tuning como modelo final.

La solución está organizada en cuatro notebooks que cubren:

- preparación y limpieza de imágenes
- análisis exploratorio del conjunto de datos
- entrenamiento y evaluación del modelo base
- entrenamiento del modelo final con las mejoras seleccionadas
- predicciones sobre imágenes cargadas por el usuario

El objetivo es desarrollar un flujo completo de clasificación multiclase, comparar una aproximación inicial con un modelo mejorado y documentar las decisiones tomadas durante la experimentación.

## Introducción

La clasificación de imágenes de Pokémon consiste en asignar una imagen a una de las 150 clases del dataset. Las diferencias de pose, fondo, resolución y estilo de representación introducen variaciones dentro de una misma clase, mientras que algunas categorías comparten características visuales similares.

Este repositorio tiene un enfoque académico y utiliza redes convolucionales para estudiar cómo la preparación de los datos, la arquitectura y los parámetros de entrenamiento influyen en el desempeño del clasificador.

## Objetivos del proyecto

- Preparar un conjunto de imágenes con entradas uniformes y particiones fijas.
- Implementar una CNN base para establecer una referencia de desempeño.
- Comparar arquitecturas preentrenadas y evaluar técnicas de mejora.
- Evaluar los modelos mediante métricas globales y por clase.
- Exportar los modelos y utilizarlos para generar predicciones con imágenes nuevas.

## Conjunto de datos

Se utiliza el dataset [pokemonclassification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification), publicado en Kaggle por `lantian773030`. La descarga se realiza mediante `kagglehub`, fijando la versión 1.

El conjunto original contiene 6,837 imágenes distribuidas en 150 clases. Después de revisar etiquetas y eliminar imágenes redundantes, se conservan 6,795 imágenes.

La preparación incluye validación de archivos, revisión de duplicados y conversión a PNG RGB de 224 × 224 píxeles. El redimensionamiento conserva la proporción de la imagen y utiliza relleno blanco cuando es necesario.

Los datos se dividen en entrenamiento, validación y prueba mediante estratificación por grupos. Las imágenes relacionadas que se detectan durante la revisión permanecen en una misma partición para reducir el riesgo de compartir copias o versiones similares entre entrenamiento y evaluación.

## Estructura del repositorio

```text
pokedex-ai/
├── src/
│   ├── 01_data_preparation.ipynb
│   ├── 02_base_model.ipynb
│   ├── 03_final_model.ipynb
│   └── 04_model_predictions.ipynb
├── README.md
├── pokedex_cnn_report.pdf
├── requirements.txt
└── .gitignore
```

### Descripción de archivos

- `src/01_data_preparation.ipynb`: descarga, análisis exploratorio, validación, limpieza, división y preprocesamiento de las imágenes. Exporta el dataset preparado.
- `src/02_base_model.ipynb`: construcción, entrenamiento, evaluación y exportación de la CNN base.
- `src/03_final_model.ipynb`: construcción, entrenamiento por fases, evaluación y exportación de ResNet50 con la configuración seleccionada en los experimentos.
- `src/04_model_predictions.ipynb`: carga del modelo base o final y generación de predicciones sobre imágenes nuevas.
- `pokedex_cnn_report.pdf`: documentación del proceso, experimentos, resultados y conclusiones.
- `requirements.txt`: dependencias para preparar un entorno local.
- `.gitignore`: exclusión de entornos virtuales, cachés, datos generados y modelos exportados.

## Requisitos

El entorno principal de ejecución es Google Colab. Se recomienda activar una GPU para entrenar los modelos.

Las dependencias principales son:

- TensorFlow y Keras
- NumPy y pandas
- Matplotlib
- scikit-learn y SciPy
- Pillow y CairoSVG
- kagglehub

Para trabajar localmente también se incluyen JupyterLab e ipykernel en `requirements.txt`.

## Cómo ejecutar el proyecto

### 1. Preparar el entorno

Clona o descarga este repositorio y abre los notebooks de `src/` en Google Colab. Para el entrenamiento, selecciona una GPU en **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.

Los notebooks de modelado y predicción permiten acceder a archivos de Google Drive. Ajusta las rutas de configuración para que correspondan con la ubicación de tus paquetes ZIP.

### 2. Preparar los datos

Ejecuta `01_data_preparation.ipynb` de principio a fin. Se generará un ZIP con:

- `images/`: imágenes preprocesadas.
- `splits.csv`: rutas, etiquetas, particiones, grupos y huellas de integridad.
- `class_names.json`: orden de las clases del modelo.

Guarda este paquete en una ubicación accesible para los notebooks de entrenamiento.

### 3. Entrenar los modelos

Configura `DATASET_ZIP` en `02_base_model.ipynb` y `03_final_model.ipynb` con la ruta del dataset preparado. Ejecuta cada notebook en orden, desde la configuración hasta la exportación.

Ambos modelos utilizan las mismas particiones. El modelo final se construye a partir de los pesos de ImageNet y entrena sus dos fases en cada ejecución completa.

Cada notebook exporta un ZIP con tres archivos:

- `base_model.keras` o `final_model.keras`: arquitectura y pesos del modelo.
- `class_names.json`: etiquetas en el orden de las salidas.
- `training_results.json`: configuración, historial y métricas de evaluación.

### 4. Realizar predicciones

Abre `04_model_predictions.ipynb` y configura `MODEL_ZIP` con el paquete del modelo base o final. Ejecuta la carga del modelo y después la carga de imágenes para obtener las clases con mayor puntuación.

Para probar otras imágenes, vuelve a ejecutar únicamente la celda de carga de imágenes y predicciones. El modelo permanece en memoria. Si deseas forzar su recarga, activa `RELOAD_MODEL` y ejecuta nuevamente la sección de carga del modelo.

### Ejecución local

Desde la raíz del repositorio, crea un entorno virtual:

```bash
python -m venv .venv
```

Actívalo en Linux o macOS:

```bash
source .venv/bin/activate
```

O en Windows con PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias y abre JupyterLab:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Los notebooks están preparados principalmente para Colab. Para ejecutarlos localmente, omite el montaje de Google Drive y cambia las rutas por ubicaciones locales. La carga interactiva de imágenes del notebook de predicciones utiliza `google.colab.files.upload()` y debe sustituirse por la lectura de archivos locales. CairoSVG puede requerir la instalación de Cairo en el sistema.

## Modelos implementados

### Modelo base

La CNN base se entrena desde cero y establece una referencia para evaluar las mejoras. Su arquitectura incluye escalado de píxeles, tres bloques convolucionales de 32, 64 y 128 filtros con max pooling, agrupación global de características y una capa densa de 128 unidades. La salida utiliza softmax para las 150 clases.

### Modelo final

El modelo final utiliza ResNet50 con pesos de ImageNet, GlobalAveragePooling2D, Dropout de 0.1 y una salida softmax.

El entrenamiento se realiza en dos fases:

1. **Entrenamiento inicial:** se entrena la capa de clasificación, conservando los pesos de ResNet50, con una tasa de aprendizaje de `1e-3`.
2. **Ajuste fino:** se ajusta el último bloque de ResNet50, identificado por `conv5_*`, junto con la salida, utilizando una tasa de aprendizaje de `1e-5`. Las capas de normalización conservan sus parámetros.

Se utiliza un batch size de 32. En ambas fases se guarda el checkpoint con menor pérdida de validación y se detiene el entrenamiento cuando deja de mejorar.

## Experimentos de mejora

La configuración final se seleccionó mediante experimentos sobre validación:

- Comparación de MobileNet, ResNet y EfficientNet mediante transfer learning.
- Evaluación de data augmentation sobre la arquitectura seleccionada.
- Ajuste fino de las últimas capas del modelo.
- Comparación de diferentes valores de Dropout.
- Evaluación del batch size y la tasa de aprendizaje del ajuste fino.

Se conservó ResNet50 con ajuste fino, Dropout de 0.1, batch size de 32 y sin la política de data augmentation evaluada. El procedimiento, los resultados y las decisiones de cada experimento se documentan en el reporte.

## Métricas de evaluación

Se calculan las siguientes métricas para entrenamiento, validación y prueba:

- Pérdida de entropía cruzada.
- Accuracy.
- Top-5 accuracy.
- Precision, recall y F1-score con promedio macro.

El promedio macro asigna el mismo peso a cada clase y complementa la exactitud global. También se examinan las métricas por clase, la matriz de confusión y ejemplos de predicciones incorrectas.

Las decisiones de mejora se toman con validación. El conjunto de prueba se utiliza para evaluar la configuración final, una vez terminada la selección de arquitectura e hiperparámetros.

## Notas

Este proyecto está orientado al aprendizaje y la experimentación. Los archivos originales, las imágenes preparadas y los modelos exportados se generan durante la ejecución y se mantienen fuera del control de versiones. El repositorio conserva los notebooks, la documentación y la configuración necesaria para reproducir el flujo.
