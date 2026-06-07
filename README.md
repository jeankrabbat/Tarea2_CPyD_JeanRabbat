# Tarea 2: Aprendizaje Distribuido para Computer Vision

## Descripción

Este proyecto implementa una Red Neuronal Convolucional (CNN) utilizando JAX y Flax NNX para clasificación de imágenes del dataset "Wonders of the World". Se realizan experimentos sistemáticos para analizar el impacto de diferentes hiperparámetros en el rendimiento del modelo.

## Requisitos

- Python 3.8+
- JAX
- Flax NNX
- Optax
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- scikit-learn
- Google Colab (recomendado para GPU/TPU)

## Instalación

Si estás usando Google Colab, ejecuta en una celda:

```python
!pip install jax flax optax numpy pandas matplotlib pillow scikit-learn
```

Para instalación local, asegúrate de tener JAX configurado correctamente para tu sistema operativo.

## Ejecución

1. Abre el notebook `Tarea2_JeanRabbat.ipynb` en Google Colab
2. Monta Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ejecuta las celdas en orden:
   - PARTE 1: Importaciones
   - PARTE 2: Google Drive
   - PARTE 3: Preparación de datos
   - PARTE 4: Definición de modelos
   - PARTE 5: Funciones de entrenamiento
   - PARTE 7: Experimentos (Batch Size, Learning Rate, Network Size, Precisión)

4. Los resultados se generarán automáticamente como tablas y gráficos

## Estructura del Proyecto

```
Tarea2_JeanRabbat/
├── Tarea2_JeanRabbat.ipynb          # Notebook principal
├── README.md                          # Este archivo
├── Informe_Tarea2.pdf                # Informe completo con resultados
└── Wonders of World/                  # Dataset
    ├── Clase 1/
    ├── Clase 2/
    └── ...
```

*Nota: Los gráficos (experimentos_completos.png) se generan automáticamente durante la ejecución del notebook.*

## Dataset

El dataset utilizado es "Wonders of the World Image Classification" disponible en Kaggle.

**Descarga:** https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification

- **12 clases** (Maravillas del mundo)
- **3,846 imágenes totales**
- **División:** 70% entrenamiento, 15% validación, 15% test
- **Tamaño de imagen:** 224×224 píxeles
- **Normalización:** [0, 1]

## Configuración del Entorno

El código detecta automáticamente dispositivos disponibles (GPU/TPU). Verifica con:

```python
import jax
print(jax.devices())
```

## Autor

Jean Carlo Rabbat Sánchez

## Fecha

Junio 2026

## Profesor

Johansell Villalobos Cubillo

## Materia

Computación Paralela y Distribuida
