# Producto de Tesis — Análisis Espectral Pre-Evento

Aplicación desarrollada como producto asociado a la tesis de Ingeniería Informática orientada al análisis espectral del terreno previo a eventos de fuego mediante imágenes Sentinel-2 y deep learning.

## Objetivo del producto

Integrar en una herramienta computacional reproducible los principales resultados obtenidos durante las tres fases del proyecto:

1. Construcción geoespacial de eventos y muestras a partir de datos NASA FIRMS.
2. Obtención y validación de parches Sentinel-2 multiespectrales.
3. Clasificación de parches pre-evento mediante un modelo de deep learning.

El producto permitirá explorar espacialmente los eventos y muestras del estudio, visualizar la información multiespectral de los parches Sentinel-2 y consultar los resultados generados por el modelo.

## Estado del desarrollo

Proyecto en construcción.

### Fases disponibles

- Fase 1: dataset geoespacial y metadata.
- Fase 2: dataset Sentinel-2 de 8 canales.
- Fase 3: modelo ResNet18 multicanal entrenado.
- Producto software: en desarrollo.

## Canales Sentinel-2 utilizados

- B2
- B3
- B4
- B8
- B11
- B12
- NDMI
- NBR

## Arquitectura general

```text
NASA FIRMS
    ↓
Construcción geoespacial
    ↓
Metadata de parches
    ↓
Sentinel-2 8 canales
    ↓
Deep Learning
    ↓
Aplicación de análisis interactivo