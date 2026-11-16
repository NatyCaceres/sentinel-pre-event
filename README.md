# Producto de tesis — Análisis espectral pre-evento

Aplicación Streamlit desarrollada como producto de la tesis de Ingeniería
Informática **“Análisis espectral del terreno previo al fuego mediante
Sentinel-2 y deep learning”**.

El sistema explora parches Sentinel-2 previos a eventos registrados y clasifica
sus patrones espectrales según su asociación con las clases definidas en el
estudio.

> **Alcance:** el resultado no predice incendios ni representa riesgo absoluto.

## Estado del MVP

El MVP local se encuentra funcional y validado:

- 1253 parches Sentinel-2 auditados físicamente.
- 244 parches pertenecientes a la prueba final congelada.
- 243 casos cartografiables y 244 disponibles mediante búsqueda directa.
- Modelo ResNet18 de 8 canales integrado.
- Reproducción de las métricas del cierre experimental.
- 121 pruebas automatizadas aprobadas.
- 7 `PendingDeprecationWarning` originados internamente por Rasterio 1.5.1.

## Flujo disponible

```text
Configurar exploración
        ↓
Explorar muestras en el mapa
        ↓
Seleccionar una muestra
        ↓
Revisar sus antecedentes
        ↓
Ejecutar el análisis espectral
        ↓
Interpretar las probabilidades por clase
```

La interfaz permite filtrar el conjunto de prueba por temporada, región,
clase, fecha y grupos multipatch. También permite buscar directamente por ID
de parche o grupo.

## Clases del estudio

- **Clase 1 — Con detección posterior:** zona con detecciones posteriores de
  fuego activo NASA FIRMS.
- **Clase 0 — Sin detección cercana:** zona comparable sin detecciones cercanas
  bajo los controles definidos.

Las probabilidades mostradas por la aplicación corresponden a estas clases y
no deben interpretarse como probabilidad de incendio.

## Contrato de los GeoTIFF

Cada parche debe cumplir:

```text
shape = (8, 64, 64)
dtype = float32
```

Orden de canales:

```text
B2, B3, B4, B8, B11, B12, NDMI, NBR
```

El contrato se encuentra en `config/spectral_contract.json`.

La carga física:

- rechaza archivos que no cumplan la forma o el tipo esperado;
- rechaza valores positivos o negativos infinitos;
- conserva los valores `NaN` para que la inferencia aplique la política
  documentada en su contrato.

La vista RGB utiliza B4/B3/B2 y un contraste percentil 2–98 exclusivamente
para visualización. Este ajuste no forma parte del preprocesamiento del modelo.

## Requisitos

- Python 3.13.15.
- Dependencias incluidas en `requirements.txt`.
- GeoTIFF del producto disponibles localmente.
- Checkpoint ResNet18 disponible localmente.

Versiones principales validadas:

```text
rasterio    1.5.1
streamlit   1.62.0
torch       2.13.0
torchvision 0.28.0
timm        1.0.29
```

## Instalación

Desde la raíz del repositorio, activa el entorno existente. Para una instalación
nueva y reproducible en Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La creación de un entorno nuevo solo es necesaria al instalar el proyecto en
otro equipo o reconstruir el entorno actual.

## Datos Sentinel-2

Por defecto, los GeoTIFF deben copiarse directamente en:

```text
data/patches/sentinel_8ch/
```

Los nombres registrados en `tif_filename` no deben modificarse.

También puede utilizarse otra ubicación mediante la variable
`SENTINEL_PATCH_DIR`:

```powershell
$env:SENTINEL_PATCH_DIR = "D:\datos\sentinel_8ch"
```

El catálogo operacional se encuentra en:

```text
data/metadata/product/product_catalog.csv
```

Los archivos `.tif` y `.tiff` están excluidos de Git por su tamaño.

## Modelo de inferencia

El checkpoint local esperado es:

```text
artifacts/models/resnet18_8ch_v2/best_resnet18_8ch_v2.pt
```

Su contrato operacional se encuentra en:

```text
artifacts/models/resnet18_8ch_v2/runtime/inference_contract.json
```

El contrato fija y valida:

- arquitectura ResNet18;
- 8 canales de entrada y 2 clases;
- orden de los canales;
- clipping por canal;
- imputación de `NaN` mediante la media de entrenamiento;
- normalización Z-score calculada únicamente con entrenamiento;
- dropout;
- umbral de decisión seleccionado en validación;
- hash SHA-256 del checkpoint.

Los archivos `.pt`, `.pth` y `.ckpt` están excluidos de Git. El checkpoint
debe proporcionarse por un medio externo y conservar exactamente su nombre y
contenido. Reemplazar solamente el `.pt` por otros pesos hará fallar la
verificación del contrato.

## Auditoría del almacenamiento

Para comparar el catálogo contra los TIFF disponibles y validar físicamente
cada archivo:

```powershell
python -m src.spectral.storage_audit
```

Para regenerar el manifest después de una auditoría completamente válida:

```powershell
python -m src.spectral.storage_audit --write-manifest
```

El manifest se guarda en:

```text
data/metadata/product/spectral_storage_manifest.json
```

## Ejecutar las pruebas

```powershell
python -m pytest -q
```

Resultado de referencia actual:

```text
166 passed, 7 warnings
```

## Reproducir la evaluación cerrada

Con los 1253 TIFF y el checkpoint disponibles:

```powershell
python -m src.spectral.evaluation
```

El comando evalúa los 244 casos de la prueba final congelada y compara las
métricas obtenidas con los artefactos de cierre. El resultado esperado es:

```text
Resultado: COINCIDE
```

## Ejecutar la aplicación

La preparación para Community Cloud y los pasos pendientes están documentados
en [docs/cloud_deployment.md](docs/cloud_deployment.md). El perfil Linux está en
`app/requirements.txt`; no sustituye al entorno local de Windows.

```powershell
python -m streamlit run app/app.py
```

La inferencia se ejecuta localmente en el proceso Python de Streamlit. El modelo
se mantiene en caché durante la sesión para evitar cargarlo nuevamente en cada
interacción.

## Explicación visual con Grad-CAM

En «Analizar muestra», primero pulsa «Analizar patrón espectral». Después abre
«Ver imagen RGB y explicación visual del parche» y pulsa «Generar explicación
visual». Se muestran RGB, Grad-CAM y su superposición, con una escala relativa
de 0 a 1. Se explica la clase asignada con el umbral congelado 0,495, aunque
difiera de la clase observada.

El cálculo reutiliza el checkpoint y el preprocesamiento de inferencia de los
ocho canales. Usa el logit de la clase elegida y la salida de `layer2.1`
(`layer2[-1]`): media espacial de gradientes, suma ponderada, ReLU,
interpolación bilineal a 64 × 64 (`align_corners=False`) y min-max por parche.
La resolución original es 8 × 8; ampliar el mapa no añade precisión espacial.
Los mapas constantes se muestran como ceros y se identifican en la interfaz.

El RGB conserva su contraste percentil 2–98 exclusivamente visual. La paleta
violeta–amarillo está definida en el código y la opacidad de superposición es
0,46. No se reentrena ni se modifican pesos, gradientes de parámetros o el
umbral. Los resultados permanecen en memoria de la sesión y se invalidan al
cambiar la muestra, sus datos o la predicción; no se guardan imágenes en disco.

Grad-CAM es un apoyo cualitativo para revisar la clasificación. No representa
probabilidad de incendio por píxel, no delimita áreas quemadas y sus intensidades
normalizadas no son comparables en términos absolutos entre parches.

## Estructura principal

```text
app/                         aplicación Streamlit
artifacts/models/            contrato, procedencia y checkpoint local
config/                      contratos y configuración del proyecto
data/metadata/               catálogos y manifests reproducibles
data/patches/sentinel_8ch/   GeoTIFF locales excluidos de Git
src/data/                    carga, filtros y validación de metadata
src/geo/                     preparación y visualización cartográfica
src/spectral/                almacenamiento, auditoría e inferencia
src/ui/                      identidad visual de la aplicación
tests/                       pruebas automatizadas
```

## Limitaciones actuales

- El MVP analiza parches previamente construidos; todavía no descarga nuevas
  imágenes Sentinel-2 desde la interfaz.
- No permite definir un área arbitraria ni ejecutar automáticamente toda la
  cadena de adquisición y preprocesamiento.
- La vista RGB es complementaria y no representa los ocho canales procesados
  por el modelo.
- El despliegue público requiere definir almacenamiento externo o una selección
  controlada de GeoTIFF, ya que los datos y el checkpoint no se versionan en Git.

Una evolución posterior podrá incorporar el flujo completo de definición de
área, selección y descarga de imágenes, preprocesamiento, cálculo de índices,
análisis, generación de mapas y validación.
