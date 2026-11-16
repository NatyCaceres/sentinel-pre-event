# Preparación para Streamlit Community Cloud

Estado: comprobación previa y paquete local verificados; aplicación todavía no publicada.
Esta preparación conserva el modelo, los datos, el frontend y el entorno local.

## Entorno de despliegue

- Entrada en Community Cloud: `app/cloud.py`, ejecutada desde la raíz del repositorio.
  Prepara los archivos y ejecuta `app/app.py`. El arranque local habitual sigue
  siendo `python -m streamlit run app/app.py`.
- Seleccionar **Python 3.13** en Advanced settings de Community Cloud.
- Destino: Linux x86_64 con glibc >= 2.28.
- Dependencias: `app/requirements.txt`. Al estar junto a la entrada,
  Community Cloud lo selecciona antes que el archivo de la raíz.
- El archivo de la raíz sigue siendo el entorno local y de pruebas.
  No instalar `app/requirements.txt` en Windows: contiene wheels específicos
  de Linux/CPython 3.13.
- PyTorch y torchvision apuntan a wheels oficiales de CPU con SHA-256.
  El paquete genérico de PyTorch declara componentes CUDA para Linux y no es
  apropiado para este despliegue CPU.
- Se fijan también las dependencias transitivas de ejecución. No se incluyen
  las herramientas geográficas usadas solo para construir los datos.

## Repetir la comprobación

Desde el entorno existente y la raíz del proyecto:

```powershell
python scripts/check_cloud_readiness.py --packages --runtime
python -m pytest -q
```

`--packages` consulta metadatos oficiales de PyPI y PyTorch, verifica versiones,
wheels para Linux/CPython 3.13 y restricciones de dependencias, incluidos extras.
Necesita Internet; no instala paquetes ni descarga los wheels.

`--runtime` verifica el catálogo, la disponibilidad de los TIFF del test y el
checkpoint mediante el cargador existente. Ejecuta inferencia y Grad-CAM en
tres casos, incluidos parches con NaN, comprueba que las probabilidades
coincidan y mide memoria residente máxima del proceso. No escribe artefactos.
Puede ejecutarse de forma independiente:

```powershell
python scripts/check_cloud_readiness.py --runtime
```

La medición del 16-11-2026 en Windows encontró los 244 TIFF (31,05 MiB), verificó
el checkpoint (42,78 MiB) y alcanzó unos 494 MiB de memoria residente máxima.
Las tres inferencias y explicaciones coincidieron. Es una referencia de un
proceso local con uso secuencial, no una garantía de memoria, concurrencia o
latencia del servidor. La consulta de metadatos aprobó las 60 dependencias.

## Paquete del modelo y las imágenes

Los TIFF y el checkpoint siguen excluidos de Git. Community Cloud clona el
repositorio: no puede acceder a los archivos del computador local. Por eso,
el arranque cloud requiere una URL de descarga directa para preparar los archivos.

El paquete ya se prepara con:

```powershell
python scripts/build_cloud_bundle.py
```

Produce un único archivo `.cloud/sentinel-test-resnet18-v1.zip` (73,87 MiB),
excluido de Git, y `config/cloud_bundle.json`, que sí debe versionarse.
Contiene los 244 TIFF del test y el checkpoint. Conserva los nombres, valida
físicamente los TIFF con la auditoría existente y compara los hashes con los
artefactos del proyecto. No incluye train, validation ni credenciales.

La fecha, el orden y los permisos internos del ZIP son fijos. Se almacena sin
compresión para que su reproducción no dependa de zlib. Dos ejecuciones con
los mismos archivos produjeron el mismo SHA-256:

```text
e0b0c3f7e1f3dfd120aff7dd80a5b02fd3d33f2c27198f51cd70787519e03ed7
```

El arranque verifica catálogo y contratos contra el manifest. Si los archivos
ya están presentes y son correctos, no descarga nada. En caso contrario,
descarga por HTTPS con un límite de tamaño, comprueba el SHA-256 del ZIP y de
cada archivo, y rechaza archivos faltantes, extras, duplicados y enlaces.
Solo instala los archivos verificados en las rutas actuales del MVP.
Los temporales se eliminan y la preparación se guarda en caché por proceso.
Al reiniciar el servidor puede ser necesario descargar nuevamente el paquete.

El catálogo original de 1253 registros y su manifest deben conservarse:
la aplicación verifica su integridad y selecciona los 244 casos de test.
`.gitattributes` fija los saltos de línea de los cuatro archivos fuente
verificados, conservando sus bytes auditados tanto en Windows como en Linux.
Una prueba exporta esos archivos desde el índice de Git sin `autocrlf` y
comprueba sus hashes; así se detectan cambios de formato que impedirían arrancar.
La auditoría completa del almacenamiento sigue requiriendo los 1253 TIFF
locales; no debe reinterpretarse como auditoría de un paquete de despliegue.

La visibilidad de la aplicación y la de los archivos son decisiones distintas.
Antes de subir un paquete a un alojamiento público se debe confirmar que se
pueden compartir tanto las imágenes como los pesos. Este paso no ha publicado
archivos ni creado releases, cuentas o servicios externos.

## Pasos externos, una vez aprobado el alojamiento de archivos

1. Subir los cambios de código, requisitos, manifest y documentación a GitHub.
   El ZIP, los TIFF y el checkpoint no se agregan a los commits.
2. Si se aprueba su descarga pública, adjuntar el ZIP a una **GitHub Release**
   del repositorio público. Releases almacena archivos adjuntos fuera del
   historial de código. Se puede preparar como borrador y publicar después.
   Una release de un repositorio privado no proporciona una descarga anónima:
   requeriría otro mecanismo de acceso. El arranque actual acepta una URL HTTPS
   directa pública o firmada; no implementa autenticación de GitHub privado.
3. Copiar el enlace de descarga del ZIP adjunto, no el enlace a la página de la
   release ni el ZIP automático del código fuente.
4. Entrar en https://share.streamlit.io y conectar la cuenta de GitHub.
5. Crear la aplicación con repositorio `NatyCaceres/sentinel-pre-event`, rama
   con los cambios publicados y entrada `app/cloud.py`.
6. En Advanced settings seleccionar Python 3.13 y configurar este secret,
   reemplazando el ejemplo por la URL real:

   ```toml
   CLOUD_BUNDLE_URL = "https://alojamiento.example/paquete.zip"
   ```

   No guardar una URL firmada ni credenciales en el repositorio. El SHA-256
   esperado se toma del manifest versionado, nunca del servidor de descarga.
7. Desplegar, revisar logs y completar las pruebas siguientes antes de compartir
   el enlace con la comisión. Una URL firmada debe seguir vigente cuando el
   servidor necesite volver a descargar los archivos.

Las credenciales se introducen personalmente en GitHub/Streamlit; no es
necesario compartir contraseñas en el chat.

## Validación necesaria antes de entregar el enlace

1. Instalación real del perfil Linux, revisando sus logs.
2. Descarga e integridad de los 244 TIFF y el checkpoint.
3. Ejecutar `python -m src.spectral.evaluation`: resultado `COINCIDE`.
4. Probar mapa, filtros, búsqueda del caso sin ubicación, inferencia y Grad-CAM.
5. Probar varias sesiones y el arranque tras reiniciar el servidor.
6. Comprobar el acceso desde otro computador con la modalidad elegida.

Conservar la versión local para la defensa. Community Cloud limita los recursos
y suspende aplicaciones sin tráfico; la primera visita puede necesitar esperar.

## Documentación oficial

- [Dependencias y prioridad de archivos](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Organización de archivos](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Despliegue y selección de Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Recursos y suspensión](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
