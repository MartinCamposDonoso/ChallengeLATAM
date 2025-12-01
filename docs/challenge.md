# Desafío: LATAM Data Scientist
**Desarrollador:** Martín Alfonso Campos Donoso
**Fecha:** 01/12/2025

---

## 1. Integridad de los Datos (Data Leakage)

Se ha realizado una validación cruzada entre los conjuntos de datos para asegurar la fiabilidad de las métricas futuras.

* **Hallazgo:** Se detectaron **10 instancias** de fuga de información (imágenes duplicadas o muy similares) entre los sets de entrenamiento y validación/prueba.
* **Acción:** Estas imágenes se han aislado y detallan en `reporte_fugas/resumen_fugas.csv`.
* **Impacto:** Es crítico removerlas para evitar un sobreajuste ficticio en las métricas de evaluación.

<div align="center">
  <img src="imagenes/fuga_informacion.png" alt="Fuga de información" width="600">
  <p><em>Figura 1: Ejemplos de imágenes duplicadas encontradas entre splits.</em></p>
</div>

---

## 2. Análisis de Distribución y Balance de Clases

Se analizó la cardinalidad y distribución de las etiquetas para detectar desbalances que puedan afectar el rendimiento del modelo.

### Distribución por Conjunto de Datos
Se observa una desproporción en el volumen de datos. **Recomendación:** Aumentar el tamaño de los conjuntos de validación y prueba para garantizar significancia estadística.

<div align="center">
  <img src="imagenes/distribucion_etiquetas.png" alt="Distribución etiquetas" width="600">
  <p><em>Figura 2: Distribución de la cantidad de etiquetas (Train/Val/Test).</em></p>
</div>

### Desbalance de Clases
Existe un **desbalance severo**:
* **Clases Dominantes:** `forklift` y `person`.
* **Clases Minoritarias:** Resto de las clases.
* **Tamaño de anotaciones:** Varía significativamente, lo cual es típico en detección de objetos pero requiere atención en los anchors del modelo.

<div align="center">
  <img src="imagenes/concentracion_clases.png" alt="Concentración clases" width="45%">
  <img src="imagenes/tamano_anotaciones.png" alt="Tamaño anotaciones" width="45%">
  <p><em>Figura 3 y 4: Concentración de clases y distribución de tamaños de bounding boxes.</em></p>
</div>

---

## 3. Calidad del Etiquetado y Aumentación

La inspección cualitativa revela problemas de calidad que introducen ruido en el entrenamiento (Label Noise).

**Problemas detectados:**
1.  **Corrupción por Aumentación:** Algunas transformaciones parecen haber degradado la imagen o las etiquetas.
2.  **Falsos Negativos (Missing Labels):** Objetos presentes (especialmente personas y contenedores) que no fueron etiquetados.

<div align="center">
  <img src="imagenes/vis_etiquetas_personas.png" alt="Etiquetas Personas" width="600">
  <p><em>Figura 5: Visualización de ground truth en la clase Personas.</em></p>
</div>

A continuación, se evidencian casos críticos de falta de etiquetado:

<div align="center">
  <img src="imagenes/fallas_etiquetado.png" width="45%">
  <img src="imagenes/falta_etiquetado_container.png" width="45%">
  <p><em>Figura 6: Ejemplos de objetos no etiquetados (Personas y Freight Containers).</em></p>
</div>

### Visualizacion de las etiquetas de personas y forklift
<div align="center">
  <img src="imagenes/vis_forklift_person.png" width="600">
  <p><em>Figura 7: Muestreo de etiquetas para Forklift y Person.</em></p>
</div>

---

## 4. Consistencia Visual y Espacio Latente

Se realizó una proyección del espacio de características (Feature Space) para validar la separabilidad de las clases.

**Análisis del Video:**
El análisis dinámico muestra que las clases **Forklift** y **Person** poseen características visuales bien definidas, agrupándose correctamente en el espacio latente. Esto sugiere que, a pesar del ruido en las etiquetas, el modelo debería ser capaz de converger en estas clases principales.

<div align="center">
  <video width="80%" controls>
    <source src="videos/clusterizacion.mp4" type="video/mp4">
    Tu navegador no soporta la etiqueta de video.
  </video>
  <p><em>Video 1: Visualización dinámica de la clusterización de etiquetas.</em></p>
</div>

<div align="center">
  <img src="imagenes/clusterizacion.png" width="600">
  <p><em>Figura 8: Clusterización estática basada en características visuales.</em></p>
</div>

---

## 5. Conclusiones y Próximos Pasos

### Resumen del Dataset
* ✅ **Integridad:** Todas las etiquetas tienen imágenes asociadas.
* ✅ **Consistencia:** Los 3 conjuntos (Train/Val/Test) comparten el mismo esquema de clases.
* ℹ️ **Background:** No existen imágenes de fondo (sin etiquetas), lo cual puede aumentar los Falsos Positivos en producción.

### Estrategia Sugerida
1.  **Limpieza de Fugas:** Eliminar inmediatamente las 10 imágenes del reporte de fugas.
2.  **Data Curation:** Priorizar una revisión manual o asistida por modelo para corregir los *missing labels* en `Person` y `Freight Container`.
3.  **Estrategia de Sampling:** Dado el desbalance, utilizar técnicas de *oversampling* para las clases minoritarias o funciones de pérdida ponderadas (Weighted Loss).

---

## 6. Entrenamiento del Modelo (Part II)

### Configuración del Modelo

Se utilizó **YOLOv8 nano (yolov8n.pt)** como modelo base para el entrenamiento, optimizando el balance entre velocidad de inferencia y precisión.

#### Hiperparámetros de Entrenamiento

```python
EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Proceso de Entrenamiento

El modelo fue entrenado utilizando el dataset procesado y validado en las secciones anteriores. El entrenamiento se realizó con las siguientes características:

- **Arquitectura:** YOLOv8 Nano
- **Dataset:** Imágenes de 640x640 píxeles
- **Épocas:** 50 iteraciones completas sobre el dataset
- **Batch Size:** 16 imágenes por lote
- **Optimizador:** AdamW (por defecto en YOLO)
- **Data Augmentation:** Augmentaciones nativas de YOLOv8 (mosaic, flip, scale, etc.)

### Resultados del Entrenamiento

El modelo entrenado se guardó en `challenge/artifacts/model/model_best.pt` y está listo para ser utilizado en inferencia.

#### Métricas de Evaluación

Las métricas de validación incluyen:
- **mAP50:** Precisión promedio con IoU threshold de 0.5
- **mAP50-95:** Precisión promedio con IoU thresholds de 0.5 a 0.95
- **Precisión y Recall** por clase

Los resultados detallados del entrenamiento se encuentran en el notebook `challenge/02_model_training.ipynb`.

---

## 7. Deployment con FastAPI en GCP (Part III & IV)

### Arquitectura del Servicio

Se implementó un servicio de inferencia completo utilizando **FastAPI** y desplegado en **Google Cloud Run** con un pipeline CI/CD automatizado.

### API Endpoints

#### 1. Health Check
```
GET /health
```
Retorna el estado del servicio y confirmación de carga del modelo.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_path": "challenge/artifacts/model/model_best.pt",
  "model_loaded": true
}
```

#### 2. Predicción con Imagen Anotada
```
POST /predict
```
Recibe una imagen y retorna la misma imagen con bounding boxes y etiquetas de los objetos detectados.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Archivo de imagen (JPG, JPEG, PNG)

**Respuesta:**
- Content-Type: `image/jpeg`
- Body: Imagen anotada con detecciones

**Ejemplo de uso:**
```bash
curl -X POST "https://[SERVICE-URL]/predict" \
  -F "file=@imagen.jpg" \
  --output resultado_anotado.jpg
```

#### 3. Predicción con JSON
```
POST /predict/json
```
Recibe una imagen y retorna un JSON con las detecciones detalladas.

**Respuesta:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": {
        "x1": 123.45,
        "y1": 67.89,
        "x2": 234.56,
        "y2": 345.67
      }
    }
  ],
  "count": 1,
  "image_shape": {
    "width": 640,
    "height": 480
  }
}
```

### Implementación Técnica

#### Estructura del Código

El servicio está implementado en `challenge/api.py` con las siguientes características:

- **Framework:** FastAPI 0.115.4
- **Modelo:** YOLOv8 cargado con Ultralytics
- **Procesamiento de Imágenes:** PIL, NumPy, OpenCV
- **Thresholds de Inferencia:**
  - Confidence: 0.25
  - IoU: 0.45

#### Dockerfile Multi-Stage

Se utilizó un Dockerfile multi-stage para optimizar el tamaño de la imagen:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY challenge/ ./challenge/
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Optimizaciones:**
- Imagen base: `python:3.11-slim`
- Build multi-stage para reducir tamaño
- Dependencias del sistema mínimas
- Health checks integrados

### CI/CD Pipeline

#### Continuous Integration (CI)

**Workflow:** `.github/workflows/ci.yml`

Ejecuta automáticamente en cada push:
1. Lint con flake8
2. Tests de importación de dependencias
3. Validación de sintaxis

#### Continuous Delivery (CD)

**Workflow:** `.github/workflows/cd.yml`

Pipeline automatizado que ejecuta en `develop` y `main`:

1. **Docker Build:**
   - Limpieza de espacio en disco (~30GB liberados)
   - Build de imagen multi-stage
   - Push a GitHub Container Registry (ghcr.io)

2. **GCP Deployment:**
   - Autenticación con Service Account
   - Deploy a Cloud Run
   - Configuración de servicio:
     - Puerto: 8080
     - CPU: 1
     - Memoria: 2GB
     - Acceso: Público (--allow-unauthenticated)

3. **Estrategia de Branches:**
   - `develop` → Despliega como `[service-name]-dev`
   - `main` → Despliega como `[service-name]` (producción)

### Configuración de GCP

#### Secrets Requeridos

Los siguientes secrets deben estar configurados en GitHub:

- `GCP_SA_KEY`: Service Account Key en formato JSON
- `GCP_PROJECT`: ID del proyecto en GCP
- `GCP_REGION`: Región de deployment (ej: `us-central1`)
- `CLOUD_RUN_SERVICE`: Nombre del servicio

#### URL del Servicio Desplegado

El servicio está disponible en:
```
https://[CLOUD_RUN_SERVICE]-dev-[hash].a.run.app
```

*(La URL exacta se muestra en los logs del workflow de CD después del deployment)*

### Pruebas del Servicio

#### Prueba Local

```bash
# Iniciar el servicio localmente
uvicorn challenge.api:app --reload --host 0.0.0.0 --port 8080

# Ejecutar tests
python test_api.py path/to/test/image.jpg
```

#### Prueba en Producción

```bash
# Health check
curl https://[SERVICE-URL]/health

# Predicción con imagen
curl -X POST "https://[SERVICE-URL]/predict" \
  -F "file=@imagen.jpg" \
  --output resultado.jpg

# Predicción con JSON
curl -X POST "https://[SERVICE-URL]/predict/json" \
  -F "file=@imagen.jpg"
```

### Monitoreo y Logs

Los logs del servicio están disponibles en:
- **GitHub Actions:** Para builds y deployments
- **GCP Cloud Run:** Para logs de aplicación y errores de runtime
- **GCP Metrics:** Para métricas de latencia, CPU y memoria

---

## 8. Conclusiones Finales

### Logros del Proyecto

1. ✅ **Análisis Exhaustivo del Dataset**
   - Detección de data leakage (10 imágenes)
   - Identificación de desbalance de clases
   - Validación de calidad de etiquetado

2. ✅ **Entrenamiento del Modelo**
   - YOLOv8 nano entrenado exitosamente
   - Modelo guardado en `model_best.pt`
   - Listo para inferencia en producción

3. ✅ **Servicio de Inferencia**
   - API completa con FastAPI
   - Endpoints para imagen anotada y JSON
   - Documentación automática con Swagger

4. ✅ **Deployment en la Nube**
   - Servicio desplegado en Google Cloud Run
   - Pipeline CI/CD completamente automatizado
   - Estrategia de staging (develop) y producción (main)

### Próximas Mejoras

1. **Modelo:**
   - Experimentar con YOLOv8 medium/large para mayor precisión
   - Implementar técnicas de manejo de desbalance de clases
   - Fine-tuning con corrección de missing labels

2. **API:**
   - Agregar autenticación y rate limiting
   - Implementar batch processing
   - Caché de predicciones frecuentes

3. **Infraestructura:**
   - Auto-scaling basado en carga
   - Monitoreo con Prometheus/Grafana
   - A/B testing entre versiones de modelo