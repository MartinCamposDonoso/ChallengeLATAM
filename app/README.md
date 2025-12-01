# YOLO Object Detection API

FastAPI service for object detection using YOLOv8 model trained on custom dataset.

## Features

- **POST /predict**: Upload an image and receive annotated image with bounding boxes
- **POST /predict/json**: Upload an image and receive JSON with detection details
- **GET /health**: Health check endpoint
- **GET /**: Root endpoint with service status

## Local Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Docker Usage

### Build the image

```bash
docker build -t yolo-detection-api .
```

### Run the container

```bash
docker run -p 8080:8080 yolo-detection-api
```

## Usage Examples

### Using cURL - Get annotated image

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  --output annotated_result.jpg
```

### Using cURL - Get JSON detections

```bash
curl -X POST "http://localhost:8080/predict/json" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8080/predict"
image_path = "path/to/your/image.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open("annotated_result.jpg", "wb") as out:
        out.write(response.content)
    print("Image saved successfully!")
```

### Using Python - JSON response

```python
import requests

url = "http://localhost:8080/predict/json"
image_path = "path/to/your/image.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"Found {data['count']} objects:")
    for detection in data['detections']:
        print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")
```

## GCP Cloud Run Deployment

The service is configured to deploy to Google Cloud Run via GitHub Actions.

Required secrets in GitHub:
- `GCP_SA_KEY`: Service account key JSON
- `GCP_PROJECT`: GCP project ID
- `GCP_REGION`: Deployment region (e.g., us-central1)
- `CLOUD_RUN_SERVICE`: Service name

### Deploy manually

```bash
# Build and push to GCR
docker build -t gcr.io/[PROJECT-ID]/yolo-detection-api .
docker push gcr.io/[PROJECT-ID]/yolo-detection-api

# Deploy to Cloud Run
gcloud run deploy yolo-detection-api \
  --image gcr.io/[PROJECT-ID]/yolo-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

## Model

The service uses the trained YOLOv8 model located at:
`challenge/artifacts/model/model_best.pt`

### Detection Parameters

- **Confidence threshold**: 0.25
- **IOU threshold**: 0.45

These can be adjusted in [app/main.py](app/main.py) in the `predict` and `predict/json` endpoints.

## API Response Examples

### JSON Response Example

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

## Troubleshooting

### Model not loading

Ensure the model file exists at the correct path:
```bash
ls challenge/artifacts/model/model_best.pt
```

### Port already in use

Change the port:
```bash
uvicorn app.main:app --port 8081
```

### Memory issues

Reduce batch size or image resolution in the inference code.
