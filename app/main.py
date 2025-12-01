from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

app = FastAPI(
    title="YOLO Object Detection API",
    description="API for object detection using YOLOv8 model",
    version="1.0.0"
)

# Load the model
MODEL_PATH = Path(__file__).parent.parent / "challenge" / "artifacts" / "model" / "model_best.pt"
model = None

@app.on_event("startup")
async def load_model():
    """Load the YOLO model on startup"""
    global model
    try:
        model = YOLO(str(MODEL_PATH))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "YOLO Object Detection API is running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform object detection on uploaded image

    Args:
        file: Image file (jpg, jpeg, png)

    Returns:
        Image with bounding boxes and annotations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert PIL Image to numpy array (RGB)
        image_np = np.array(image)

        # Run inference
        results = model.predict(image_np, conf=0.25, iou=0.45)

        # Get annotated image
        annotated_image = results[0].plot()

        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        output_image = Image.fromarray(annotated_image_rgb)

        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=annotated_{file.filename}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/json")
async def predict_json(file: UploadFile = File(...)):
    """
    Perform object detection and return JSON with detections

    Args:
        file: Image file (jpg, jpeg, png)

    Returns:
        JSON with detection results including bounding boxes, classes, and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Run inference
        results = model.predict(image_np, conf=0.25, iou=0.45)

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = {
                    "class_id": int(boxes.cls[i]),
                    "class_name": result.names[int(boxes.cls[i])],
                    "confidence": float(boxes.conf[i]),
                    "bbox": {
                        "x1": float(boxes.xyxy[i][0]),
                        "y1": float(boxes.xyxy[i][1]),
                        "x2": float(boxes.xyxy[i][2]),
                        "y2": float(boxes.xyxy[i][3])
                    }
                }
                detections.append(detection)

        return {
            "detections": detections,
            "count": len(detections),
            "image_shape": {
                "width": image_np.shape[1],
                "height": image_np.shape[0]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
