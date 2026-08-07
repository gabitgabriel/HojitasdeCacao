import base64
import io
import os
from typing import Annotated
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(
    title="LeafScan MDD - API de Inferencia Fitosanitaria",
    description="Servidor Backend de alto rendimiento para Clasificación y Segmentación de Hojas de Cacao",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de los modelos entrenados (Soporta pesos nativos PyTorch .pt o .onnx)
PT_SEGMENTER = os.path.join("runs", "cacao_seg_run", "weights", "best.pt")
ONNX_SEGMENTER = os.path.join("frontend", "assets", "model", "leaf_segmenter_yolo26.onnx")

SEGMENTER_MODEL_PATH = PT_SEGMENTER if os.path.exists(PT_SEGMENTER) else ONNX_SEGMENTER
CLASSIFIER_MODEL_PATH = os.path.join("frontend", "assets", "model", "leaf_classifier_yolo26.onnx")
LABELS_PATH = os.path.join("frontend", "assets", "model", "labels.txt")

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = [line.strip() for line in f if line.strip()]
else:
    LABELS = ["DISEASED_VSD", "Fosforo", "Fosforo Potasio", "Nitrogeno", "Potasio", "Sana"]

classifier_model = None
segmenter_model = None


def _classify_image(temp_filename: str):
    cls_results = classifier_model(temp_filename)
    probabilities = []
    
    if cls_results and len(cls_results) > 0:
        probs = cls_results[0].probs
        if probs is not None:
            scores = probs.data.cpu().numpy()
            for idx, score in enumerate(scores):
                label_name = LABELS[idx] if idx < len(LABELS) else f"Clase_{idx}"
                probabilities.append({
                    "label": label_name,
                    "confidence": float(score)
                })
            probabilities.sort(key=lambda x: x["confidence"], reverse=True)
            
    return probabilities


def _segment_image(temp_filename: str, target_size: tuple[int, int]) -> str:
    seg_results = segmenter_model(temp_filename)
    if not seg_results or len(seg_results) == 0:
        return ""

    res = seg_results[0]
    if res.masks is None or len(res.masks) == 0:
        return ""

    # Combinar todas las máscaras de hojas detectadas (Multi-Leaf Detection)
    combined_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    for mask in res.masks.data:
        mask_np = mask.cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(target_size, Image.Resampling.BILINEAR)
        combined_mask = np.maximum(combined_mask, np.array(mask_pil))

    mask_combined_pil = Image.fromarray(combined_mask)
    mask_bytes_io = io.BytesIO()
    mask_combined_pil.save(mask_bytes_io, format="PNG")
    return base64.b64encode(mask_bytes_io.getvalue()).decode("utf-8")


def _load_models():
    global classifier_model, segmenter_model, SEGMENTER_MODEL_PATH
    PT_SEGMENTER = os.path.join("runs", "cacao_seg_run", "weights", "best.pt")
    ONNX_SEGMENTER = os.path.join("frontend", "assets", "model", "leaf_segmenter_yolo26.onnx")
    SEGMENTER_MODEL_PATH = PT_SEGMENTER if os.path.exists(PT_SEGMENTER) else ONNX_SEGMENTER

    print(f"[BACKEND] Cargando modelo Clasificador desde {CLASSIFIER_MODEL_PATH}...")
    classifier_model = YOLO(CLASSIFIER_MODEL_PATH, task="classify")
    print(f"[BACKEND] Cargando modelo Segmentador desde {SEGMENTER_MODEL_PATH}...")
    segmenter_model = YOLO(SEGMENTER_MODEL_PATH, task="segment")
    print("[BACKEND] Ambos modelos cargados exitosamente en la memoria del Servidor.")

_load_models()


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "LeafScan MDD Backend API",
        "models_ready": classifier_model is not None and segmenter_model is not None,
        "segmenter_path": SEGMENTER_MODEL_PATH
    }


@app.post("/reload-models")
def reload_models():
    try:
        _load_models()
        return {
            "status": "success",
            "message": "Modelos recargados exitosamente tras el entrenamiento",
            "segmenter_path": SEGMENTER_MODEL_PATH
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al recargar modelos: {str(e)}"
        )


@app.post(
    "/analyze",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "El archivo proporcionado no es una imagen válida"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error interno durante la inferencia fitosanitaria"}
    }
)
async def analyze_leaf(
    file: Annotated[UploadFile, File(description="Foto de la hoja a analizar")],
    crop_square: bool = True
):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo proporcionado debe ser una imagen válida (.jpg, .png, .webp)"
        )

    try:
        if crop_square:
            # Recorte a proporción cuadrada 1:1 centrada opcional
            w, h = pil_image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            pil_image = pil_image.crop((left, top, left + min_dim, top + min_dim))

        original_size = pil_image.size
        temp_filename = "temp_input.jpg"
        pil_image.save(temp_filename)

        probabilities = _classify_image(temp_filename)
        top_label = probabilities[0]["label"] if probabilities else "Sana"
        mask_base64 = _segment_image(temp_filename, original_size)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        return {
            "top_label": top_label,
            "top_confidence": probabilities[0]["confidence"] if probabilities else 0.0,
            "diagnoses": probabilities,
            "mask_png_base64": mask_base64
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento fitosanitario: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
