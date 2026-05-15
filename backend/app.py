from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Callable
from zipfile import BadZipFile, ZipFile

import numpy as np
import keras
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from keras.applications import (
    densenet,
    efficientnet,
    inception_v3,
    mobilenet_v2,
    resnet50,
)


BASE_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 10 * 1024 * 1024


# 0 = non recyclable, 1 = organic, 2 = recyclable.
CLASS_LABELS = ["non recyclable", "organic", "recyclable"]


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    image_size: tuple[int, int]
    preprocess: Callable[[np.ndarray], np.ndarray]


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "inceptionv3": ModelConfig(
        path=BASE_DIR / "models" / "inceptionV3_split.keras",
        image_size=(299, 299),
        preprocess=inception_v3.preprocess_input,
    ),
    "mobilenetv2": ModelConfig(
        path=BASE_DIR / "models" / "waste_mobilenetv2_final.keras",
        image_size=(224, 224),
        preprocess=mobilenet_v2.preprocess_input,
    ),
    "resnet": ModelConfig(
        path=BASE_DIR / "models" / "waste_resnet50_final.keras",
        image_size=(224, 224),
        preprocess=resnet50.preprocess_input,
    ),
    "efficientnet": ModelConfig(
        path=BASE_DIR / "models" / "waste_efficientnetb0_final(split).keras",
        image_size=(224, 224),
        preprocess=efficientnet.preprocess_input,
    ),
    "densenet121": ModelConfig(
        path=BASE_DIR / "models" / "densenet_split.keras",
        image_size=(224, 224),
        preprocess=densenet.preprocess_input,
    ),
}


models: dict[str, keras.Model] = {}
model_locks: dict[str, Lock] = {model_name: Lock() for model_name in MODEL_CONFIGS}


def load_all_models() -> None:
    for model_name, config in MODEL_CONFIGS.items():
        if not config.path.exists():
            raise RuntimeError(f"Model file for '{model_name}' not found: {config.path}")
        models[model_name] = load_model(config.path)


def load_model(path: Path) -> keras.Model:
    try:
        return keras.saving.load_model(path, compile=False)
    except KeyError as exc:
        if "config.json" not in str(exc):
            raise
        return load_nested_keras_model(path)


def load_nested_keras_model(path: Path) -> keras.Model:
    try:
        with ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) != 1:
                raise RuntimeError(f"Cannot load nested Keras model from {path}.")

            with TemporaryDirectory() as temp_dir:
                extracted_path = Path(temp_dir) / entries[0].filename
                archive.extract(entries[0], temp_dir)
                return keras.saving.load_model(extracted_path, compile=False)
    except BadZipFile as exc:
        raise RuntimeError(f"Invalid Keras archive: {path}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    models.clear()


app = FastAPI(
    title="Waste Classification API",
    description="Predict recyclable, organic, or non recyclable waste from an uploaded image.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Missing model name.",
                "allowed_models": sorted(MODEL_CONFIGS),
            },
        )

    normalized = model_name.strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": f"Invalid model name: {model_name}",
                "allowed_models": sorted(MODEL_CONFIGS),
            },
        )
    return normalized


async def read_image_bytes(image: UploadFile) -> bytes:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image is too large. Maximum size is 10 MB.",
        )
    return image_bytes


def preprocess_image(image_bytes: bytes, model_name: str) -> np.ndarray:
    config = MODEL_CONFIGS[model_name]

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize(config.image_size, Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from exc

    batch = np.expand_dims(image_array, axis=0)
    return config.preprocess(batch)


def build_predictions(raw_predictions: np.ndarray) -> list[dict[str, float | str]]:
    scores = np.asarray(raw_predictions[0], dtype=np.float32)

    if scores.shape[0] != len(CLASS_LABELS):
        raise RuntimeError(
            f"Model returned {scores.shape[0]} classes, but {len(CLASS_LABELS)} labels are configured."
        )

    # If the model already ends with softmax this is effectively a no-op;
    # otherwise this converts logits into probabilities for the API response.
    if not np.isclose(float(np.sum(scores)), 1.0, atol=1e-3):
        scores = tf.nn.softmax(scores).numpy()

    predictions = [
        {
            "class_index": index,
            "label": label,
            "confidence": float(scores[index]),
        }
        for index, label in enumerate(CLASS_LABELS)
    ]
    return sorted(predictions, key=lambda item: item["confidence"], reverse=True)


@app.get("/")
def root() -> dict[str, str | list[str]]:
    return {
        "message": "Waste Classification API is running.",
        "models": sorted(MODEL_CONFIGS),
        "classes": CLASS_LABELS,
    }


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "models_loaded": len(models)}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: str = Form(...),
) -> dict[str, str | float | list[dict[str, float | str]]]:
    model_name = normalize_model_name(model)
    image_bytes = await read_image_bytes(image)
    processed_image = preprocess_image(image_bytes, model_name)

    selected_model = models.get(model_name)
    if selected_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not loaded.",
        )

    try:
        with model_locks[model_name]:
            raw_predictions = selected_model.predict(processed_image, verbose=0)
        predictions = build_predictions(raw_predictions)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    top_prediction = predictions[0]
    return {
        "model": model_name,
        "predicted_class": str(top_prediction["label"]),
        "confidence": float(top_prediction["confidence"]),
        "predictions": predictions,
    }
