import time

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError

from app.constants import (
    ALLOWED_IMAGE_TYPES,
    API_PREFIX,
    APP_TITLE,
    MAX_IMAGE_SIZE_BYTES,
    MODES,
    OPENROUTER_MODEL,
    PLATFORMS,
    TONES,
    error_response,
)
from app.models import GenerateAIRequest
from app.services.ai_service import build_ai_prompt, call_openrouter, transcribe_image


app = FastAPI(title=APP_TITLE)


@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    return error_response(400, "INVALID_INPUT", "Invalid request payload")


@app.get(f"{API_PREFIX}/health")
def health_check():
    return {"status": "ok"}


@app.get(f"{API_PREFIX}/ping")
def ping():
    return {"message": "pong"}


@app.get(f"{API_PREFIX}/config")
def get_config():
    return {
        "tones": TONES,
        "platforms": PLATFORMS,
        "modes": MODES,
    }


@app.post(f"{API_PREFIX}/ai/generate")
def generate_ai_response(request: GenerateAIRequest):
    started_at = time.perf_counter()

    if not request.input.text or not request.input.text.strip():
        return error_response(400, "INVALID_INPUT", "Input text is required")

    try:
        output = call_openrouter(build_ai_prompt(request))
    except requests.exceptions.Timeout:
        return error_response(504, "AI_TIMEOUT", "AI request timed out")
    except PermissionError:
        return error_response(429, "RATE_LIMIT", "Rate limit exceeded")
    except (
        requests.exceptions.RequestException,
        KeyError,
        IndexError,
        RuntimeError,
        ValueError,
    ):
        return error_response(502, "AI_ERROR", "AI service failed")

    processing_time_ms = round((time.perf_counter() - started_at) * 1000)

    return {
        "success": True,
        "data": {
            "output": output,
            "mode": request.mode,
            "tone": request.tone,
            "platform": request.platform,
        },
        "meta": {
            "processingTimeMs": processing_time_ms,
            "model": OPENROUTER_MODEL,
        },
    }


@app.post(f"{API_PREFIX}/image/transcribe")
def transcribe_uploaded_image(file: UploadFile = File(...)):
    started_at = time.perf_counter()

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return error_response(400, "INVALID_INPUT", "Unsupported image type")

    image_bytes = file.file.read()
    if not image_bytes:
        return error_response(400, "INVALID_INPUT", "Image file is required")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return error_response(400, "INVALID_INPUT", "Image file is too large")

    try:
        transcript = transcribe_image(image_bytes, file.content_type)
    except requests.exceptions.Timeout:
        return error_response(504, "AI_TIMEOUT", "AI request timed out")
    except PermissionError:
        return error_response(429, "RATE_LIMIT", "Rate limit exceeded")
    except (
        requests.exceptions.RequestException,
        KeyError,
        IndexError,
        RuntimeError,
        ValueError,
    ):
        return error_response(502, "AI_ERROR", "AI service failed")

    processing_time_ms = round((time.perf_counter() - started_at) * 1000)

    return {
        "success": True,
        "data": {
            "transcript": transcript,
        },
        "meta": {
            "processingTimeMs": processing_time_ms,
            "model": OPENROUTER_MODEL,
        },
    }
