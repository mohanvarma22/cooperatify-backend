import os
from functools import lru_cache
import json

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

load_dotenv()

APP_TITLE = "Cooperatify Backend"
API_PREFIX = "/api/v1"

OPENROUTER_API_KEY_SECRET_NAME = os.getenv(
    "OPENROUTER_API_KEY_SECRET_NAME",
    "OPENROUTER_API_KEY",
)
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost:8000")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", APP_TITLE)
AI_TIMEOUT_SECONDS = 30
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"]

TONES = ["gentle", "balanced", "spicy"]
PLATFORMS = ["slack", "email", "linkedin"]
MODES = ["translate", "reply"]


@lru_cache(maxsize=1)
def get_openrouter_api_key() -> str | None:
    local_api_key = os.getenv("OPENROUTER_API_KEY")
    if local_api_key:
        return local_api_key

    try:
        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        response = client.get_secret_value(SecretId=OPENROUTER_API_KEY_SECRET_NAME)
    except (BotoCoreError, ClientError):
        return None

    secret = response.get("SecretString")
    if not secret:
        return None

    try:
        parsed = json.loads(secret)
    except json.JSONDecodeError:
        return secret

    if isinstance(parsed, dict):
        return parsed.get("OPENROUTER_API_KEY") or parsed.get(
            OPENROUTER_API_KEY_SECRET_NAME
        )

    return secret


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )
