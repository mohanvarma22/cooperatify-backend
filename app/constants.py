import os

from dotenv import load_dotenv
from fastapi.responses import JSONResponse

load_dotenv()

APP_TITLE = "Cooperatify Backend"
API_PREFIX = "/api/v1"

AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "amazon.nova-lite-v1:0")
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"]

TONES = ["gentle", "balanced", "spicy"]
PLATFORMS = ["slack", "email", "linkedin"]
MODES = ["translate", "reply"]


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
