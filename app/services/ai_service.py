import base64
from typing import Any

import requests

from app import constants
from app.models import GenerateAIRequest


def build_ai_prompt(request: GenerateAIRequest) -> str:
    return (
        "Rewrite the user's message for workplace communication.\n"
        f"Mode: {request.mode}\n"
        f"Tone: {request.tone}\n"
        f"Platform: {request.platform}\n"
        f"Language: {request.language}\n"
        f"Length: {request.options.length}\n"
        f"Style: {request.options.style}\n\n"
        "Return only the rewritten message. Do not add explanations, quotes, "
        "labels, markdown, or alternatives.\n\n"
        f"User message: {request.input.text}"
    )


def call_openrouter(prompt: str) -> str:
    if not constants.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    response = requests.post(
        constants.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {constants.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": constants.OPENROUTER_REFERER,
            "X-Title": constants.OPENROUTER_APP_TITLE,
        },
        json={
            "model": constants.OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that improves workplace messages. "
                        "Be concise, natural, and context-aware."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 300,
        },
        timeout=constants.AI_TIMEOUT_SECONDS,
    )

    if response.status_code == 429:
        raise PermissionError("RATE_LIMIT")

    response.raise_for_status()
    data: dict[str, Any] = response.json()

    return data["choices"][0]["message"]["content"].strip()


def transcribe_image(image_bytes: bytes, content_type: str) -> str:
    if not constants.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{base64_image}"

    response = requests.post(
        constants.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {constants.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": constants.OPENROUTER_REFERER,
            "X-Title": constants.OPENROUTER_APP_TITLE,
        },
        json={
            "model": constants.OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable text from this image. "
                                "Return only the transcript."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1000,
        },
        timeout=constants.AI_TIMEOUT_SECONDS,
    )

    if response.status_code == 429:
        raise PermissionError("RATE_LIMIT")

    response.raise_for_status()
    data: dict[str, Any] = response.json()

    return data["choices"][0]["message"]["content"].strip()
