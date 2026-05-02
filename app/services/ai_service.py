import boto3

from app import constants
from app.models import GenerateAIRequest


BEDROCK_IMAGE_FORMATS = {
    "image/gif": "gif",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


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


def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=constants.AWS_REGION)


def _extract_bedrock_text(response: dict) -> str:
    content = response["output"]["message"]["content"]
    return "".join(block.get("text", "") for block in content).strip()


def call_bedrock(prompt: str) -> str:
    response = _bedrock_client().converse(
        modelId=constants.BEDROCK_MODEL,
        system=[
            {
                "text": (
                    "You are an assistant that improves workplace messages. "
                    "Be concise, natural, and context-aware."
                )
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.4,
        },
    )

    return _extract_bedrock_text(response)


def transcribe_image(image_bytes: bytes, content_type: str) -> str:
    image_format = BEDROCK_IMAGE_FORMATS.get(content_type)
    if not image_format:
        raise ValueError("Unsupported image type")

    response = _bedrock_client().converse(
        modelId=constants.BEDROCK_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Extract all readable text from this image. "
                            "Return only the transcript."
                        )
                    },
                    {
                        "image": {
                            "format": image_format,
                            "source": {
                                "bytes": image_bytes,
                            },
                        },
                    },
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 1000,
            "temperature": 0,
        },
    )

    return _extract_bedrock_text(response)
