from pydantic import BaseModel, Field


class AIInput(BaseModel):
    text: str | None = None
    image: str | None = None


class AIOptions(BaseModel):
    length: str = "auto"
    style: str = "auto"


class GenerateAIRequest(BaseModel):
    input: AIInput
    mode: str
    tone: str = "balanced"
    platform: str = "slack"
    language: str = "en"
    options: AIOptions = Field(default_factory=AIOptions)
