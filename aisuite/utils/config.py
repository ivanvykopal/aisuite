from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Union, Optional, List

# I need to define the standard structure for the kwargs for the chat_completions_create, it should use OpenAI chat completion parameters as reference.


class AudioID(BaseModel):
    """Configuration class for unique audio voice identifiers."""

    id: str


class AudioConfig(BaseModel):
    """Configuration class for audio-specific parameters."""

    format: str = None  # wav, mp3, flac, opus, pcm16
    voice: Union[str, AudioID] = None


class PredictionConfig(BaseModel):
    content: Union[List[str], str] = None
    type: str = None  # e.g., "content"


class StreamOptions(BaseModel):
    include_obfuscation: Optional[bool] = False
    include_usage: Optional[bool] = False


class ApproximateLocation(BaseModel):
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None


class UserLocation(BaseModel):
    approximate: Optional[ApproximateLocation] = None
    type: Optional[str] = None  # e.g., "ip", "gps"


class WebSearchOptions(BaseModel):
    search_context_size: Optional[str] = None  # e.g., "low", "medium", "high"
    user_location: Optional[UserLocation] = None


class Config(BaseModel):
    """Configuration class for chat completion parameters."""

    model_config = ConfigDict(extra="forbid")

    """OpenAI Compatible"""
    audio: Optional[AudioConfig] = None
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    max_completion_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = {}
    modalities: Optional[List[str]] = None
    n: Optional[int] = 1
    parallel_tool_calls: Optional[bool] = False
    prediction: Optional[str] = None
    presence_penalty: Optional[float] = 0.0  # -2.0 to 2.0
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None  # e.g., "24h"
    reasoning_effort: Optional[str] = (
        None  # e.g., "none", "minimal", "low", "medium", "high", and "xhigh"
    )
    response_format: Optional[dict] = None
    safety_identifier: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    store: Optional[bool] = False
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 1.0
    tool_choice: Optional[str] = None
    tools: Optional[List[str]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = 1.0
    verbosity: Optional[str] = None  # e.g., "none", "low", "medium", "high"
    web_search_options: Optional[Dict[str, Any]] = None

    """VLLM Specific"""
    quantization: Optional[str] = None  # e.g., "bitsandbytes", "fp16"
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    best_of: Optional[int] = None
    min_p: Optional[float] = None
    seed: Optional[int] = None
    min_tokens: Optional[int] = None
    skip_special_tokens: Optional[bool] = True
