from __future__ import annotations

import base64
import inspect
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    APIConnectionError = None  # type: ignore
    APIError = None  # type: ignore
    APIStatusError = None  # type: ignore
    APITimeoutError = None  # type: ignore
    RateLimitError = None  # type: ignore

import torch

from .utils import configure_logging

LOGGER = configure_logging("snapcheck.vlm")


@dataclass
class VLMResponse:
    text: str
    raw: Any
    latency: Optional[float] = None
    token_usage: Optional[int] = None


class VLMBackend:
    def generate(self, image: Image.Image, prompt: str) -> VLMResponse:  # pragma: no cover - interface
        raise NotImplementedError



class HuggingFaceVLM(VLMBackend):
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        max_new_tokens: int = 64,
        temperature: Optional[float] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        do_sample: Optional[bool] = None,
    ) -> None:
        if AutoModelForVision2Seq is None:
            raise ImportError("transformers is required for HuggingFaceVLM")

        if device == "cuda_if_available":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)

        processor_kwargs = dict(processor_kwargs or {})
        if revision:
            processor_kwargs.setdefault("revision", revision)
        processor_kwargs.setdefault("trust_remote_code", trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

        model_kwargs = dict(model_kwargs or {})
        if revision:
            model_kwargs.setdefault("revision", revision)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)

        dtype_value: Optional[Any]
        if torch_dtype is None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                dtype_value = torch.float16
            else:
                dtype_value = None
        elif torch_dtype == "auto":
            dtype_value = "auto"
        else:
            try:
                dtype_value = getattr(torch, torch_dtype)
            except AttributeError as exc:  # pragma: no cover - invalid dtype
                raise ValueError(f"Unsupported torch_dtype value: {torch_dtype}") from exc
        if dtype_value is not None:
            model_kwargs.setdefault("torch_dtype", dtype_value)

        if device_map:
            model_kwargs.setdefault("device_map", device_map)
        elif load_in_4bit or load_in_8bit:
            model_kwargs.setdefault("device_map", "auto")

        if load_in_4bit:
            model_kwargs.setdefault("load_in_4bit", True)
        if load_in_8bit:
            model_kwargs.setdefault("load_in_8bit", True)
        if attn_implementation:
            model_kwargs.setdefault("attn_implementation", attn_implementation)

        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)

        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)

        self.max_new_tokens = max_new_tokens
        default_do_sample = bool(getattr(self.model.generation_config, "do_sample", True))
        if temperature is not None and temperature <= 0:
            LOGGER.info("Received non-positive temperature %.3f; forcing greedy decoding.", temperature)
            temperature = None
            if do_sample is None:
                default_do_sample = False
        self.temperature = temperature
        if do_sample is not None:
            self.do_sample = bool(do_sample)
        else:
            self.do_sample = default_do_sample

        target_device = getattr(self.model, "device", None)
        if target_device is None:
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if isinstance(hf_device_map, dict) and hf_device_map:
                first_device = next(iter(hf_device_map.values()))
                if isinstance(first_device, str):
                    target_device = torch.device(first_device)
        if target_device is None:
            target_device = self.device
        self._input_device = target_device

    def generate(self, image: Image.Image, prompt: str) -> VLMResponse:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_kwargs: Dict[str, Any] = {"add_generation_prompt": True}
            try:
                signature = inspect.signature(self.processor.apply_chat_template)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                signature = None
            target_argument = "conversation"
            if signature:
                param_names = list(signature.parameters.keys())
                if param_names:
                    first_param = param_names[0]
                    if first_param in {"conversation", "messages"}:
                        target_argument = first_param
                if "tokenize" in signature.parameters:
                    chat_kwargs["tokenize"] = False
            chat_kwargs[target_argument] = messages
            text_input = self.processor.apply_chat_template(**chat_kwargs)
            processor_inputs: Dict[str, Any] = {"images": image, "return_tensors": "pt"}
            if isinstance(text_input, dict):
                processor_inputs.update(text_input)
            else:
                processor_inputs["text"] = text_input
            inputs = self.processor(**processor_inputs)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        if self._input_device is not None:
            inputs = inputs.to(self._input_device)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        elif hasattr(self.processor, "tokenizer"):
            text = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:  # pragma: no cover - defensive fallback
            raise AttributeError("Processor does not provide a batch_decode method")
        return VLMResponse(text=text.strip(), raw=output_ids)


class OpenAIVLM(VLMBackend):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 150,
        image_size: Optional[int] = 224,
        image_format: str = "JPEG",
        jpeg_quality: int = 85,
        retries: int = 0,
        supports_temperature: bool = True,
        retry_delay: float = 1.0,
        request_timeout: Optional[float] = 60.0,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIVLM")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_size = image_size
        fmt = image_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"
        self.image_format = fmt
        self.jpeg_quality = jpeg_quality
        self._image_mime = "jpeg" if self.image_format == "JPEG" else self.image_format.lower()
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)
        self.request_timeout = request_timeout
        if self.request_timeout is not None and self.request_timeout <= 0:
            self.request_timeout = None
        self._retryable_exceptions: Tuple[type[BaseException], ...] = tuple(
            exc for exc in (APITimeoutError, APIConnectionError) if exc is not None
        )
        self._status_error = APIStatusError
        self._rate_limit_error = RateLimitError
        self.supports_temperature = supports_temperature
        self.reasoning = reasoning

    def _encode_image(self, image: Image.Image) -> str:
        processed = image.copy()
        if self.image_size:
            resample_attr = getattr(Image, "Resampling", None)
            resample = resample_attr.LANCZOS if resample_attr else Image.LANCZOS
            processed = ImageOps.fit(processed, (self.image_size, self.image_size), method=resample)
        if self.image_format == "JPEG" and processed.mode not in ("RGB", "L"):
            processed = processed.convert("RGB")
        buffered = io.BytesIO()
        save_kwargs: Dict[str, Any] = {}
        if self.image_format == "JPEG":
            save_kwargs["quality"] = self.jpeg_quality
            save_kwargs["optimize"] = True
        processed.save(buffered, format=self.image_format, **save_kwargs)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.retries:
            return False
        if self._retryable_exceptions and isinstance(error, self._retryable_exceptions):
            return True
        if self._status_error is not None and isinstance(error, self._status_error):
            status_code = getattr(error, "status_code", None)
            return bool(status_code and status_code >= 500)
        if self._rate_limit_error is not None and isinstance(error, self._rate_limit_error):
            return True
        return False

    def _backoff_delay(self, attempt: int) -> float:
        if self.retry_delay <= 0:
            return 0.0
        return self.retry_delay * (2 ** attempt)

    def generate(self, image: Image.Image, prompt: str) -> VLMResponse:
        b64_image = self._encode_image(image)
        attempt = 0
        while True:
            try:
                request_kwargs = {
                    "model": self.model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": f"data:image/{self._image_mime};base64,{b64_image}"},
                            ],
                        }
                    ],
                    "max_output_tokens": self.max_tokens,
                }
                if self.supports_temperature and self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                if self.request_timeout is not None:
                    request_kwargs["timeout"] = self.request_timeout
                if self.reasoning:
                    request_kwargs["reasoning"] = self.reasoning
                response = self.client.responses.create(**request_kwargs)
                break
            except Exception as exc:  # pragma: no cover - network errors
                if not self._should_retry(exc, attempt):
                    raise
                remaining = self.retries - attempt
                wait_time = self._backoff_delay(attempt)
                LOGGER.warning(
                    "OpenAI request failed (%s retries remaining): %s",
                    remaining,
                    exc,
                )
                if wait_time > 0:
                    time.sleep(wait_time)
                attempt += 1
                continue
        try:
            output_text = _extract_openai_text(response)
        except ValueError:
            dump_method = getattr(response, "model_dump", None) or getattr(response, "dict", None)
            if dump_method is not None:
                try:
                    LOGGER.error("OpenAI response missing text: %s", dump_method())
                except Exception:  # pragma: no cover
                    LOGGER.error("OpenAI response missing text and could not serialize dump.")
            else:
                LOGGER.error("OpenAI response missing text and no dump method available.")
            raise
        usage = getattr(response, "usage", None)
        token_total = getattr(usage, "total_tokens", None) if usage else None
        return VLMResponse(text=output_text, raw=response, token_usage=token_total)




def load_vlm_backend(config: Dict[str, Any]) -> VLMBackend:
    backend_type = config.get("type", "huggingface")
    if backend_type == "huggingface":
        return HuggingFaceVLM(
            model_id=config["model_id"],
            device=config.get("device", "cpu"),
            max_new_tokens=config.get("max_new_tokens", 64),
            temperature=config.get("temperature"),
            torch_dtype=config.get("torch_dtype"),
            device_map=config.get("device_map"),
            load_in_4bit=config.get("load_in_4bit", False),
            load_in_8bit=config.get("load_in_8bit", False),
            attn_implementation=config.get("attn_implementation"),
            revision=config.get("revision"),
            trust_remote_code=config.get("trust_remote_code", False),
            model_kwargs=config.get("model_kwargs"),
            processor_kwargs=config.get("processor_kwargs"),
            do_sample=config.get("do_sample"),
        )
    if backend_type == "openai":
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env")
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env)
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIVLM(
            model=config["model"],
            api_key=api_key,
            api_base=config.get("api_base"),
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens", 150),
            image_size=config.get("image_size", 224),
            image_format=config.get("image_format", "JPEG"),
            jpeg_quality=config.get("jpeg_quality", 85),
            retries=config.get("retries", 0),
            retry_delay=config.get("retry_delay", 1.0),
            request_timeout=config.get("request_timeout", 60.0),
            supports_temperature=config.get("supports_temperature", True),
            reasoning=config.get("reasoning"),
        )
    raise ValueError(f"Unknown VLM backend type: {backend_type}")


__all__ = [
    "VLMBackend",
    "HuggingFaceVLM",
    "OpenAIVLM",
    "VLMResponse",
    "load_vlm_backend",
]










def _extract_openai_text(response: Any) -> str:
    text_value = getattr(response, "output_text", None)
    if isinstance(text_value, str) and text_value.strip():
        return text_value.strip()
    output_value = getattr(response, "output", None)
    if output_value:
        try:
            first_item = output_value[0]
            content = getattr(first_item, "content", None)
            if content:
                content_item = content[0]
                candidate = getattr(content_item, "text", None) or content_item.get("text")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        except Exception:
            pass
    dump_method = getattr(response, "model_dump", None) or getattr(response, "dict", None)
    if dump_method:
        try:
            data = dump_method()
            if isinstance(data, dict):
                text_candidate = data.get("output_text")
                if isinstance(text_candidate, str) and text_candidate.strip():
                    return text_candidate.strip()
                output_data = data.get("output") or data.get("choices")
                if isinstance(output_data, list) and output_data:
                    first = output_data[0]
                    if isinstance(first, dict):
                        content = first.get("content") or first.get("message", {}).get("content")
                        if isinstance(content, list) and content:
                            piece = content[0]
                            if isinstance(piece, dict):
                                text_candidate = piece.get("text") or piece.get("value")
                                if isinstance(text_candidate, str) and text_candidate.strip():
                                    return text_candidate.strip()
                        text_candidate = first.get("text")
                        if isinstance(text_candidate, str) and text_candidate.strip():
                            return text_candidate.strip()
        except Exception:
            pass
    raise ValueError("OpenAI response did not contain text output")
