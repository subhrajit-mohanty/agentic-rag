"""
LLM Client Service

Provides async LLM interaction with support for OpenAI (default),
Ollama (local), and Anthropic.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> T:
        """Generate structured output matching a Pydantic model."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text completion."""
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI LLM client (DEFAULT).
    
    Supports:
    - Text generation with GPT models
    - Structured output with JSON mode
    - Streaming responses
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key or settings.llm.openai_api_key
        self.model = model or settings.llm.openai_model
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=timeout
        )
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion using OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model to use (overrides default)
            
        Returns:
            Generated text
        """
        settings = get_settings()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or settings.llm.temperature,
                max_tokens=max_tokens or settings.llm.max_tokens,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Generate structured output matching a Pydantic model.
        
        Uses JSON mode for reliable structured output.
        
        Args:
            prompt: User prompt
            response_model: Pydantic model class for response
            system_prompt: Optional system prompt
            
        Returns:
            Parsed Pydantic model instance
        """
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        
        # Build system prompt with schema
        json_system = f"""You must respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no other text or markdown."""

        full_system = f"{system_prompt}\n\n{json_system}" if system_prompt else json_system
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use low temperature for structured output
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse JSON and validate against model
            parsed = json.loads(content)
            return response_model.model_validate(parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            raise ValueError(f"Invalid JSON from OpenAI: {e}")
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use
            
        Yields:
            Text chunks as they're generated
        """
        settings = get_settings()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=settings.llm.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            # Simple completion to verify API access
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self.model
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai",
                "message": str(e)
            }


class OllamaClient(BaseLLMClient):
    """
    Ollama LLM client for local model inference.
    
    Supports:
    - Text generation
    - Structured output with JSON mode
    - Streaming responses
    - Multiple models
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        settings = get_settings()
        self.host = host or settings.llm.ollama_host
        self.model = model or settings.llm.ollama_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text completion using Ollama."""
        settings = get_settings()
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.llm.temperature,
                "num_predict": max_tokens or settings.llm.max_tokens,
            }
        }
        
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "")
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> T:
        """Generate structured output matching a Pydantic model."""
        schema = response_model.model_json_schema()
        
        json_system = f"""You must respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with the JSON object, no other text."""

        full_system = f"{system_prompt}\n\n{json_system}" if system_prompt else json_system
        
        client = await self._get_client()
        
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("message", {}).get("content", "{}")
            
            parsed = json.loads(content)
            return response_model.model_validate(parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama JSON response: {e}")
            raise ValueError(f"Invalid JSON from Ollama: {e}")
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text completion."""
        settings = get_settings()
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": settings.llm.temperature}
        }
        
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama health and available models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "status": "healthy",
                    "provider": "ollama",
                    "host": self.host,
                    "available_models": models,
                    "default_model": self.model
                }
            else:
                return {
                    "status": "unhealthy",
                    "provider": "ollama",
                    "message": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "ollama",
                "message": str(e)
            }


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and development without API keys."""
    
    def __init__(self, latency_ms: int = 500):
        self.latency_ms = latency_ms
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        await asyncio.sleep(self.latency_ms / 1000)
        
        prompt_lower = prompt.lower()
        
        if "leave" in prompt_lower or "pto" in prompt_lower:
            return "Based on our HR policies, full-time employees are entitled to 20 days of paid time off per year. You can carry over up to 5 unused days to the next year."
        elif "s3" in prompt_lower or "aws" in prompt_lower:
            return "S3 bucket access requires the 'Engineering-Role' IAM role. All requests must be authenticated through SSO. Please ensure data encryption is enabled."
        elif "project ares" in prompt_lower:
            return "Project Ares is our 2025 cloud migration initiative. All legacy systems will be migrated to AWS. Please refer to the Project Ares documentation for detailed specifications."
        else:
            return f"I found relevant information in our knowledge base regarding your query about: {prompt[:100]}..."
    
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs
    ) -> T:
        await asyncio.sleep(self.latency_ms / 1000)
        return response_model.model_validate({})
    
    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        response = await self.generate(prompt, **kwargs)
        words = response.split()
        
        for word in words:
            await asyncio.sleep(0.05)
            yield word + " "


# Global LLM client instance
_llm_client: Optional[BaseLLMClient] = None


def get_llm_client(
    provider: Optional[str] = None,
    use_mock: bool = False
) -> BaseLLMClient:
    """
    Get or create LLM client instance.
    
    Args:
        provider: LLM provider ("openai", "ollama", or "mock")
        use_mock: If True, return mock client for testing
        
    Returns:
        LLM client instance
    """
    global _llm_client
    
    if use_mock:
        return MockLLMClient()
    
    settings = get_settings()
    provider = provider or settings.llm.provider
    
    if _llm_client is None:
        if provider == "openai":
            _llm_client = OpenAIClient()
        elif provider == "ollama":
            _llm_client = OllamaClient()
        else:
            # Default to OpenAI
            _llm_client = OpenAIClient()
        
        logger.info(f"LLM client created: {provider}")
    
    return _llm_client


async def close_llm_client() -> None:
    """Close LLM client connections."""
    global _llm_client
    if _llm_client:
        if isinstance(_llm_client, OllamaClient):
            await _llm_client.close()
        _llm_client = None
