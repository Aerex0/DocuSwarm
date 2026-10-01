"""Groq API client wrapper"""

import os
from typing import Any, List, Dict, Optional
from langchain_groq import ChatGroq
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.config import config


class GroqClient:
    """Wrapper for Groq API with retry logic and fallback"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key = config.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.model = model or config.model_reasoning
        self.temperature = (
            temperature if temperature is not None else config.temperature
        )

        # Reasoning models bill chain-of-thought against max_tokens; a caller
        # asking for 800 tokens can end up with an empty answer. Clamp upward.
        requested = max_tokens or config.max_tokens
        if config.is_reasoning_model:
            self.max_tokens = max(requested, config.min_reasoning_max_tokens)
        else:
            self.max_tokens = requested

        # Initialize LangChain Groq chat
        self.chat = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Initialize native Groq client for embeddings
        self.client = Groq(api_key=self.api_key)

    @staticmethod
    def _extract_content(response: Any) -> str:
        """
        Return answer text from a chat response.

        Reasoning models can return empty content when the token budget is
        consumed by reasoning, so fall back to the reasoning field rather than
        handing callers an empty string.
        """
        content = (getattr(response, "content", None) or "").strip()
        if content:
            return content

        reasoning = (
            getattr(response, "reasoning_content", None)
            or getattr(response, "additional_kwargs", {}).get("reasoning_content")
            or ""
        )
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        return content

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def invoke(self, prompt: str) -> str:
        """Invoke Groq LLM with retry logic"""
        try:
            response = self.chat.invoke(prompt)
            return self._extract_content(response)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                # Try fallback model
                fallback_chat = ChatGroq(
                    api_key=self.api_key,
                    model=config.model_fallback,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response = fallback_chat.invoke(prompt)
                return self._extract_content(response)
            raise e

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using Groq"""
        embeddings = []

        for text in texts:
            try:
                # Groq doesn't have native embedding endpoint yet,
                # so we'll use a workaround with sentence-transformers
                from sentence_transformers import SentenceTransformer

                # Use a lightweight model for embeddings
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embedding = model.encode(text).tolist()
                embeddings.append(embedding)

            except Exception as e:
                print(f"Error embedding text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 384)  # all-MiniLM-L6-v2 dimension

        return embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        return self.embed([text])[0]


class GroqEmbeddings:
    """Embeddings function for ChromaDB (v0.4.16+ compatible)"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return self.model.encode(input).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain/Chroma compatibility for document embeddings."""
        return self.model.encode(texts).tolist()

    def embed_query(self, input: str) -> List[float]:
        """LangChain/Chroma compatibility for query embedding."""
        return self.model.encode(input).tolist()

    def name(self) -> str:
        """Return the name of the embedding function"""
        return "sentence-transformers-all-MiniLM-L6-v2"


def call_groq_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Convenience wrapper used by agents to call Groq LLM."""
    client = GroqClient(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return client.invoke(prompt)
