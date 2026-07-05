"""LLM provider abstraction — Groq and Ollama behind one interface.

Both ChatGroq and ChatOllama implement LangChain's BaseChatModel, so downstream
code (generator, critic, reformulator) never knows which backend is active. Switch
via the LLM_PROVIDER env var ("groq" or "ollama").

Example:
    from rag.llm_provider import get_generator_llm, get_critic_llm

    llm = get_generator_llm()          # 8B model, fast/cheap
    critic = get_critic_llm()          # 70B model, stronger judgment
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings


@lru_cache(maxsize=4)
def _get_groq_llm(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Create a ChatGroq instance (cached per model+temperature)."""
    from langchain_groq import ChatGroq

    if not settings.llm.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and add it to your .env file."
        )

    return ChatGroq(
        api_key=settings.llm.groq_api_key,
        model=model,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def _get_ollama_llm(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Create a ChatOllama instance (cached per model+temperature)."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "Ollama support requires the langchain-ollama package. "
            "Install it with: pip install 'guarded-rag-system[ollama]'"
        )

    return ChatOllama(
        model=model,
        base_url=settings.llm.ollama_base_url,
        temperature=temperature,
    )


def get_generator_llm(
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Get the LLM used for answer generation (smaller, faster model).

    Args:
        model: Override the default model name. If None, uses settings.
        temperature: LLM temperature (0.0 = deterministic).

    - Groq:   llama-3.1-8b-instant (default)
    - Ollama: llama3.1:8b-instruct-q4_K_M (default)
    """
    if settings.llm.provider == "ollama":
        return _get_ollama_llm(model or settings.llm.ollama_model, temperature)
    return _get_groq_llm(model or settings.llm.groq_generator_model, temperature)


def get_critic_llm(temperature: float = 0.0) -> BaseChatModel:
    """Get the LLM used for self-critique and evaluation (larger, stronger model).

    Uses a *different* model than the generator to avoid correlated blind spots:
    - Groq:   llama-3.3-70b-versatile (default)
    - Ollama: falls back to the same local model (limited by local GPU VRAM)
    """
    if settings.llm.provider == "ollama":
        # Ollama can't easily run a 70B model locally — use the same model
        return _get_ollama_llm(settings.llm.ollama_model, temperature)
    return _get_groq_llm(settings.llm.groq_critic_model, temperature)
