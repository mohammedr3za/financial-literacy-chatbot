"""Chat-model and embedder factories. The graph never constructs models directly."""
from .config import Settings, get_settings


def get_chat_model(settings: Settings | None = None):
    s = settings or get_settings()
    if s.backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=s.model, base_url=s.base_url, reasoning=s.thinking, num_ctx=s.num_ctx,
                          temperature=s.temperature, num_predict=s.max_output_tokens, keep_alive="60m")
    if s.backend == "openai":  # vLLM or any OpenAI-compatible server (fine-tune data generation)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use FINBOT_BACKEND=openai.") from exc
        return ChatOpenAI(model=s.model, base_url=s.base_url, api_key="not-needed", temperature=s.temperature)
    raise ValueError(f"Unknown FINBOT_BACKEND: {s.backend}")


def get_embedder(settings: Settings | None = None):
    s = settings or get_settings()
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=s.embed_model, base_url=s.embed_url)
