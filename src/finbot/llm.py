"""Chat-model and embedder factories. The graph never constructs models directly."""
from .config import Settings, get_settings


def get_chat_model(settings: Settings | None = None):
    s = settings or get_settings()
    if s.backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=s.model, base_url=s.base_url, reasoning=s.thinking, num_ctx=s.num_ctx,
                          temperature=s.temperature, num_predict=s.output_cap("ollama"), keep_alive="60m")
    if s.backend == "openai":  # vLLM or any OpenAI-compatible server (fine-tune data generation)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use FINBOT_BACKEND=openai.") from exc
        return ChatOpenAI(model=s.model, base_url=s.base_url, api_key="not-needed", temperature=s.temperature,
                          max_tokens=s.output_cap("openai"))
    raise ValueError(f"Unknown FINBOT_BACKEND: {s.backend}")


def _ollama(model: str, url: str, s: Settings):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=url, reasoning=s.thinking, num_ctx=s.num_ctx, temperature=s.temperature,
                      num_predict=s.output_cap("ollama"), keep_alive="60m")


def get_runnables(tools: list, settings: Settings | None = None):
    """(plain model, model with tools). With the GPU backend, each falls back to the local Ollama model on any error
    (unreachable tunnel, timeout, server error), so the booth keeps working offline (SPEC section 2)."""
    s = settings or get_settings()
    from .tools.schemas import compact_tool_schemas
    if s.backend != "openai":
        m = _ollama(s.model, s.base_url, s)
        return m, m.bind_tools(compact_tool_schemas(tools) if s.compact_tools_ollama else tools)
    primary = _openai(s)
    compact = compact_tool_schemas(tools)   # vLLM renders schemas into the 16k context: send the compact copies (SPEC 3.1)
    if not _has_fallback(s):  # e.g. on the GPU box itself
        return primary, primary.bind_tools(compact)
    fallback = _ollama(s.fallback_model, s.fallback_url, s)
    return (primary.with_fallbacks([fallback]),
            primary.bind_tools(compact).with_fallbacks([fallback.bind_tools(tools)]))


def get_forced_tool_runnable(tools: list, settings: Settings | None = None):
    """The model with tools that must call one (tool_choice "required"), for the revision of a calculation answered
    without a tool (SPEC 3.1). None on Ollama, which has no tool_choice."""
    s = settings or get_settings()
    if s.backend != "openai":
        return None
    from .tools.schemas import compact_tool_schemas
    forced = _openai(s).bind_tools(compact_tool_schemas(tools), tool_choice="required")
    if not _has_fallback(s):
        return forced
    return forced.with_fallbacks([_ollama(s.fallback_model, s.fallback_url, s).bind_tools(tools)])


def _openai(s: Settings):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install langchain-openai to use FINBOT_BACKEND=openai.") from exc
    import httpx
    # vLLM closes idle keep-alive connections after 5 s; reusing one it just closed fails with a connection error
    # (max_retries=0), so drop idle connections sooner (2 live question-bank runs lost answers to this)
    client = httpx.Client(timeout=s.request_timeout, limits=httpx.Limits(keepalive_expiry=2.0))
    return ChatOpenAI(model=s.model, base_url=s.base_url, api_key="not-needed", temperature=s.temperature,
                      max_tokens=s.output_cap("openai"), timeout=s.request_timeout, max_retries=0, http_client=client,
                      extra_body={"chat_template_kwargs": {"enable_thinking": bool(s.thinking)}})


def _has_fallback(s: Settings) -> bool:
    return bool(s.fallback_model) and s.fallback_model.lower() != "none"


def get_embedder(settings: Settings | None = None):
    s = settings or get_settings()
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=s.embed_model, base_url=s.embed_url)
