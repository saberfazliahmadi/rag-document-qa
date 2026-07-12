import pytest

from rag.config import Settings


def test_defaults(monkeypatch):
    for var in ("SEARCH_MODE", "CHUNK_SIZE", "USE_RERANKER", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    assert settings.search_mode == "hybrid"
    assert settings.chunk_size == 500
    assert settings.use_reranker is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SEARCH_MODE", "dense")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("USE_RERANKER", "false")
    settings = Settings()
    assert settings.search_mode == "dense"
    assert settings.chunk_size == 800
    assert settings.use_reranker is False


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        settings.require_api_key()
