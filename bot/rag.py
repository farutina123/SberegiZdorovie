from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from postgrest.exceptions import APIError
from supabase import create_client


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_empty_openai_url_env() -> None:
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        v = os.environ.get(key)
        if v is not None and not str(v).strip():
            del os.environ[key]


def normalize_supabase_url(url: str) -> str:
    u = url.strip().rstrip("/")
    lower = u.lower()
    if "/dashboard" in lower or "supabase.com/dashboard" in lower:
        raise ValueError(
            "SUPABASE_URL должен быть вида https://xxxx.supabase.co (не ссылка из браузера dashboard)."
        )
    if "/rest/" in u:
        u = u.split("/rest/", 1)[0].rstrip("/")
    return u


def _supabase_postgrest_timeout_sec() -> float:
    try:
        return float(os.getenv("SUPABASE_POSTGREST_TIMEOUT", "300").strip())
    except ValueError:
        return 300.0


def create_supabase_client(url: str, key: str):
    try:
        from supabase import ClientOptions
    except ImportError:  # pragma: no cover
        from supabase.lib.client_options import ClientOptions

    t = _supabase_postgrest_timeout_sec()
    timeout = httpx.Timeout(t, connect=60.0)
    verify_tls = not _env_truthy("SUPABASE_INSECURE_SKIP_VERIFY")
    http_client = httpx.Client(verify=verify_tls, timeout=timeout)
    opts = ClientOptions(httpx_client=http_client, postgrest_client_timeout=t)
    return create_client(url, key, options=opts)


def normalize_openai_base_url(raw: str | None) -> str | None:
    if raw is None:
        return None
    u = raw.strip()
    if not u:
        return None
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u.lstrip("/")
    return u.rstrip("/")


def _build_embeddings(openai_api_key: str) -> OpenAIEmbeddings:
    base_url = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    emb_kwargs: dict = {"model": model, "api_key": openai_api_key}
    if base_url:
        emb_kwargs["base_url"] = base_url
    emb_kwargs["tiktoken_enabled"] = False
    emb_kwargs["check_embedding_ctx_length"] = False
    if _env_truthy("OPENAI_INSECURE_SKIP_VERIFY"):
        emb_kwargs["http_client"] = httpx.Client(verify=False)
    return OpenAIEmbeddings(**emb_kwargs)


@dataclass(frozen=True)
class RagRetriever:
    client: Any
    embeddings: OpenAIEmbeddings
    rpc_name: str
    top_k: int


def search_documents_sync(retriever: RagRetriever, query: str) -> list[tuple[Document, float]]:
    """
    Вызов SQL match_documents (pgvector) через PostgREST.
    Обходит поломку LangChain SupabaseVectorStore на части связок supabase-py.
    """
    vec = retriever.embeddings.embed_query(query)
    res = (
        retriever.client.rpc(
            retriever.rpc_name,
            {"query_embedding": vec, "match_count": retriever.top_k, "filter": {}},
        ).execute()
    )
    rows = res.data or []
    out: list[tuple[Document, float]] = []
    for row in rows:
        raw_meta = row.get("metadata")
        meta: dict
        if isinstance(raw_meta, dict):
            meta = raw_meta
        elif isinstance(raw_meta, str) and raw_meta.strip():
            try:
                meta = json.loads(raw_meta)
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = {}
        doc = Document(page_content=(row.get("content") or "").strip(), metadata=meta)
        sim = float(row.get("similarity") if row.get("similarity") is not None else 0.0)
        out.append((doc, sim))
    return out


def init_rag_optional(logger: logging.Logger, openai_api_key: str) -> RagRetriever | None:
    """Поднимает клиент Supabase + эмбеддинги для RPC match_documents."""
    if not _env_truthy("RAG_ENABLED", default="1"):
        logger.info("RAG отключён (RAG_ENABLED=0).")
        return None

    _strip_empty_openai_url_env()

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip() or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    if not url or not key:
        logger.warning("RAG недоступен: не заданы SUPABASE_URL или SUPABASE_SERVICE_KEY.")
        return None

    try:
        url_n = normalize_supabase_url(url)
    except ValueError as exc:
        logger.warning("RAG недоступен: %s", exc)
        return None

    table_name = os.getenv("SUPABASE_TABLE", "documents").strip()
    query_name = os.getenv("SUPABASE_MATCH_RPC", "match_documents").strip()

    try:
        client = create_supabase_client(url_n, key)
        client.table(table_name).select("id").limit(1).execute()
    except APIError as exc:
        logger.warning("RAG недоступен: не удалось проверить таблицу «%s»: %s", table_name, exc)
        return None
    except Exception:
        logger.exception("RAG недоступен: ошибка подключения к Supabase.")
        return None

    if not openai_api_key.strip():
        logger.warning("RAG недоступен: пустой OPENAI_API_KEY.")
        return None

    try:
        embeddings = _build_embeddings(openai_api_key)
        # Smoke RPC (та же сигнатура, что при запросе оператора)
        _probe = embeddings.embed_query("ping")
        client.rpc(
            query_name,
            {"query_embedding": _probe, "match_count": 1, "filter": {}},
        ).execute()
    except Exception:
        logger.exception(
            "RAG недоступен: не удалось создать эмбеддинги или вызвать RPC «%s».",
            query_name,
        )
        return None

    try:
        top_k = int(os.getenv("RAG_TOP_K", "6").strip())
    except ValueError:
        top_k = 6
    top_k = max(1, min(top_k, 20))

    logger.info("RAG включён: таблица=%s, rpc=%s, top_k=%s", table_name, query_name, top_k)
    return RagRetriever(client=client, embeddings=embeddings, rpc_name=query_name, top_k=top_k)


def _preview(text: str, limit: int = 220) -> str:
    t = text.replace("\r", " ").replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


async def retrieve_rag_context(retriever: RagRetriever, query: str, logger: logging.Logger) -> str:
    """
    Возвращает текстовый блок для системного промпта + логирует найденные фрагменты в консоль.
    При ошибке или пустом результате — пустая строка.
    """

    def _run() -> list[tuple[Document, float]]:
        return search_documents_sync(retriever, query)

    try:
        scored = await asyncio.to_thread(_run)
    except Exception:
        logger.exception("RAG: ошибка векторного поиска для запроса.")
        return ""

    if not scored:
        logger.info('RAG: совпадений не найдено для запроса "%s"', _preview(query, 160))
        return ""

    logger.info("RAG: найдено фрагментов: %s", len(scored))
    parts: list[str] = []
    for i, (doc, score) in enumerate(scored, start=1):
        src = doc.metadata.get("source", "?")
        chunk_idx = doc.metadata.get("chunk_index")
        logger.info(
            "RAG [%s/%s] score=%.4f source=%s chunk_index=%s preview=%s",
            i,
            len(scored),
            float(score),
            src,
            chunk_idx,
            _preview(doc.page_content),
        )
        head = f"[{i}] источник: {src}"
        if chunk_idx is not None:
            head += f", chunk_index: {chunk_idx}"
        parts.append(f"{head}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
