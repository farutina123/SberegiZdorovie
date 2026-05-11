#!/usr/bin/env python3
"""
Индексация Markdown-документов в Supabase (pgvector) через LangChain + OpenAI Embeddings.

Запуск вручную после обновления базы знаний:
  python scripts/index_knowledge.py

Переменные окружения (.env):
  OPENAI_API_KEY           — обязательно
  SUPABASE_URL             — обязательно
  SUPABASE_SERVICE_KEY     — service_role (рекомендуется) или совместимость с SUPABASE_SERVICE_ROLE_KEY
  CONTEXT_DIR              — каталог с .md (по умолчанию context)
  OPENAI_EMBEDDING_MODEL   — по умолчанию text-embedding-3-small (1536 измерений)
  OPENAI_BASE_URL          — опционально, совместимый API
  CHUNK_SIZE               — размер фрагмента символов (по умолчанию 1200)
  CHUNK_OVERLAP            — перекрытие (по умолчанию 150)
  SUPABASE_TABLE           — имя таблицы (по умолчанию documents)
  SUPABASE_MATCH_RPC       — имя RPC для поиска (по умолчанию match_documents)
  INDEX_BATCH_SIZE         — размер батча при вставке в Supabase (по умолчанию 80; меньше — надёжнее через VPN)
  SUPABASE_POSTGREST_TIMEOUT — таймаут HTTP к PostgREST, сек (по умолчанию 300)
  SUPABASE_INSECURE_SKIP_VERIFY — если 1: отключить проверку TLS к Supabase (только отладка / корпоративный SSL)
  SKIP_SUPABASE_TRUNCATE       — если 1: не вызывать DELETE (очистите таблицу вручную в SQL Editor: TRUNCATE documents;)

Перед первым запуском выполните SQL из scripts/supabase_rag_schema.sql в Supabase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from postgrest.exceptions import APIError
from supabase import create_client


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_supabase_url(url: str) -> str:
    """
    Ожидается только хост проекта: https://<ref>.supabase.co
    Без /rest/v1, без ссылки на dashboard.
    """
    u = url.strip().rstrip("/")
    lower = u.lower()
    if "/dashboard" in lower or "supabase.com/dashboard" in lower:
        raise ValueError(
            "SUPABASE_URL выглядит как адрес из браузера (dashboard). "
            "Возьмите Project URL из Settings → General или Integrations → Data API "
            "(формат https://xxxx.supabase.co)."
        )
    if "/rest/" in u:
        u = u.split("/rest/", 1)[0].rstrip("/")
    return u


def normalize_openai_base_url(raw: str | None) -> str | None:
    """Если в .env указали домен без https — добавляем схему (иначе httpx: UnsupportedProtocol)."""
    if raw is None:
        return None
    u = raw.strip()
    if not u:
        return None
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u.lstrip("/")
    return u.rstrip("/")


def _supabase_postgrest_timeout_sec() -> float:
    try:
        return float(os.getenv("SUPABASE_POSTGREST_TIMEOUT", "300").strip())
    except ValueError:
        return 300.0


def create_supabase_client(url: str, key: str):
    """Создаёт клиент Supabase; при проблемах с корпоративным CA см. SUPABASE_INSECURE_SKIP_VERIFY."""
    import httpx

    try:
        from supabase import ClientOptions
    except ImportError:  # pragma: no cover
        from supabase.lib.client_options import ClientOptions

    t = _supabase_postgrest_timeout_sec()
    # Иначе при медленном канале (VPN) upsert векторов отваливается по ReadTimeout (~дефолт httpx).
    timeout = httpx.Timeout(t, connect=60.0)
    verify_tls = not _env_truthy("SUPABASE_INSECURE_SKIP_VERIFY")
    http_client = httpx.Client(verify=verify_tls, timeout=timeout)
    opts = ClientOptions(httpx_client=http_client, postgrest_client_timeout=t)
    return create_client(url, key, options=opts)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251", errors="replace")


def load_markdown_documents(base: Path) -> list[Document]:
    """Загружает все *.md под base рекурсивно."""
    docs: list[Document] = []
    if not base.is_dir():
        return docs

    for path in sorted(base.rglob("*.md")):
        if not path.is_file():
            continue
        text = _read_text_file(path).strip()
        if not text:
            continue
        rel = str(path.relative_to(base)).replace("\\", "/")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": rel},
            )
        )
    return docs


def truncate_documents_table(client, table_name: str) -> None:
    """Удаляет все строки перед полной переиндексацией."""
    nil = "00000000-0000-0000-0000-000000000000"
    try:
        client.table(table_name).delete().neq("id", nil).execute()
    except APIError as exc:
        code = exc.code or ""
        msg = (
            f"Не удалось очистить таблицу «{table_name}» (PostgREST {code}). "
            "Проверьте: 1) выполнен SQL из scripts/supabase_rag_schema.sql; "
            "2) таблица в схеме public и совпадает с SUPABASE_TABLE; "
            "3) SUPABASE_URL — только https://<ref>.supabase.co без лишних путей."
        )
        raise RuntimeError(msg.strip()) from exc


def main() -> int:
    load_dotenv()
    # Пустая строка OPENAI_BASE_URL= в .env попадает в os.environ как '' и ломает клиент (UnsupportedProtocol).
    for _key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        _v = os.environ.get(_key)
        if _v is not None and not str(_v).strip():
            del os.environ[_key]

    repo_root = Path(__file__).resolve().parents[1]

    context_dir = Path(os.getenv("CONTEXT_DIR", "context").strip())
    if not context_dir.is_absolute():
        context_dir = repo_root / context_dir

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    if not openai_key:
        print("Ошибка: задайте OPENAI_API_KEY", file=sys.stderr)
        return 1
    if not supabase_url or not supabase_key:
        print(
            "Ошибка: задайте SUPABASE_URL и SUPABASE_SERVICE_KEY (или SUPABASE_SERVICE_ROLE_KEY)",
            file=sys.stderr,
        )
        return 1

    try:
        supabase_url = normalize_supabase_url(supabase_url)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    table_name = os.getenv("SUPABASE_TABLE", "documents").strip()
    query_name = os.getenv("SUPABASE_MATCH_RPC", "match_documents").strip()
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200").strip())
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150").strip())
    batch_size = int(os.getenv("INDEX_BATCH_SIZE", "80").strip())

    base_url = normalize_openai_base_url(os.getenv("OPENAI_BASE_URL"))

    raw_docs = load_markdown_documents(context_dir)
    if not raw_docs:
        print(f"Нет .md файлов в каталоге: {context_dir}", file=sys.stderr)
        return 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    for i, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = i

    emb_kwargs: dict = {"model": embedding_model, "api_key": openai_key}
    if base_url:
        emb_kwargs["base_url"] = base_url
    # Tiktoken по умолчанию качает BPE с blob.core.windows.net (requests) — на части сетей SSL падает.
    # Размер фрагментов уже задаётся RecursiveCharacterTextSplitter.
    emb_kwargs["tiktoken_enabled"] = False
    emb_kwargs["check_embedding_ctx_length"] = False
    if _env_truthy("OPENAI_INSECURE_SKIP_VERIFY"):
        try:
            import httpx

            emb_kwargs["http_client"] = httpx.Client(verify=False)
        except Exception as exc:  # pragma: no cover
            print(f"Предупреждение: не удалось отключить SSL для OpenAI: {exc}", file=sys.stderr)

    embeddings = OpenAIEmbeddings(**emb_kwargs)

    client = create_supabase_client(supabase_url, supabase_key)

    if _env_truthy("SKIP_SUPABASE_TRUNCATE"):
        print(
            "Пропуск очистки таблицы (SKIP_SUPABASE_TRUNCATE=1). "
            "Убедитесь, что таблица пуста или вас устраивают дубликаты при повторной загрузке.",
            file=sys.stderr,
        )
    else:
        truncate_documents_table(client, table_name)

    SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=client,
        table_name=table_name,
        query_name=query_name,
        chunk_size=batch_size,
    )

    print(f"Загружено фрагментов: {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
