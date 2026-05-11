-- Однократная настройка Supabase для RAG (pgvector).
-- Выполните в SQL Editor проекта Supabase.

create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid (),
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector (1536)
);

-- Индекс для поиска по косинусному расстоянию (опционально, после первой загрузки данных).
-- create index on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Функция для retrieval (LangChain SupabaseVectorStore по умолчанию ожидает имя match_documents).
create or replace function match_documents (
  query_embedding vector (1536),
  match_count int default 10,
  filter jsonb default '{}'::jsonb
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    (1 - (documents.embedding <=> query_embedding))::float as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
