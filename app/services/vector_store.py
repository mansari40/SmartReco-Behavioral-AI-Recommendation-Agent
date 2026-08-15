"""
Thin async-friendly wrapper around a local persistent ChromaDB collection.
Chroma's client is sync, so calls are pushed to a thread via asyncio.to_thread
to keep this non-blocking inside FastAPI's async routes.

Products are dual-written here from app/routers/products.py (next step):
SQL row first, then this store. Failures here should NOT be swallowed —
the caller is responsible for recording sync_status/sync_error on the
Product row based on what this module raises.
"""
import sys
import types


if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

import asyncio
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import settings


class _NoOpEmbeddingFunction(EmbeddingFunction):
    """This app always supplies embeddings explicitly (via OpenRouter) on
    every upsert/query call, so Chroma's own embedding function is never
    invoked. This stub exists only to satisfy Chroma's API requirement
    that a collection have an embedding_function."""
    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError(
            "vector_store's collection should never be asked to embed text itself — "
            "embeddings must always be passed explicitly."
        )


_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection,
    embedding_function=_NoOpEmbeddingFunction(),
)


def _upsert_sync(vector_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    _collection.upsert(
        ids=[vector_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata],
    )


def _delete_sync(vector_id: str) -> None:
    _collection.delete(ids=[vector_id])


def _query_sync(embedding: list[float], top_k: int, where: dict | None) -> dict:
    return _collection.query(query_embeddings=[embedding], n_results=top_k, where=where)


async def upsert_product(vector_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    await asyncio.to_thread(_upsert_sync, vector_id, embedding, document, metadata)


async def delete_product(vector_id: str) -> None:
    await asyncio.to_thread(_delete_sync, vector_id)


async def query(embedding: list[float], top_k: int = 10, where: dict | None = None) -> dict:
    """`where` enables metadata filtering (e.g. category, price range) for
    hybrid retrieval — used by the agent's retrieve node later on."""
    return await asyncio.to_thread(_query_sync, embedding, top_k, where)