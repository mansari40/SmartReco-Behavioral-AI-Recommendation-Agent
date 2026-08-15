"""
Thin async-friendly wrapper around a local persistent ChromaDB collection.
...
"""
import sys
import types

print("[VECSTORE CHECKPOINT] vector_store.py starting", flush=True)

if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

import asyncio
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("[VECSTORE CHECKPOINT] about to import chromadb", flush=True)
import chromadb
print("[VECSTORE CHECKPOINT] chromadb imported successfully", flush=True)

from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import settings


class _NoOpEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError(
            "vector_store's collection should never be asked to embed text itself — "
            "embeddings must always be passed explicitly."
        )


print("[VECSTORE CHECKPOINT] about to create PersistentClient", flush=True)
_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)
print("[VECSTORE CHECKPOINT] PersistentClient created, about to get_or_create_collection", flush=True)
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection,
    embedding_function=_NoOpEmbeddingFunction(),
)
print("[VECSTORE CHECKPOINT] collection ready", flush=True)


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
    return await asyncio.to_thread(_query_sync, embedding, top_k, where)