"""
embeddings.py
-------------
ChromaDB + OpenAI text-embedding-3-small for semantic search over screenshots.

Flow:
  describe_screenshot() → rich text description
       ↓
  embed(description + tags)   [OpenAI text-embedding-3-small]
       ↓
  store in ChromaDB            [local persistent DB]
       ↓
  search("react hooks error")  → nearest vectors → ranked results
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI

from models import IndexEntry, SearchResult, ScreenshotDescription

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME   = "screenshots"
EMBEDDING_MODEL   = "text-embedding-3-small"
DEFAULT_DB_PATH   = os.path.expanduser("~/.screenshot-brain/chroma")
DEFAULT_N_RESULTS = 10


# ── Embedding store ───────────────────────────────────────────────────────────

class ScreenshotIndex:
    """
    Persistent vector index for screenshots.
    One instance per session — cheap to create, reuses the on-disk DB.
    """

    def __init__(
        self,
        db_path: str | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._db_path = db_path or os.getenv("CHROMA_DB_PATH", DEFAULT_DB_PATH)
        Path(self._db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        self._openai = openai_client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        logger.info(
            "ScreenshotIndex ready — db: %s | indexed: %d",
            self._db_path,
            self._collection.count(),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_document(self, desc: ScreenshotDescription) -> str:
        """
        Build the text that gets embedded.
        Richer text → better semantic search.
        """
        tags_str = " ".join(desc.tags)
        return (
            f"{desc.description}\n"
            f"Category: {desc.suggested_category.value}\n"
            f"Tags: {tags_str}\n"
            f"Filename: {desc.filename}"
        )

    def _make_id(self, path: str) -> str:
        """Stable ID for a screenshot — its absolute path."""
        return path

    async def _embed(self, text: str) -> list[float]:
        """Get OpenAI embedding for a piece of text."""
        response = await self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    # ── Public API ────────────────────────────────────────────────────────────

    async def index(
        self,
        desc: ScreenshotDescription,
        size_kb: float = 0.0,
        modified: str | None = None,
    ) -> IndexEntry:
        """
        Embed a screenshot description and store it in ChromaDB.
        If the same path was already indexed, it gets updated (upsert).
        """
        doc_text  = self._make_document(desc)
        embedding = await self._embed(doc_text)
        now       = datetime.utcnow().isoformat()

        entry = IndexEntry(
            path=desc.path,
            filename=desc.filename,
            description=desc.description,
            category=desc.suggested_category.value,
            tags=desc.tags,
            indexed_at=now,
            size_kb=size_kb,
            modified=modified or now,
        )

        self._collection.upsert(
            ids=[self._make_id(desc.path)],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "path":        entry.path,
                "filename":    entry.filename,
                "description": entry.description,
                "category":    entry.category,
                "tags":        ",".join(entry.tags),   # ChromaDB metadata must be scalar
                "indexed_at":  entry.indexed_at,
                "size_kb":     entry.size_kb,
                "modified":    entry.modified,
            }],
        )

        logger.debug("Indexed: %s → %s", desc.filename, desc.suggested_category.value)
        return entry

    async def search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        category_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search over indexed screenshots.

        Args:
            query:           Natural language query, e.g. "react hooks error"
            n_results:       Max results to return
            category_filter: Optional category to restrict search (e.g. "code/errors")

        Returns:
            List of SearchResult sorted by similarity descending.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = await self._embed(query)

        where = {"category": category_filter} if category_filter else None

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            where=where,
            include=["metadatas", "distances"],
        )

        results: list[SearchResult] = []
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for meta, dist in zip(metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, 0 = opposite
            similarity = round(1 - (dist / 2), 4)
            results.append(SearchResult(
                path=meta["path"],
                filename=meta["filename"],
                category=meta["category"],
                description=meta["description"],
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                similarity=similarity,
                modified=meta.get("modified", ""),
            ))

        return sorted(results, key=lambda r: r.similarity, reverse=True)

    def update_path(self, old_path: str, new_path: str, new_filename: str) -> bool:
        """
        Update the stored path/filename after a move or rename.
        Returns True if the entry existed and was updated.
        """
        old_id = self._make_id(old_path)
        existing = self._collection.get(ids=[old_id], include=["embeddings", "documents", "metadatas"])

        if not existing["ids"]:
            return False

        meta = existing["metadatas"][0]
        meta["path"]     = new_path
        meta["filename"] = new_filename

        # Re-insert with new ID, delete old
        self._collection.upsert(
            ids=[self._make_id(new_path)],
            embeddings=[existing["embeddings"][0]],
            documents=[existing["documents"][0]],
            metadatas=[meta],
        )
        self._collection.delete(ids=[old_id])
        return True

    def delete(self, path: str) -> bool:
        """Remove a screenshot from the index."""
        self._collection.delete(ids=[self._make_id(path)])
        return True

    def is_indexed(self, path: str) -> bool:
        """Check if a screenshot is already in the index."""
        result = self._collection.get(ids=[self._make_id(path)])
        return len(result["ids"]) > 0

    @property
    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict:
        """Return index stats — total count, breakdown by category."""
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        by_category: dict[str, int] = {}
        for m in all_meta:
            cat = m.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
        return {
            "total_indexed": len(all_meta),
            "by_category": by_category,
            "db_path": self._db_path,
        }