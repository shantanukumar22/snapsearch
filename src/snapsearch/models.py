"""
Shared Pydantic models for snapsearch.
These are the data contracts between vision, embeddings, and the MCP tools.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ── Categories ────────────────────────────────────────────────────────────────

class Category(str, Enum):
    CODE_ERROR    = "code/errors"
    CODE_SNIPPET  = "code/snippets"
    DESIGN_FIGMA  = "design/figma"
    DESIGN_OTHER  = "design/other"
    CHATS         = "chats"
    DOCS          = "docs"
    WEB           = "web"
    MEMES         = "memes"
    WALLPAPERS    = "wallpapers"
    MISC          = "misc"


# ── Screenshot metadata ───────────────────────────────────────────────────────

class ScreenshotMeta(BaseModel):
    filename: str
    path: str
    size_kb: float
    modified: str                       # ISO string
    extension: str
    category: Optional[str] = None      # relative subfolder or None if uncategorized

    @classmethod
    def from_path(cls, p: Path, root: Path) -> "ScreenshotMeta":
        stat = p.stat()
        relative = p.relative_to(root)
        cat = str(relative.parent) if str(relative.parent) != "." else None
        return cls(
            filename=p.name,
            path=str(p),
            size_kb=round(stat.st_size / 1024, 1),
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            extension=p.suffix.lower(),
            category=cat,
        )


# ── Vision description (what GPT-4o sees) ────────────────────────────────────

class ScreenshotDescription(BaseModel):
    path: str
    filename: str
    description: str                    # rich natural language description
    suggested_category: Category
    suggested_name: str                 # human-readable filename without extension
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# ── Index entry (stored in ChromaDB metadata) ─────────────────────────────────

class IndexEntry(BaseModel):
    path: str
    filename: str
    description: str
    category: str
    tags: list[str]
    indexed_at: str                     # ISO string
    size_kb: float
    modified: str


# ── Search result ─────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    path: str
    filename: str
    category: str
    description: str
    tags: list[str]
    similarity: float                   # 0–1, higher is more relevant
    modified: str


# ── Tool responses (what MCP returns as JSON) ──────────────────────────────────

class ScanResult(BaseModel):
    root: str
    total_images: int
    uncategorized: int
    files: list[ScreenshotMeta]


class MoveResult(BaseModel):
    status: str
    from_path: str
    to_path: str


class RenameResult(BaseModel):
    status: str
    from_name: str
    to_name: str
    path: str


class IndexResult(BaseModel):
    status: str
    path: str
    description: str
    category: str
    tags: list[str]


class CategorySummary(BaseModel):
    root: str
    categories: dict[str, int]
    total_indexed: int
    total_unindexed: int