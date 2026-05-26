"""
server.py — screenshot-brain MCP Server
-----------------------------------------
Production MCP server exposing 8 tools:

  scan_screenshots      → list files + metadata
  read_screenshot       → base64 image for agent vision
  describe_and_index    → GPT-4o vision + embed into ChromaDB  ← the magic tool
  search_screenshots    → semantic search via embeddings
  move_screenshot       → move to category subfolder
  rename_screenshot     → human-readable rename
  list_categories       → folder structure + counts
  index_stats           → ChromaDB index health

Architecture:
  MCP tool call
    → vision.describe_screenshot()   [GPT-4o, sees the image]
    → embeddings.ScreenshotIndex.index()  [OpenAI embeddings + ChromaDB]
    → filesystem action (move/rename)

The index stays in sync with the filesystem automatically.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openai import AsyncOpenAI

from embeddings import ScreenshotIndex
from models import Category, ScreenshotMeta, ScanResult
from vision import describe_screenshot
from dotenv import load_dotenv

load_dotenv()
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("screenshot-brain.server")

# ── Config from env ───────────────────────────────────────────────────────────
DEFAULT_ROOT = os.path.expanduser(
    os.getenv("SCREENSHOTS_DIR", "~/Desktop/test-screenshot")
)

# ── Singletons (created once, reused across all tool calls) ───────────────────
_openai_client: AsyncOpenAI | None = None
_index: ScreenshotIndex | None = None


def get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_index() -> ScreenshotIndex:
    global _index
    if _index is None:
        _index = ScreenshotIndex(openai_client=get_openai())
    return _index


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def get_root(folder: str | None = None) -> Path:
    root = Path(folder).expanduser() if folder else Path(DEFAULT_ROOT).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ok(data: dict | list) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


# ── Server ────────────────────────────────────────────────────────────────────

app = Server("screenshot-brain")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="scan_screenshots",
            description=(
                "Scan a folder for screenshots and return metadata for every image file. "
                "Returns filename, path, size, modified date, and current category. "
                "Call this first to understand what exists before deciding what to do."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Absolute path to screenshots folder. Uses default if omitted."},
                    "recursive": {"type": "boolean", "description": "Also scan subfolders. Default false.", "default": False},
                    "uncategorized_only": {"type": "boolean", "description": "Only return files in the root (not yet organized). Default false.", "default": False},
                },
            },
        ),
        Tool(
            name="read_screenshot",
            description=(
                "Read a single screenshot and return it as a base64-encoded image "
                "so you can visually inspect its contents. "
                "Use this when you need to directly see an image before deciding how to handle it."
            ),
            inputSchema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the image file."},
                },
            },
        ),
        Tool(
            name="describe_and_index",
            description=(
                "THE MAIN ORGANIZE TOOL. "
                "Uses GPT-4o vision to analyze a screenshot, generate a rich description and tags, "
                "suggest a category and human-readable name, then stores the embedding in ChromaDB "
                "for semantic search. "
                "Returns the suggested category and name so you can then call move_screenshot and rename_screenshot. "
                "Always call this before moving/renaming a file."
            ),
            inputSchema={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the screenshot."},
                    "size_kb": {"type": "number", "description": "File size in KB (from scan_screenshots). Used for index metadata."},
                    "modified": {"type": "string", "description": "ISO modified date (from scan_screenshots)."},
                },
            },
        ),
        Tool(
            name="search_screenshots",
            description=(
                "Semantic search over all indexed screenshots using natural language. "
                "This is NOT a filename search — it understands meaning. "
                "E.g. 'react hooks error' will find a screenshot even if the filename is 'Screenshot 2024-01-14 at 3.22.png'. "
                "Returns ranked results with similarity scores."
            ),
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query. E.g. 'dark mode figma mockup' or 'python import error'."},
                    "n_results": {"type": "integer", "description": "Max results to return. Default 10.", "default": 10},
                    "category_filter": {"type": "string", "description": "Optional: restrict search to a specific category. E.g. 'code/errors'."},
                },
            },
        ),
        Tool(
            name="move_screenshot",
            description=(
                "Move a screenshot file into a category subfolder inside the screenshots root. "
                "Also updates the path in the search index automatically. "
                "Create nested paths like 'code/errors' or 'design/figma'."
            ),
            inputSchema={
                "type": "object",
                "required": ["path", "category"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the image file."},
                    "category": {"type": "string", "description": "Subfolder relative to screenshots root. E.g. 'code/errors' or 'chats'."},
                    "root": {"type": "string", "description": "Screenshots root. Uses default if omitted."},
                },
            },
        ),
        Tool(
            name="rename_screenshot",
            description=(
                "Rename a screenshot to a human-readable name. "
                "Also updates the filename in the search index. "
                "Good: 'react-hooks-error-jan24.png'. Bad: 'Screenshot 2024-01-14.png'."
            ),
            inputSchema={
                "type": "object",
                "required": ["path", "new_name"],
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the image file."},
                    "new_name": {"type": "string", "description": "New filename WITH extension. E.g. 'figma-darkmode-v2.png'."},
                },
            },
        ),
        Tool(
            name="list_categories",
            description=(
                "List all category subfolders inside the screenshots root with image counts. "
                "Also shows how many files are unindexed (not yet processed by describe_and_index)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Screenshots root. Uses default if omitted."},
                },
            },
        ),
        Tool(
            name="index_stats",
            description=(
                "Show ChromaDB index health: total screenshots indexed, breakdown by category, "
                "and the database path. Use this to understand the state of the semantic search index."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Tool implementations ──────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict):

    # ── scan_screenshots ──────────────────────────────────────────────────────
    if name == "scan_screenshots":
        root = get_root(arguments.get("folder"))
        recursive = arguments.get("recursive", False)
        uncategorized_only = arguments.get("uncategorized_only", False)

        pattern = "**/*" if recursive else "*"
        files = []
        for p in sorted(root.glob(pattern)):
            if not (p.is_file() and is_image(p)):
                continue
            if uncategorized_only:
                # Only root-level files (not in any subfolder)
                if p.parent != root:
                    continue
            files.append(ScreenshotMeta.from_path(p, root).model_dump())

        result = ScanResult(
            root=str(root),
            total_images=len(files),
            uncategorized=sum(1 for f in files if f["category"] is None),
            files=files,
        )
        return ok(result.model_dump())

    # ── read_screenshot ───────────────────────────────────────────────────────
    elif name == "read_screenshot":
        p = Path(arguments["path"]).expanduser()
        if not p.exists():
            return err(f"File not found: {p}")
        if not is_image(p):
            return err(f"Not a supported image: {p.suffix}")

        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode()
        ext = p.suffix.lower().lstrip(".")
        mime = "jpeg" if ext == "jpg" else ext

        return ok({
            "filename": p.name,
            "path": str(p),
            "size_kb": round(len(raw) / 1024, 1),
            "image_data": f"data:image/{mime};base64,{b64}",
        })

    # ── describe_and_index ────────────────────────────────────────────────────
    elif name == "describe_and_index":
        p = Path(arguments["path"]).expanduser()
        if not p.exists():
            return err(f"File not found: {p}")
        if not is_image(p):
            return err(f"Not a supported image: {p.suffix}")

        try:
            desc = await describe_screenshot(p, client=get_openai())
        except Exception as e:
            logger.error("Vision failed for %s: %s", p.name, e)
            return err(f"Vision API failed: {e}")

        try:
            size_kb = arguments.get("size_kb") or round(p.stat().st_size / 1024, 1)
            modified = arguments.get("modified", "")
            await get_index().index(desc, size_kb=size_kb, modified=modified)
        except Exception as e:
            logger.error("Indexing failed for %s: %s", p.name, e)
            return err(f"Indexing failed: {e}")

        return ok({
            "status": "indexed",
            "path": desc.path,
            "filename": desc.filename,
            "description": desc.description,
            "suggested_category": desc.suggested_category.value,
            "suggested_name": f"{desc.suggested_name}{p.suffix.lower()}",
            "tags": desc.tags,
            "confidence": desc.confidence,
        })

    # ── search_screenshots ────────────────────────────────────────────────────
    elif name == "search_screenshots":
        query = arguments.get("query", "").strip()
        if not query:
            return err("query is required")

        n = arguments.get("n_results", 10)
        category_filter = arguments.get("category_filter")

        try:
            results = await get_index().search(query, n_results=n, category_filter=category_filter)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return err(f"Search failed: {e}")

        return ok({
            "query": query,
            "total_results": len(results),
            "results": [r.model_dump() for r in results],
        })

    # ── move_screenshot ───────────────────────────────────────────────────────
    elif name == "move_screenshot":
        src = Path(arguments["path"]).expanduser()
        if not src.exists():
            return err(f"File not found: {src}")

        root = get_root(arguments.get("root"))
        dest_dir = root / arguments["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        # Avoid silent overwrites
        if dest.exists() and dest != src:
            stem, suffix = src.stem, src.suffix
            i = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{i}{suffix}"
                i += 1

        old_path = str(src)
        shutil.move(str(src), str(dest))

        # Keep index in sync
        get_index().update_path(old_path, str(dest), dest.name)

        return ok({"status": "moved", "from": old_path, "to": str(dest)})

    # ── rename_screenshot ─────────────────────────────────────────────────────
    elif name == "rename_screenshot":
        src = Path(arguments["path"]).expanduser()
        if not src.exists():
            return err(f"File not found: {src}")

        new_name = arguments["new_name"]
        dest = src.parent / new_name
        if dest.exists() and dest != src:
            return err(f"File already exists: {new_name}")

        old_path = str(src)
        src.rename(dest)

        # Keep index in sync
        get_index().update_path(old_path, str(dest), new_name)

        return ok({"status": "renamed", "from": src.name, "to": new_name, "path": str(dest)})

    # ── list_categories ───────────────────────────────────────────────────────
    elif name == "list_categories":
        root = get_root(arguments.get("root"))
        categories: dict[str, int] = {}

        root_images = [p for p in root.iterdir() if p.is_file() and is_image(p)]
        if root_images:
            categories["uncategorized"] = len(root_images)

        for d in sorted(root.rglob("*")):
            if d.is_dir():
                count = len([p for p in d.rglob("*") if p.is_file() and is_image(p)])
                if count:
                    categories[str(d.relative_to(root))] = count

        stats = get_index().stats()
        return ok({
            "root": str(root),
            "folders": categories,
            "total_images": sum(categories.values()),
            "total_indexed_in_db": stats["total_indexed"],
            "unindexed": sum(categories.values()) - stats["total_indexed"],
        })

    # ── index_stats ───────────────────────────────────────────────────────────
    elif name == "index_stats":
        return ok(get_index().stats())

    else:
        return err(f"Unknown tool: {name}")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    logger.info("screenshot-brain MCP server starting...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())