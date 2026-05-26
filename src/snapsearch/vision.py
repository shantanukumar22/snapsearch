"""
vision.py
---------
Uses GPT-4o vision to generate rich descriptions of screenshots.
This is the "understanding" layer — turns pixels into meaning.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from .models import Category, ScreenshotDescription

logger = logging.getLogger(__name__)

# ── Supported image types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# ── Vision prompt ─────────────────────────────────────────────────────────────
VISION_SYSTEM_PROMPT = """
You are a screenshot analysis expert. Your job is to look at a screenshot and return a
structured JSON description of what you see.

Return ONLY valid JSON with this exact structure (no markdown, no preamble):
{
  "description": "A detailed description of what this screenshot shows",
  "suggested_category": "<one of the valid categories>",
  "suggested_name": "short-descriptive-name-no-extension",
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.95
}

Valid categories:
- "code/errors"     → error messages, stack traces, terminal crashes, exception tracebacks
- "code/snippets"   → code editors, VS Code, Vim, IDE screenshots, terminal commands
- "design/figma"    → Figma UI, wireframes, mockups in Figma
- "design/other"    → Sketch, Adobe XD, Canva, other design tools
- "chats"           → iMessage, WhatsApp, Slack, Discord, Telegram, Teams conversations
- "docs"            → Documents, PDFs, Notion, Google Docs, Word, Confluence
- "web"             → Websites, articles, browser screenshots, dashboards
- "memes"           → Memes, funny images, screenshots from social media for fun
- "wallpapers"      → Aesthetic images, wallpapers, backgrounds
- "misc"            → Anything that doesn't fit the above

For suggested_name:
- lowercase, hyphens only, no spaces, no extension
- max 40 characters
- be specific: "react-hooks-error-useeffect" not just "error"
- include context if visible: tool name, error type, topic

For tags:
- 3-6 short lowercase tags
- include: tool/app name, content type, topic/domain
- example: ["figma", "dark-mode", "mobile", "ui-mockup"]

For confidence:
- 1.0 if you're sure what it is
- 0.7 if content is partially visible or ambiguous
- 0.4 if you genuinely can't tell
"""


def _encode_image(path: Path) -> tuple[str, str]:
    """Read image file and return (base64_data, mime_type)."""
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/png")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, mime


async def describe_screenshot(
    path: Path,
    client: AsyncOpenAI | None = None,
) -> ScreenshotDescription:
    """
    Use GPT-4o vision to generate a structured description of a screenshot.

    Args:
        path:   Path to the image file.
        client: AsyncOpenAI client (creates one if not provided).

    Returns:
        ScreenshotDescription with category, name, tags, and description.

    Raises:
        ValueError: If the file is not a supported image type.
        RuntimeError: If the vision API call fails.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix}")

    if client is None:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    b64, mime = _encode_image(path)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "low",   # cheaper + fast enough for categorization
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Analyze this screenshot. Filename hint: {path.name}",
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Vision API call failed for {path.name}: {e}") from e

    raw = response.choices[0].message.content or ""

    # Strip any accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Vision returned non-JSON for %s: %s", path.name, raw[:200])
        # Graceful fallback
        data = {
            "description": f"Screenshot: {path.name}",
            "suggested_category": "misc",
            "suggested_name": path.stem.lower().replace(" ", "-")[:40],
            "tags": [],
            "confidence": 0.3,
        }

    # Validate category — fall back to misc if model hallucinates
    raw_cat = data.get("suggested_category", "misc")
    try:
        category = Category(raw_cat)
    except ValueError:
        logger.warning("Unknown category '%s' for %s, using misc", raw_cat, path.name)
        category = Category.MISC

    return ScreenshotDescription(
        path=str(path),
        filename=path.name,
        description=data.get("description", ""),
        suggested_category=category,
        suggested_name=data.get("suggested_name", path.stem)[:40],
        tags=data.get("tags", []),
        confidence=float(data.get("confidence", 1.0)),
    )