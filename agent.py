"""
agent.py — screenshot-brain Agent
-----------------------------------
Autonomous screenshot organizer using OpenAI Agents SDK + custom MCP server.

Usage:
    python agent.py                                    # full auto-organize
    python agent.py "find my react errors"             # semantic search
    python agent.py "organize only design screenshots" # partial organize
    python agent.py --stats                            # index health check
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
print("DIR:", os.getenv("SCREENSHOTS_DIR"))
print("Files:", list(Path(os.getenv("SCREENSHOTS_DIR", "")).expanduser().glob("*")))
SCREENSHOTS_DIR: str = os.getenv(
    "SCREENSHOTS_DIR",
    os.path.expanduser("~/Desktop/screenshots-demo"),  # ← change this
)

SERVER_PATH: str = str(Path(__file__).parent / "server.py")

# ── Agent system prompt ───────────────────────────────────────────────────────

INSTRUCTIONS = """
You are screenshot-brain, an autonomous AI agent that organizes Mac screenshots.

## Your tools
- scan_screenshots      → see all files + metadata
- read_screenshot       → base64 image (use only when you need to directly see something)
- describe_and_index    → THE MAIN TOOL: GPT-4o vision + embed into ChromaDB
- search_screenshots    → semantic search (understands meaning, not just filenames)
- move_screenshot       → move to category subfolder
- rename_screenshot     → give a human-readable name
- list_categories       → current folder structure
- index_stats           → ChromaDB index health

## Organize workflow (for each screenshot)
1. Call describe_and_index(path) — this does vision + embedding in one shot
2. Use the returned suggested_category and suggested_name
3. Call move_screenshot(path, category)  
4. Call rename_screenshot(new_path, suggested_name)

## After organizing
Always call list_categories and give a clean summary:
- How many files processed
- Breakdown by category
- Any interesting patterns you noticed (e.g. "you have 40 error screenshots from January")

## Search workflow
When asked to find something:
1. Call search_screenshots(query) — this is semantic, not filename matching
2. Return the top results with their paths and descriptions
3. If the user wants to open/view one, call read_screenshot

## Rules
- Be decisive — don't ask for confirmation on each file
- Process ALL uncategorized files, not just a sample
- If confidence < 0.5 from describe_and_index, use 'misc' category
- Keep rename suggestions under 40 chars, lowercase, hyphens only
- Always keep the original file extension
- If a file fails vision (network error etc), skip it and continue — never stop
"""

# ── Task templates ────────────────────────────────────────────────────────────

def build_prompt(task: str | None, screenshots_dir: str) -> str:
    if task and task.startswith("--stats"):
        return f"""
        Call index_stats and list_categories for: {screenshots_dir}
        Give a clean summary of the index health and folder structure.
        """

    if task is None:
        return f"""
        Fully organize the screenshots folder: {screenshots_dir}

        Steps:
        1. scan_screenshots to see all files
        2. list_categories to understand current state
        3. For each uncategorized screenshot:
           a. describe_and_index(path)
           b. move_screenshot(path, suggested_category)
           c. rename_screenshot(new_path, suggested_name)
        4. Final summary: files processed, categories created, anything interesting

        Be thorough — process EVERY uncategorized file.
        """

    return f"""
    Screenshots folder: {screenshots_dir}
    
    User request: {task}
    
    Guidelines:
    - For search requests: use search_screenshots (semantic search, not filename matching)
    - For organize requests: describe_and_index → move → rename
    - For stats: index_stats + list_categories
    - Always end with a helpful summary
    """


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    prompt = build_prompt(task, SCREENSHOTS_DIR)

    server_params = {
        "command": "python",
        "args": [SERVER_PATH],
        "env": {
            **os.environ,
            "SCREENSHOTS_DIR": SCREENSHOTS_DIR,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        },
    }

    print("🧠 screenshot-brain")
    print(f"📁 {SCREENSHOTS_DIR}")
    print(f"🎯 {'Auto-organize' if task is None else task}")
    print("─" * 50)

    async with MCPServerStdio(
        params=server_params,
        client_session_timeout_seconds=120,  # vision calls can be slow
    ) as mcp_server:

        agent = Agent(
            name="screenshot_brain",
            instructions=INSTRUCTIONS,
            model="gpt-4o",              # needs vision for read_screenshot fallback
            mcp_servers=[mcp_server],
        )

        with trace("screenshot-brain"):
            result = await Runner.run(
                agent,
                prompt,
                max_turns=100,           # large folders need many turns
            )

    print("\n" + "═" * 50)
    print("✅ Done")
    print("═" * 50)
    print(result.final_output)


def sync_main() -> None:
    """Entry point for pyproject.toml script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())