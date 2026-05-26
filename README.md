# 🧠 screenshot-brain

> Your Mac screenshots folder is a disaster. This fixes it — with vision, embeddings, and semantic search.

An AI agent + custom MCP server that **sees** your screenshots, understands what's in them, organizes them into clean folders, and lets you find anything with natural language.

**Fully local. No cloud uploads. No subscriptions.**

---

## How it works

```
screenshot.png
      │
      ▼
GPT-4o vision         "React error about invalid hook call in App.js, line 23"
      │
      ▼
text-embedding-3-small  [0.021, -0.847, 0.334, ...]
      │
      ▼
ChromaDB (local)      stored permanently on disk
      │
      ▼
search("hooks problem") → finds it, even with a garbage filename
```

---

## The result

```
~/Screenshots/
├── code/
│   ├── errors/         react-hooks-invalid-call.png
│   │                   python-importerror-requests.png
│   └── snippets/       vim-config-lsp-setup.png
├── design/
│   ├── figma/          darkmode-mobile-v3.png
│   └── other/          canva-instagram-post.png
├── chats/              whatsapp-trip-planning-march.png
├── docs/               notion-q2-sprint-board.png
├── memes/              drake-hotline-bling-coding.png
└── web/                vercel-deployment-dashboard.png
```

---

## Architecture

```
agent.py  (OpenAI Agents SDK)
    │
    │   MCPServerStdio
    ▼
src/screenshot_brain/
    ├── server.py        MCP server — 8 tools
    ├── vision.py        GPT-4o vision → structured description
    ├── embeddings.py    OpenAI embeddings + ChromaDB
    └── models.py        Pydantic data models
```

**Why MCP and not just a script?**

A script renames blindly. This agent:
- Understands context across a conversation
- Lets you say "actually split design into figma vs other" and re-organizes
- Finds screenshots by meaning: "that slack message about the deploy" works
- Remembers decisions across sessions (ChromaDB persists to `~/.screenshot-brain/`)

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/yourname/screenshot-brain
cd screenshot-brain
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Set your screenshots folder

```bash
export SCREENSHOTS_DIR="/Users/yourname/Desktop"
# or edit agent.py line 22
```

### 4. Run

```bash
# Full auto-organize (scan → vision → embed → move → rename)
python agent.py

# Semantic search
python agent.py "find my react error screenshots"
python agent.py "show me figma mockups from last month"
python agent.py "that slack conversation about the deployment"

# Partial organize
python agent.py "organize only the code screenshots"

# Index health
python agent.py --stats
```

---

## MCP Tools

| Tool | What it does |
|------|-------------|
| `scan_screenshots` | List all images + metadata |
| `read_screenshot` | Return base64 image for direct visual inspection |
| `describe_and_index` | **Main tool** — GPT-4o vision + ChromaDB embedding |
| `search_screenshots` | Semantic search (meaning, not filenames) |
| `move_screenshot` | Move to category subfolder, updates index |
| `rename_screenshot` | Human-readable rename, updates index |
| `list_categories` | Folder structure + counts |
| `index_stats` | ChromaDB health check |

---

## Use in Claude Desktop (MCP server only)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "screenshot-brain": {
      "command": "python",
      "args": ["/absolute/path/to/screenshot-brain/src/screenshot_brain/server.py"],
      "env": {
        "SCREENSHOTS_DIR": "/Users/yourname/Desktop",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## Where data lives

| What | Where |
|------|-------|
| ChromaDB index | `~/.screenshot-brain/chroma/` |
| Your screenshots | wherever `SCREENSHOTS_DIR` points |
| Nothing else | nothing is uploaded anywhere |

---

## Stack

- **MCP server** — [`mcp`](https://github.com/anthropics/mcp) Python SDK v1.27+
- **Agent** — [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) v0.17+
- **Vision** — `gpt-4o` (low detail mode — fast + cheap)
- **Embeddings** — `text-embedding-3-small`
- **Vector DB** — [ChromaDB](https://www.trychroma.com/) (local, persistent)
- **Validation** — Pydantic v2

---

## License

MIT