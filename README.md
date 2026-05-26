<div align="center">
  <h1>
    __                                           _     
    / _\_ __   __ _ _ __  ___  ___  __ _ _ __ ___| |__  
    \ \| '_ \ / _` | '_ \/ __|/ _ \/ _` | '__/ __| '_ \ 
    _\ \ | | | (_| | |_) \__ \  __/ (_| | | | (__| | | |
    \__/_| |_|\__,_| .__/|___/\___|\__,_|_|  \___|_| |_|
                |_|                                   </h1>
  <p><strong>Your Mac screenshots folder is a disaster. This fixes it — with vision, embeddings, and semantic search.</strong></p>

  <p>
    <a href="https://pypi.org/project/snapsearch/"><img src="https://img.shields.io/pypi/v/snapsearch.svg" alt="PyPI Version"></a>
    <a href="https://pypi.org/project/snapsearch/"><img src="https://img.shields.io/pypi/pyversions/snapsearch.svg" alt="Python Versions"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  </p>
</div>

An AI agent + custom MCP server that **sees** your screenshots, understands what's in them, organizes them into clean folders, and lets you find anything with natural language.

**Fully local. No cloud uploads. No subscriptions.**

---

## ✨ Features

- **Context-Aware Organization**: Unlike scripts that blindly rename files, this agent understands context and categorizes intelligently.
- **Semantic Search**: Find screenshots by meaning (e.g., *"that slack message about the deploy"*), even with garbage filenames.
- **Adaptive**: Tell it to *"actually split design into figma vs other"* and it re-organizes on the fly.
- **Persistent Memory**: Decisions are remembered across sessions using a local ChromaDB.

---

## 🚀 Quickstart (No Clone Needed!)

Using `uvx` (the modern Python alternative to `npx`), you can run **snapsearch** instantly without manual cloning or pip installs.

### 1. Set your Environment Variables

You need to provide your OpenAI API key and optionally your screenshots directory. You can set them in your environment:

```bash
export OPENAI_API_KEY="sk-..."
export SCREENSHOTS_DIR="/Users/yourname/Desktop" # Optional, defaults to ~/Desktop/screenshots-demo
```

### 2. Run the Agent

Trigger the autonomous organizer right away with these commands:

```bash
# Full auto-organize (scan → vision → embed → move → rename)
uvx snapsearch

# Semantic search queries
uvx snapsearch "find my react error screenshots"
uvx snapsearch "show me figma mockups from last month"
uvx snapsearch "that slack conversation about the deployment"

# Partial organize
uvx snapsearch "organize only the code screenshots"

# Index health check
uvx snapsearch --stats
```

---

## 🤖 How it works

Behind the scenes, **snapsearch** builds a powerful pipeline to understand your images:

```text
screenshot.png
      │
      ▼
GPT-4o vision         "React error about invalid hook call in App.js, line 23"
      │
      ▼
text-embedding-3-small  [0.021, -0.847, 0.334, ...]
      │
      ▼
ChromaDB (local)      Stored permanently on disk
      │
      ▼
search("hooks problem") → Finds it instantly!
```

### The Resulting Organization

Your chaotic screenshots folder turns into a clean, categorized structure:

```text
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

## 🛠️ Architecture

```text
snapsearch (Agent)
    │
    │   MCPServerStdio
    ▼
snapsearch-mcp (MCP server — 8 tools)
    ├── vision.py        GPT-4o vision → structured description
    ├── embeddings.py    OpenAI embeddings + ChromaDB
    └── models.py        Pydantic data models
```

### MCP Tools Available

| Tool | Description |
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

## 🔌 Use in Claude Desktop (MCP server only)

Because it's distributed on PyPI, using it in Claude Desktop is extremely simple.

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "snapsearch": {
      "command": "uvx",
      "args": ["snapsearch-mcp"],
      "env": {
        "SCREENSHOTS_DIR": "/Users/yourname/Desktop",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## 🔒 Privacy & Data

| Component | Location |
|------|-------|
| **ChromaDB index** | `~/.snapsearch/chroma/` |
| **Your screenshots** | wherever `SCREENSHOTS_DIR` points |
| **Cloud Uploads** | **None.** Nothing is uploaded anywhere. |

---

## 🏗️ Stack

- **MCP server** — [`mcp`](https://github.com/anthropics/mcp) Python SDK v1.27+
- **Agent** — [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) v0.17+
- **Vision** — `gpt-4o` (low detail mode — fast + cheap)
- **Embeddings** — `text-embedding-3-small`
- **Vector DB** — [ChromaDB](https://www.trychroma.com/) (local, persistent)
- **Validation** — Pydantic v2

---

## 📄 License

MIT