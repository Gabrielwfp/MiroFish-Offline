<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline

**Fully local fork of [MiroFish](https://github.com/666ghj/MiroFish) — no cloud APIs required. English UI.**

*A multi-agent swarm intelligence engine that simulates public opinion, market sentiment, and social dynamics. Entirely on your hardware.*

[![GitHub Stars](https://img.shields.io/github/stars/nikmcfly/MiroFish-Offline?style=flat-square&color=DAA520)](https://github.com/nikmcfly/MiroFish-Offline/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/nikmcfly/MiroFish-Offline?style=flat-square)](https://github.com/nikmcfly/MiroFish-Offline/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates hundreds of AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

The [original MiroFish](https://github.com/666ghj/MiroFish) was built for the Chinese market (Chinese UI, Zep Cloud for knowledge graphs, DashScope API). This fork makes it **fully local and fully English**:

| Original MiroFish | MiroFish-Offline |
|---|---|
| Chinese UI | **English UI** (1,000+ strings translated) |
| Zep Cloud (graph memory) | **Neo4j Community Edition 5.18+** |
| DashScope / OpenAI API (LLM) | **Ollama** (GGUF or MLX models) |
| Zep Cloud embeddings | **qwen3-embedding:0.6b** via Ollama (1024d) |
| Cloud API keys required | **Zero cloud dependencies** |

## Workflow

1. **Graph Build** — Extracts entities (people, companies, events) and relationships from your document. Builds a knowledge graph with individual and group memory via Neo4j.
2. **Env Setup** — Generates hundreds of agent personas, each with unique personality, opinion bias, reaction speed, influence level, and memory of past events.
3. **Simulation** — Agents interact on simulated social platforms: posting, replying, arguing, shifting opinions. The system tracks sentiment evolution, topic propagation, and influence dynamics in real time.
4. **Report** — A ReportAgent analyzes the post-simulation environment, interviews a focus group of agents, searches the knowledge graph for evidence, and generates a structured analysis.
5. **Interaction** — Chat with any agent from the simulated world. Ask them why they posted what they posted. Full memory and personality persists.

## Screenshot

<div align="center">
<img src="./static/image/mirofish-offline-screenshot.jpg" alt="MiroFish Offline — English UI" width="100%"/>
</div>

## Quick Start

### Prerequisites

- **Ollama running natively on the host** (macOS app or `ollama serve`, port 11434), version 0.40+ (JSON-schema outputs). It is not a compose service.
- Neo4j **5.18+** (relationship vector search). `docker compose` provides it.
- Docker & Docker Compose (recommended), **or** Python **3.11 exactly** (`camel-oasis` requires `<3.12`), [uv](https://docs.astral.sh/uv/) and Node.js 18+.

### 1. Pull the models

```bash
ollama pull qwen3-embedding:0.6b          # embeddings (1024d)
ollama pull ornith-1.5:9b                 # chat / NER / agents (GGUF), or any other chat model

# Optional but recommended for the simulation: a derived model with a fixed context window
printf 'FROM ornith-1.5:9b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create ornith-1.5:9b-mirofish-ctx32k -f Modelfile
```

### 2. Configure

```bash
cp .env.example .env     # then adjust models and URLs (see Configuration)
```

### Option A: Docker

```bash
docker compose up -d     # Neo4j + the MiroFish app
```

Open `http://localhost:3000`.

Inside the app container `localhost` is the container itself. `docker-compose.yml` therefore overrides `NEO4J_URI` (`bolt://neo4j:7687`) and `EMBEDDING_BASE_URL` (`http://host.docker.internal:11434/v1`). `LLM_BASE_URL` and `OPENAI_API_BASE_URL` still come from your `.env`: use `http://host.docker.internal:11434/v1` (or a LAN address), never `localhost`.

### Option B: Manual

```bash
docker compose up -d neo4j          # or any Neo4j >= 5.18
npm run setup:all                   # root + frontend deps, and `uv sync` in backend
npm run dev                         # backend (Flask :5001) + frontend (Vite :3000)
```

Open `http://localhost:3000`. The backend must run on Python 3.11 (`backend/.python-version`); pass `--python 3.11` to `uv` if your default is newer.

### Tests

```bash
cd backend && uv run --python 3.11 --with pytest python -m pytest tests -q
```

Mocked, no network or Ollama required.

## Configuration

All settings are in a single root `.env` (copy from `.env.example`, which documents every variable):

```bash
# LLM (chat) — Ollama's OpenAI-compatible API. The key is ignored but must be non-empty
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1        # Docker: http://host.docker.internal:11434/v1
LLM_MODEL_NAME=ornith-1.5:9b-mirofish-ctx32k

# Neo4j
NEO4J_URI=bolt://localhost:7687               # Docker: overridden by docker-compose.yml
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish

# Embeddings (base URL must include /v1; independent from the chat settings)
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_DIMENSIONS=1024                     # sizes the Neo4j vector indexes
```

Works with any OpenAI-compatible API by changing `LLM_BASE_URL` and `LLM_API_KEY`. Changing `EMBEDDING_DIMENSIONS` on an existing database requires dropping the `entity_embedding` / `fact_embedding` indexes and re-embedding.

### Context window (`num_ctx`)

| `LLM_API_MODE` | Transport | `num_ctx` |
|---|---|---|
| `openai` (default) | `/v1` | Not sent; Ollama uses its server default (`OLLAMA_CONTEXT_LENGTH`) |
| `ollama_native` | `/api/chat` | `OLLAMA_NUM_CTX_DEFAULT` (32768) or `OLLAMA_NUM_CTX_{ONTOLOGY,NER,PROFILE,SIM_CONFIG,REPORT,CHAT}` per operation |

- Every request is checked before sending (estimated input + output reserve ≤ `num_ctx`). A request that does not fit fails with a visible `ContextOverflowError`; nothing is truncated silently.
- `num_ctx` is honored only by Ollama's llama.cpp runner (GGUF models). The MLX runner (e.g. `qwen3.8:*-nvfp4/mxfp8`) ignores it.
- The simulation (CAMEL/OASIS) always uses `/v1` and cannot send `num_ctx`. Set `SIMULATION_LLM_MODEL_NAME` to the derived model above so it loads at 32768 instead of the server default. With a slow thinking model, set `SIMULATION_DISABLE_THINKING=true`.
- Validation results, memory measurements and how to switch back to the OpenAI mode: [docs/ollama-integration-report.md](./docs/ollama-integration-report.md).

## Architecture

This fork introduces a clean abstraction layer between the application and the graph database:

```
┌─────────────────────────────────────────┐
│              Flask API                   │
│  graph.py  simulation.py  report.py     │
└──────────────┬──────────────────────────┘
               │ app.extensions['neo4j_storage']
┌──────────────▼──────────────────────────┐
│           Service Layer                  │
│  EntityReader  GraphToolsService         │
│  GraphMemoryUpdater  ReportAgent         │
└──────────────┬──────────────────────────┘
               │ storage: GraphStorage
┌──────────────▼──────────────────────────┐
│         GraphStorage (abstract)          │
│              │                            │
│    ┌─────────▼─────────┐                │
│    │   Neo4jStorage     │                │
│    │  ┌───────────────┐ │                │
│    │  │ EmbeddingService│ ← Ollama       │
│    │  │ NERExtractor   │ ← Ollama LLM   │
│    │  │ SearchService  │ ← Hybrid search │
│    │  └───────────────┘ │                │
│    └───────────────────┘                │
└─────────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │  Neo4j CE   │
        │  5.18+      │
        └─────────────┘
```

**Key design decisions:**

- `GraphStorage` is an abstract interface — swap Neo4j for any other graph DB by implementing one class
- Dependency injection via Flask `app.extensions` — no global singletons
- Hybrid search: 0.7 × vector similarity + 0.3 × BM25 keyword search
- Synchronous NER/RE extraction via local LLM (replaces Zep's async episodes), with a JSON-schema response format built from the ontology and validated with Pydantic
- No silent failures: embedding, NER and context errors fail the build task with the cause
- All original dataclasses and LLM tools (InsightForge, Panorama, Agent Interviews) preserved

## Hardware Requirements

Measured on an Apple M4 Max (64 GB) with Ollama 0.40.17:

| Setup | GPU memory |
|---|---|
| `ornith-1.5:9b` (GGUF) at 131072 context (server default) | ~19.4 GB |
| `ornith-1.5:9b-mirofish-ctx32k` at 32768 | ~9.9 GB |
| Full pipeline with that model (incl. embeddings) | 9.7–13.6 GB |
| `qwen3.8:27b` (MLX), full pipeline | 30–43 GB |

CPU-only mode works but is much slower. Pick a smaller chat model if memory is tight.

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Trading signal generation** — feed financial news and observe simulated market sentiment
- **Policy impact analysis** — test draft regulations against simulated public response
- **Creative experiments** — someone fed it a classical Chinese novel with a lost ending; the agents wrote a narratively consistent conclusion

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits & Attribution

This is a modified fork of [MiroFish](https://github.com/666ghj/MiroFish) by [666ghj](https://github.com/666ghj), originally supported by [Shanda Group](https://www.shanda.com/). The simulation engine is powered by [OASIS](https://github.com/camel-ai/oasis) from the CAMEL-AI team.

**Modifications in this fork:**
- Backend migrated from Zep Cloud to local Neo4j CE 5.18+ + Ollama (chat, embeddings and structured NER)
- Per-operation context window control (`ollama_native`), request preflight and visible failures
- Entire frontend translated from Chinese to English (20 files, 1,000+ strings)
- All Zep references replaced with Neo4j across the UI
- Rebranded to MiroFish Offline
