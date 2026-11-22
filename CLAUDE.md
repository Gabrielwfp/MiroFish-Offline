# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MiroFish-Offline: a local-first fork of MiroFish. Upload a document → build a knowledge graph → generate agent personas → run an OASIS (CAMEL-AI) social-media simulation → generate a report / chat with agents. Zep Cloud and DashScope were replaced with **Neo4j Community + Ollama** (OpenAI-compatible API). UI and LLM output are English-only; much of the backend still has leftover Chinese comments/prompts from the upstream project — don't reintroduce Chinese output (see recent commit "force English-only output in report agent").

## Commands

```bash
npm run setup:all      # root npm deps + frontend deps + `uv sync` in backend
npm run dev            # backend (Flask :5001) + frontend (Vite :3000) via concurrently
npm run backend        # cd backend && uv run python run.py
npm run frontend       # cd frontend && npm run dev
npm run build          # frontend production build (vite build)
docker compose up -d   # neo4j (+ the mirofish app container, which runs `npm run dev`)
```

- Config lives in a single root `.env` (copy `.env.example`); `backend/app/config.py` loads it with `override=True`. `Config.validate()` in `run.py` requires `LLM_API_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD`. Vite proxies `/api` → `localhost:5001`.
- Backend needs **Python 3.11 exactly**: `camel-oasis==0.2.5` declares `requires-python <3.12` (camel-ai 0.2.78: `<3.13`), and `tiktoken 0.7.0` has no wheels for 3.13+, so a newer default interpreter tries a Rust source build and fails. `pyproject.toml` still says `>=3.11` with no upper bound — always pass `--python 3.11`. Deps are managed by `uv` (`backend/pyproject.toml` + `uv.lock`); `requirements.txt` is a parallel copy.
- The committed `backend/uv.lock` predates the fork's `pyproject.toml` (root still `mirofish-backend` 0.1.0, `neo4j` only transitive via camel-oasis, leftover `zep-cloud`), so `uv sync`/`uv run` rewrites it and `--frozen` fails. Re-locking with `requires-python = ">=3.11,<3.12"` changes no package versions (checked); not applied yet.
- Unit tests (mocked, no network): `cd backend && uv run --python 3.11 --with pytest python -m pytest tests -q` (until the lock is regenerated this also rewrites `uv.lock`; alternatively run pytest from any 3.11 env with the deps installed). They cover `EmbeddingService`, embedding config, the vector-index schema, the NER contract, `LLMClient` structured-output options and `add_text` failure propagation. CI only builds/pushes the Docker image on tags.
- Needs a running Neo4j **≥5.18** (relationship vector search; compose pins `5.18-community`, README still says 5.15) and **Ollama running natively on the host** (macOS app / `ollama serve`, port 11434; not a compose service). Chat and embeddings both go through Ollama's OpenAI-compatible `/v1` API; `LLM_API_KEY` just needs a non-empty value. Verified with Ollama 0.40.17, `qwen3.8:27b-mxfp8` (chat/NER) and `qwen3-embedding:0.6b`. The simulation scripts overwrite `OPENAI_API_KEY`/`OPENAI_API_BASE_URL` with `LLM_*` when set. From the app container use `host.docker.internal` (compose maps it via `host-gateway` for Linux); `localhost` there is the container itself.
- Embeddings: `EMBEDDING_*` settings are independent from the chat ones (`EMBEDDING_BASE_URL` must include `/v1`; `EMBEDDING_API_KEY` falls back to `LLM_API_KEY`). Defaults are `qwen3-embedding:0.6b` / `EMBEDDING_DIMENSIONS=1024`. The dimension sizes the Neo4j vector indexes and every response is checked against it. The indexes are created `IF NOT EXISTS` at `Neo4jStorage` startup, so changing the dimension on an existing database needs the old `entity_embedding`/`fact_embedding` indexes dropped and the vectors regenerated.
- **Structured LLM output:** plain `response_format: json_object` only guarantees *some* JSON. Models invent their own keys (`relationships`, `subject`/`object`), which used to make NER silently extract nothing. `LLMClient.chat_json(schema=...)` sends an OpenAI `json_schema` response format that Ollama enforces. `NERExtractor` builds that schema from the ontology (types as enums), validates the response with Pydantic (`NERResult`), retries a bounded number of times, then raises `NERExtractionError`. `disable_thinking=True` sends `reasoning_effort: "none"`, only to Ollama URLs. It is used for NER and ontology generation, because reasoning tokens count against `max_tokens` and truncated the ontology JSON. A truncated JSON response (`finish_reason == "length"`) raises an explicit error.
- **No silent failures:** `Neo4jStorage.add_text` propagates embedding and NER errors before writing anything for the chunk, so the build task ends `FAILED` with the cause. A build where every chunk succeeded but nothing matched the ontology ends `COMPLETED` with `result.warning` (a legitimately empty document is not an error). `graph_memory_updater` (simulation → graph) still catches, retries and counts its failures.
- **Context window (`num_ctx`).** `LLM_API_MODE` selects the transport in `LLMClient`:
  - `openai` (default): `/v1`, where Ollama ignores any `num_ctx`.
  - `ollama_native`: `/api/chat` with `options.num_ctx` from `Config.num_ctx_for(operation)` (`OLLAMA_NUM_CTX_DEFAULT` plus `OLLAMA_NUM_CTX_{ONTOLOGY,NER,PROFILE,SIM_CONFIG,REPORT,CHAT}`).
  - Every LLM call site passes an `operation`. `oasis_profile_generator` and `simulation_config_generator` go through `LLMClient.complete()`, not their own OpenAI clients.
  - A preflight rejects any request whose estimated input (chars / `CONTEXT_CHARS_PER_TOKEN`, default 3.5) plus `max_tokens` exceeds `num_ctx`, raising `ContextOverflowError`. Ollama's HTTP 400 "exceeds the available context size" maps to the same error. Generators re-raise it instead of using their rule-based fallbacks. Nothing is truncated silently.
  - **Only Ollama's llama.cpp runner (GGUF) honors `num_ctx`.** The MLX runner (safetensors, e.g. `qwen3.8:*-nvfp4/mxfp8`) ignores it on every API (ollama/ollama#18125, closed upstream as intended). On MLX, memory grows with the tokens actually processed.
- **Simulation (CAMEL)** always uses `/v1` and cannot send `num_ctx`. Point `SIMULATION_LLM_MODEL_NAME` (and ideally `LLM_MODEL_NAME`) at an Ollama model derived with `PARAMETER num_ctx`, such as `ornith-1.5:9b-mirofish-ctx32k`. Otherwise `/v1` loads the model at the server default (`OLLAMA_CONTEXT_LENGTH`, 131072 here), and a GGUF model reloads whenever the context changes. Keep NER, report and chat at the same `num_ctx` as that model, because they run during the simulation. `SIMULATION_DISABLE_THINKING=true` sends `reasoning_effort: "none"` from the agents through CAMEL's `model_config_dict`.
- **Thinking tokens count against `max_tokens`/`num_predict` on Ollama.** That is why NER, ontology, profiles, sim config and the interview summary pass `disable_thinking=True`, report sections reserve 8192 tokens, and the report loop retries on an empty answer.

## Architecture

**Pipeline stages** map 1:1 to frontend steps (`frontend/src/components/Step1GraphBuild … Step5Interaction.vue`, routed in `router/index.js`) and to backend blueprints in `backend/app/api/`: `graph_bp` (`/api/graph`: ontology generation, graph build, tasks), `simulation_bp` (`/api/simulation`: entities, prepare, start/stop, interviews — a 2.7k-line file), `report_bp` (`/api/report`).

**Storage abstraction (`backend/app/storage/`).** `GraphStorage` (ABC) is the only thing services depend on; `Neo4jStorage` implements it, composing `EmbeddingService` (Ollama embeddings), `NERExtractor` (LLM entity/relation extraction), and `SearchService` (hybrid search: 0.7 vector + 0.3 BM25). `neo4j_schema.py` holds the Cypher for constraints/vector/fulltext indexes. The single instance is created in `create_app()` and stored at `app.extensions['neo4j_storage']` (set to `None` on connection failure — endpoints fetch it via `current_app.extensions.get(...)` and must return 503 when missing). Services receive `storage` by constructor injection; avoid module-level singletons. NER/RE is **synchronous** (unlike Zep's async episodes), so `add_text_batch`/`wait_for_processing` keep Zep-shaped signatures but block.

**Service layer (`backend/app/services/`).** `ontology_generator` → `graph_builder` (chunking via `text_processor`, `utils/file_parser`) → `entity_reader` → `oasis_profile_generator` + `simulation_config_generator` → `simulation_manager` (state/files per simulation) → `simulation_runner` → `report_agent` (uses `graph_tools`: InsightForge / Panorama / agent interviews). `graph_memory_updater` writes simulation actions back into the graph. Long-running work (graph build, prepare, report generation) runs in background threads tracked by `models/task.py` `TaskManager`; the frontend polls `/…/status` endpoints. `models/project.py` persists projects as files under `backend/uploads/` (gitignored), as do simulations (`uploads/simulations/<sim_id>/`) and reports.

**Simulation runs out-of-process.** `SimulationRunner` launches `backend/scripts/run_parallel_simulation.py` (or the single-platform `run_twitter_/run_reddit_simulation.py`) with `subprocess.Popen`; these scripts drive OASIS for Twitter and Reddit in parallel and write `actions.jsonl` / `run_state.json` that the API reads. Flask ↔ script communication (agent interviews, close-env) is **filesystem IPC** in `services/simulation_ipc.py`: Flask writes into `commands/`, the script polls it and writes into `responses/`. `SimulationRunner.register_cleanup()` (called in `create_app`) terminates child processes on shutdown. Scripts import CAMEL/OASIS and read the same `.env`; they are not part of the Flask package, so changes to shared code need to work in both contexts.

**Frontend** is Vue 3 + Vite (JS, no TypeScript) with axios wrappers in `frontend/src/api/` (`requestWithRetry`), d3 for graph visualization (`GraphPanel.vue`), and a tiny `store/pendingUpload.js` for handing the uploaded file from Home to the process view.

## Other notes

- `ROADMAP.md` lists planned work (Python 3.12 support, `/api/status`, configurable search weights); `docs/progress.md` is the Zep→Neo4j migration log (some paths there, e.g. `backend/app/llm/`, were later moved to `backend/app/storage/` and `utils/`).
- `frontend/CHINESE_TEXT_INVENTORY.md` tracks strings translated from Chinese.
