# UPulse - Behavioral AI Recommendation Agent

A catalog-grounded product recommendation engine that watches how a user actually
behaves, builds a structured *cognitive model* of that behavior, and produces a
personally persuasive recommendation - without calling an LLM on every event.

Built for the hackathon brief: **"an AI agent that recommends courses based on
the user's past activity and behavioral patterns."** UPulse turns that into a
production-shaped system: event ingestion, a LangGraph agent, semantic retrieval
over the catalog, cost-controlled LLM spend, a real scheduler, and an observability
console - not a script.

---

## The story

Most "recommendation agents" in this space are thin wrappers that paste a user's
recent searches into a prompt and ask the LLM to pick courses. That is expensive,
unscalable, and often ungrounded, the LLM can invent courses that don't exist.

UPulse takes a different stance:

1. **Behavior is the raw material.** Every page view, search, product view, click,
   and time-on-page is captured as a typed behavioral event.
2. **An agent infers intent.** A LangGraph pipeline turns raw events into a
   structured cognitive model - stated intents, inferred interests, decision
   stage, purchase readiness, category affinity, session arc.
3. **Retrieval grounds everything.** The catalog is embedded and searched
   semantically (filtered by the user's proven category affinity). The agent
   can only ever recommend **real catalog products**.
4. **The LLM generates, it doesn't guess.** The model writes a persuasive
   narrative *for a user it actually understands*, conditioned on a real,
   retrieved candidate set - never on thin air.
5. **Every LLM call is auditable and paid for once.** A single trigger gate,
   a 30-minute freshness TTL, a per-user in-flight lock, and a scheduler that
   skips stale users keep spend bounded.

---

## Architecture

```
┌─────────────┐   events (typed, batched)   ┌──────────────┐
│  Browser /  │ ───────────────────────────▶│  FastAPI app │
│  mobile     │        POST /api/events      │   (UPulse)   │
└─────────────┘                              └──────┬───────┘
                                                     │ ingest
                                                     ▼
                                    ┌────────────────────────────┐
                                    │   SQL (SQLite / Postgres)  │
                                    │  users, events, products,  │
                                    │  recommendations,          │
                                    │  cognitive models,         │
                                    │  llm_call_logs             │
                                    └────────────────────────────┘
                                                     │ dual-write
                                                     ▼
                                    ┌────────────────────────────┐
                                    │  Vector store              │
                                    │  Chroma (dev) / pgvector   │
                                    │  (production)              │
                                    └────────────────────────────┘
                                                     ▲
                                    trigger gate      │  run agent
                                                     ▼
                                    ┌────────────────────────────┐
                                    │      LangGraph agent       │
                                    │  model_user → retrieve →   │
                                    │  assess → evaluate →       │
                                    │  filter → generate →       │
                                    │  reflect → store           │
                                    │                            │
                                    │  assess ──(poor)──▶ retry  │
                                    │  retry ──▶ evaluate        │
                                    └────────────────────────────┘
                                                     │
                                    ┌────────────────┴─────────────┐
                                    │  OpenRouter (LLM gateway)     │
                                    │  • 2 chat calls per run       │
                                    │  • 1 embedding call per run   │
                                    └──────────────────────────────┘
```

- **Backend:** FastAPI + SQLAlchemy 2.0 (async). SQLite for local dev, PostgreSQL
  for production.
- **Agent:** LangGraph (`app/agent/graph.py`) - a compiled, explicit state machine.
- **LLM gateway:** OpenRouter, reached through a single choke point
  (`app/services/llm_client.py`) that logs every call.
- **Vector store:** a facade (`app/services/vector_store.py`) that lazily selects
  Chroma (dev/SQLite) or pgvector (production/PostgreSQL). Both expose the same
  async API and return Chroma-shaped results.
- **Scheduling:** APScheduler runs the daily digest at 16:00 server time.

---

## The agent workflow (LangGraph)

The graph is the heart of the product. Every node reads the shared `AgentState`
and returns a partial update; LangGraph merges them.

```
model_user ─▶ retrieve ─▶ assess ─▶ evaluate ─▶ filter ─▶ generate ─▶ reflect ─▶ store ─▶ END
                                 │
                                 └─▶ retry (conditional, deterministic)
```

| Node | Role | LLM calls |
|------|------|-----------|
| `model_user` | Reads up to 50 recent events, asks the LLM to update the user's cognitive model (interests, decision stage, readiness, category affinity, session arc), persists it. | **1 chat** |
| `retrieve` | Builds a query from the cognitive model, embeds it, and runs a metadata-filtered semantic search over the catalog. **Dual-write guarantee:** only real catalog rows come back. | **1 embedding** |
| `assess` | Deterministic retrieval-quality gate: are there enough candidates, and is the best match actually similar? No LLM. | 0 |
| `retry` | If retrieval quality is poor *and* a category filter was applied, re-runs the search with the **same embedding** and no filter - a deterministic, zero-extra-cost adjustment. | 0 |
| `evaluate` | Deterministically scores each candidate against the cognitive model (semantic similarity + category affinity + inferred-intent keyword overlap). No LLM. | 0 |
| `filter` | Deterministically keeps the top-N candidates. No LLM. | 0 |
| `generate` | LLM writes a persuasive, personalized narrative grounded in the filtered candidates. | **1 chat** |
| `reflect` | Deterministic grounding/quality gate; never requests regeneration (so spend stays at exactly 3 AI calls/run). | 0 |
| `store` | Persists the recommendation (deactivating any previous active one), records reasoning + alternatives, all grounded in the real catalog. | 0 |

### Why the graph is genuinely adaptive

The `assess` node makes a real, observable decision per run:

- If retrieval quality is **good** (enough candidates, best match is similar), the
  graph flows straight through to `evaluate` - the normal 3-call path.
- If retrieval quality is **poor** and a category filter was narrowing the search,
  the graph routes to `retry`, which **relaxes the filter and re-searches with the
  already-computed embedding** (no new embedding call, no chat call).
- If quality is still poor, the run proceeds with the best grounded candidates it
  has rather than failing or inventing products.

The decision uses deterministic signals (candidate count, best cosine similarity)
that are already in state - no extra LLM call is added to "decide."

### Cost optimization (why it's exactly 3 AI calls per run)

- **2 chat calls** (`model_user` + `generate`) - only the two places that
  genuinely need a model.
- **1 embedding call** (`retrieve`) - reused verbatim if the conditional retry
  fires.
- `evaluate`, `filter`, `reflect`, `store`, and `assess`/`retry` are
  deterministic - **zero** LLM calls.

---

## Behavioral tracking

The `/api/events` endpoint accepts batched, typed events:

| Type | Weight | Meaning |
|------|--------|---------|
| `page_view` | 0.4 | Navigational interest |
| `product_view` | 1.5 | Real consideration signal |
| `search` | 2.0 | Explicit intent (+0.5 if a query is present) |
| `click` | 1.0 | Engagement |
| `time_spent` | 0.6 | Engagement depth (scaled, capped) |
| `add_to_cart` | 3.0 | High-intent |
| `checkout_start` | 4.0 | Highest intent |

Events are user-isolated, batched, and fed to `model_user`, which converts them
into a structured cognitive model:

- **stated_intents** - what the user explicitly searched for
- **inferred_intents** - deeper interests inferred from behavior
- **decision_stage / purchase_readiness / price_sensitivity**
- **category_affinity** - categories actually engaged with
- **session_arc** - one-sentence narrative of the browsing session

---

## Dual-write: SQL + vector store

Products exist in **two** places, kept in sync:

1. **SQL** (`products` table) - the source of truth for catalog data (title,
   description, category, price, level, rating).
2. **Vector store** (`course_embeddings` via pgvector, or the Chroma collection
   in dev) - embeddings for semantic search, keyed by the same product
   `vector_id` and carrying `sql_id` for grounding.

Every product create/update writes both; deletes remove both. Retrieval joins
vector hits back to SQL rows, so a recommendation can **never** reference a
product that isn't in the catalog.

---

## OpenRouter integration

All AI traffic flows through `app/services/llm_client.py`:

- **Chat:** `openrouter/free` by default (configurable). `response_format_json`
  with an in-prompt JSON contract plus defensive parsing.
- **Embeddings:** `openai/text-embedding-3-small` via OpenRouter.
- **Retry policy:** 5xx and network errors retry up to 3 times with exponential
  backoff; **429 is fail-fast** (retrying a quota-exhausted call only amplifies
  cost); 4xx are never retried.
- **Auditability:** every call (success or failure) is written to `llm_call_logs`
  - the Agent Console shows real numbers, never invented ones.

---

## Recommendation triggering (when does the agent run?)

Running the full pipeline costs 3 AI calls, so the trigger gate is deliberately
conservative. After each event batch it decides whether to run the agent:

- **Count-based cursor:** only events the agent hasn't seen yet count as new,
  selected **deterministically** (newest-first by `created_at`, then `id`, so
  events sharing an identical timestamp can never be mis-selected).
- **Threshold:** at least 3 new events **or** an aggregate weighted score ≥ 3.0.
- **Fresh-recommendation TTL:** if the user received a recommendation within the
  last 30 minutes, the agent skips the run unless the new activity contains a
  high-intent signal (search / add-to-cart / checkout) - this alone removes the
  largest waste class (re-running after light post-recommendation browsing).
- **Cooldown:** no more than one run per user per 60 seconds.
- **In-flight lock:** a per-user `asyncio` guard prevents concurrent runs for the
  same user, so a batch racing the daily digest can never double-run or double-write.

---

## Scheduled digest

`app/services/scheduler.py` runs a **real** daily digest (APScheduler, 16:00
server time) independent of any HTTP request:

1. Finds users with activity in the last 24 hours.
2. Skips users whose last recommendation is fresh (within the TTL) with no strong
   new signal - the same gate the event trigger uses.
3. Runs the agent for every qualifying user with reason `daily digest (scheduled)`.

The digest is started/stopped with the app lifespan and never blocks the API.

---

## Setup & run

### Prerequisites

- Python 3.11+
- An OpenRouter API key (free tier works)

### Local development (SQLite + Chroma)

```bash
# 1. Clone and enter the repo
cd SmartReco-Behavioral-AI-Recommendation-Agent

# 2. Create a virtualenv and install deps
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt

# 3. Configure environment
cp .env.example .env
#   - set OPENROUTER_API_KEY
#   - set SECRET_KEY to a random string
#   - set ADMIN_EMAIL / ADMIN_PASSWORD (the admin account is bootstrapped at startup)
#   - set MOCK_EMBEDDINGS=false for real semantic retrieval

# 4. Run
uvicorn app.main:app --reload
```

On startup the app idempotently bootstraps: schema → admin user → 52-course
catalog → course embeddings (all via OpenRouter embeddings). Open
`http://localhost:8000`.

### Production (PostgreSQL + pgvector)

`render.yaml` codifies the deployment. The app auto-creates schema, admin,
catalog, and pgvector embeddings on boot. Set these in the Render dashboard:

- `ADMIN_PASSWORD`, `SECRET_KEY`, `OPENROUTER_API_KEY` (never committed)
- `PUBLIC_BASE_URL` is wired automatically to the service URL via `fromService`
  (used to build working password-reset links).

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `production` on Render |
| `DATABASE_URL` | `sqlite+aiosqlite:///./upulse.db` | Postgres in production |
| `SECRET_KEY` | dev-only | JWT signing key - set a long random value |
| `ADMIN_EMAIL` | `upulse@admin.com` | Bootstrap admin email |
| `ADMIN_PASSWORD` | *(empty)* | Bootstrap admin password |
| `OPENROUTER_API_KEY` | *(empty)* | LLM gateway key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Gateway base URL |
| `OPENROUTER_MODEL` | `openrouter/free` | Chat model |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `OPENROUTER_MAX_TOKENS` | `4096` | Per-request token cap |
| `MOCK_EMBEDDINGS` | `false` | `true` = deterministic fake vectors (offline dev/tests) |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` / `SMTP_USE_TLS` | *(empty)* | Password-reset email delivery |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Public app URL for reset links |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` | `./chroma_data` / `products` | Dev vector store (ignored in prod) |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | - | Optional LangSmith observability |

---

## Implemented bonus features

- **Agent Console** (`/console`) - live view of the agent graph's runs: cognitive
  models, retrieval candidates, evaluation scores, narrative, reasoning chain,
  alternatives considered, and honest LLM-call logs (including mocked calls
  flagged as such).
- **Per-user in-flight lock** - no duplicate concurrent agent runs, ever.
- **Offline test suite** - 84 tests, fully mocked (no API keys, no network),
  covering every node, the full graph, the trigger gate, the scheduler, and
  dual-write correctness.
- **Deterministic adaptive retrieval** - conditional retrieval-quality gate with a
  filter-relaxing retry that costs zero extra AI calls.
- **Honest cost accounting** - exactly 3 AI calls per normal agent run, enforced
  by tests.
- **Idempotent bootstrap** - safe to restart repeatedly; never duplicates schema,
  admin, catalog, or embeddings.