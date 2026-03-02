## Financial Intelligence Multi‑Agent Demo

This project is a small, production‑ready demo of a **multi‑agent financial intelligence system** built for technical interviews. It combines **LangGraph** for orchestration, **Vertex AI Gemini** (via `google-genai`) for reasoning, **FastAPI** for serving, and a **Tailwind UI dashboard** that streams agent activity in real time.

---

### High‑Level Architecture

- **Frontend (`index.html`)**
  - Single‑page “Bloomberg‑style” dark UI using TailwindCSS (CDN).
  - Inputs:
    - Text field for a stock ticker (e.g. `PYPL`).
    - “Run Analysis” button with loading spinner.
  - Real‑time log window implemented with `EventSource` (SSE):
    - Connects to `GET /analyze?ticker=...`.
    - Streams JSON `data:` events and renders a terminal‑like log:
      - **Node name** (Supervisor / Researcher / Analyst), color‑coded.
      - **Snippet** of each agent’s output per step.
      - Error lines and a `[DONE]` completion line.

- **API Layer (`main.py`, FastAPI)**
  - Endpoints:
    - `GET /healthz` – simple health check.
    - `POST /invoke` – SSE endpoint (JSON body `{ "ticker": "PYPL" }`); useful for CLI / tools.
    - `GET /analyze` – SSE endpoint (query `?ticker=PYPL`) for browser EventSource.
  - Streaming format:
    - `media_type="text/event-stream"` with lines of the form:
      - `data: {"node": "...", "ticker": "...", "snippet": "..."}\n\n`
      - Special events:
        - `{"event": "error", ...}` – validation or runtime errors.
        - `{"event": "end", ...}` – workflow finished.
  - Ticker validation:
    - Syntactic: must match `^[A-Z]{1,5}$`.
    - Semantic: small LLM call checks if the ticker is a real, actively traded stock.
      - If invalid → a single SSE error event is returned and the workflow does **not** run.

- **Orchestration Layer (LangGraph)**
  - Shared state type (`AgentState`):
    - `ticker`: the requested symbol.
    - `research`: synthesized web research summary.
    - `analysis`: technical risk assessment.
    - `supervisor_summary`: high‑level narrative of the workflow.
  - Nodes:
    - **Supervisor (`supervisor_node`)**
      - Coordinates the process and maintains an overall narrative.
      - Runs at the beginning and after each specialist.
      - Logs and writes a short “status update” into `supervisor_summary`.
    - **Researcher (`researcher_node`)**
      - Uses `DuckDuckGoSearchRun` to fetch latest public info for the ticker.
      - Feeds raw search results to Gemini, which produces a focused bullet‑point summary (news, fundamentals, key risks & catalysts).
      - Writes summary into `research`.
    - **Technical Analyst (`analyst_node`)**
      - Consumes `research` and produces a risk view:
        - Risk level (Low / Medium / High).
        - Numeric risk score (0–100).
        - Short narrative explaining risk / opportunity.
      - Writes output into `analysis`.
  - Routing logic:
    - Entry point: `supervisor`.
    - `supervisor_router(state)`:
      - If `research` is empty → go to `researcher`.
      - Else if `analysis` is empty → go to `analyst`.
      - Else → `END`.
    - After `researcher` and `analyst`, edges lead back to `supervisor`, so each specialist step is followed by a coordinating supervisor step.

---

### LLM & Tools Layer

- **LLM Client (`GenAIGeminiChat`)**
  - Thin wrapper around `google.genai.Client` that exposes `.invoke(messages)` to match LangGraph’s expectations.
  - Uses **Application Default Credentials (`google.auth.default()`)**:
    - Local: `gcloud auth application-default login` or a service account key via `GOOGLE_APPLICATION_CREDENTIALS`.
    - Cloud Run: uses the service account attached to the service (no keys required).
  - Configuration:
    - Project: inferred from ADC.
    - Location: `GENAI_LOCATION` env var (default `global`), matching the reference code.
    - Model: `VERTEX_MODEL_NAME` env var (default `gemini-3.1-pro-preview`).
  - Generation settings (`GenerateContentConfig`):
    - `max_output_tokens=2048`.
    - `temperature=0.2` (stable outputs for financial reasoning).

- **Web Research Tool**
  - `DuckDuckGoSearchRun` from `langchain-community`.
  - Depends on the `ddgs` package (added to `requirements.txt`).
  - Researcher agent prompts Gemini to extract a clean, concise set of bullet points from noisy search results.

---

### Deployment & Runtime Design

- **FastAPI App (`main.py`)**
  - Single module containing:
    - LLM / tools factory (`get_llms_and_tools`) with `@lru_cache` for reuse.
    - LangGraph construction (`get_graph`) with `@lru_cache`.
    - Node functions (Supervisor, Researcher, Analyst).
    - Streaming generator (`stream_graph_updates`) and HTTP endpoints.
  - CORS:
    - Configured to allow all origins so the static `index.html` opened from disk can call local or Cloud Run URLs.

- **Containerization (`Dockerfile`)**
  - Base image: `python:3.11-slim`.
  - Installs dependencies from `requirements.txt`.
  - Runs `uvicorn main:app` on port `8080` (Cloud Run compatible).

- **Deployment Script (`deploy.sh`)**
  - Deploys to **Cloud Run** using:
    - `PROJECT_ID`, `REGION`, optional `SERVICE_NAME`.
    - `--allow-unauthenticated` (public demo).
    - `--set-env-vars "GCP_REGION=${REGION}"` (for any region‑specific behavior).
    - Can be extended to set `GENAI_LOCATION` and `VERTEX_MODEL_NAME` per environment.
  - Intended to be used with a dedicated service account, e.g.:
    - `agentic-stock-sa@<PROJECT_ID>.iam.gserviceaccount.com`.
    - With at least `roles/aiplatform.user` granted on the project.

---

### Local Development Flow

1. **Python & venv**
   - Use Python **3.11** (matches Docker base image).
   - Create and activate a virtualenv, then install deps:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Credentials & model configuration**
   - One‑time ADC setup:

   ```bash
   gcloud auth application-default login
   gcloud config set project <PROJECT_ID>
   ```

   - Environment variables:

   ```bash
   export GENAI_LOCATION="global"
   export VERTEX_MODEL_NAME="gemini-3.1-pro-preview"
   ```

3. **Run the API locally**

   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8080
   ```

4. **Use the dashboard**
   - Ensure `index.html` has `API_BASE_URL = "http://localhost:8080"`.
   - Open `index.html` directly in a browser.
   - Enter a ticker (e.g. `PYPL`) and click **Run Analysis** to watch the agents stream their reasoning and coordination in real time.

This architecture is intentionally small but realistic: it demonstrates **orchestrated agents, external tools, Vertex AI integration, streaming UX, and Cloud Run deployment**, all in a single, easily explainable codebase. 

# agentic-stock
