CREDITCARD RISK COPILOT — PROJECT GUIDE
======================================

WHAT THIS PROJECT DOES
----------------------
This is a decision-support prototype for credit-card acquisition and portfolio
early-warning workflows. It combines Hybrid RAG (FAISS + BM25 + reranking) with
a Planner -> Executor -> Critic agent loop. It is not a validated credit scoring
model and must not autonomously approve or decline real applications. A qualified
human decision-maker must review every assessment.

PROJECT CONTENTS
----------------
creditcopilot/agents.py       Planner/Executor/Critic orchestration and revision loop
creditcopilot/retrieval.py    Hybrid retrieval and reranking
creditcopilot/metrics.py      Deterministic financial metrics and warning triggers
creditcopilot/state.py        SQLite conversation state across turns
creditcopilot/observability.py JSONL tracing with trace IDs
creditcopilot/failures.py     Failure taxonomy and exception classification
creditcopilot/security.py     Prompt-injection screening and PII detection/redaction helpers
creditcopilot/cache.py        TTL cache and process-local rate limiter
app.py                       Gradio front end
Dockerfile / docker-compose.yml Container deployment files
tests/                       Security, cache, rate-limit, JSON and failure tests
.github/workflows/tests.yml  GitHub Actions test workflow
CreditCardRisk_Colab_11Nov2026_v2.ipynb Colab notebook with agent invocation

GOOGLE COLAB QUICK START
------------------------
1. Upload and extract the ZIP in Colab.
2. Change directory to the extracted colab_project folder.
3. Install dependencies: %pip install -r requirements.txt
4. Add Colab Secrets: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL. Add CA_CERT_PATH only
   if your endpoint uses a private CA certificate. Never upload private keys.
5. Run notebook cells in order. Build/load the knowledge base and index first.
6. Run the section named "Agentic layer: Planner -> Executor -> Critic". It uses
   synthetic example data and writes trace/state files under logs/ and data/.

RUN THE FRONT END LOCALLY OR ON A SERVER
----------------------------------------
1. Install Python 3.11. Create a virtual environment.
2. Copy .env.example to .env and fill in the API key and endpoint.
3. Put the built retrieval index in ./index (dense.faiss and chunks.json).
4. Install: pip install -r requirements.txt
5. Start: python app.py
6. Open http://localhost:7860

DOCKER DEPLOYMENT
-----------------
Ensure .env and index/ exist, then run: docker compose up --build
The app listens on port 7860. Persist data/ and logs/ volumes. Put the service
behind HTTPS and an authenticated reverse proxy before exposing it publicly.

GIT / GITHUB — WHAT IT MEANS
----------------------------
Git records versions of your code so you can track changes and collaborate. A Git
repository is the project folder plus its version history. This delivered project folder is already initialized as a local Git repository and
contains an initial commit, .gitignore and a GitHub Actions test workflow. To publish
on GitHub, create an empty private repository, then in this folder run the commands
GitHub provides to add its remote and push the main branch (typically `git remote add
origin YOUR_REPOSITORY_URL`, `git branch -M main`, and `git push -u origin main`).
Do not commit .env,
API keys, certificates, customer data, index artifacts or runtime logs.

AGENT WORKFLOW
--------------
Planner creates a structured plan. Executor calls deterministic financial metrics
and Hybrid RAG, then prepares an evidence-backed report. Critic checks the report
and may request one revision. Malformed planner/critic JSON is classified and handled
with a safe planner fallback or failed critic status. Tool/retrieval/provider failures
are traced and surfaced as typed failures. StateStore keeps a bounded summary of
recent turns in SQLite. Supply the same session_id to continue a conversation.

SECURITY AND RELIABILITY
------------------------
- Prompt injection: heuristic screening flags suspicious input; retrieved documents
  are explicitly treated as untrusted evidence, never instructions.
- PII: basic email/phone/card-like/Aadhaar-like patterns are detected; PII-like input
  is blocked in the agent entry point. False positives/negatives are possible. This is
  not a substitute for a vetted DLP service, tokenization or privacy review.
- Rate limiting and caching in this prototype are process-local. For multiple workers,
  use Redis or an API gateway and authenticated user identities.
- Tracing writes event metadata, not raw prompts/profiles. Protect logs and set retention.
- JSON parsing tolerates fenced JSON and rejects malformed/non-object responses.
- Always validate tool outputs, source citations and policy thresholds.

TESTS
-----
Run: pytest -q
The tests cover JSON parsing, failure classification, rate limiting, cache behavior,
prompt-injection pattern detection, and PII detection/redaction. Add integration tests
with mocked LLM/RAG tools before production.

PRODUCTION READINESS CHECKLIST
------------------------------
[ ] Add authentication, authorization and per-user quotas.
[ ] Replace in-memory rate limit/cache with Redis or gateway controls.
[ ] Add persistent centralized traces, metrics, alerts and retention policies.
[ ] Use a proper DLP/PII detector and data minimization/tokenization.
[ ] Add adversarial prompt-injection red-team tests and retrieved-document scanning.
[ ] Add tool allowlists, strict schemas, timeouts, retries and idempotency.
[ ] Validate credit policy, explainability, fairness, drift and regulatory obligations.
[ ] Conduct security review, penetration testing and human-oversight sign-off.
[ ] Deploy behind HTTPS; store secrets in a secret manager, not in Git.

IMPORTANT LIMITATIONS
---------------------
This is a deployable prototype scaffold, not a production-certified banking system.
The process-local rate limiter/cache do not coordinate across replicas. The PII and
prompt-injection detectors are heuristic. The workflow requires a working LLM endpoint
and a prebuilt FAISS/BM25 index. Never use real customer data in demos or public logs.
