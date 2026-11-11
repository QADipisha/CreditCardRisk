# CreditCardRisk Copilot

See **README.txt** for the complete setup, Colab, deployment, Git, security and operational guide.

This repository-ready prototype includes a Planner → Executor → Critic loop, SQLite multi-turn state, failure taxonomy, JSONL tracing, Gradio front end, process-local rate limiting and TTL caching, prompt-injection/PII screening, tests, Docker deployment files and a Colab notebook with explicit agentic invocation.

**Safety:** decision support only; require human review. Do not use real customer data in demos. The in-memory controls and heuristic security checks require hardening before production.
