import os
import json
import uuid
import pandas as pd
import gradio as gr

from creditcopilot.config import settings
from creditcopilot.llm import QwenClient
from creditcopilot.retrieval import HybridRAG
from creditcopilot.agents import CreditAgents
from creditcopilot.state import StateStore
from creditcopilot.cache import TTLCache, RateLimiter
from creditcopilot.observability import Tracer
from creditcopilot.failures import classify_exception

GOLD_FILE = "gold_standard_reports.json"
GOLD_RESULTS_JSON = "gold_report_eval_results.json"
LEGACY_RESULTS_JSON = "agent_llm_judge_results.json"

EXCEL_DATASET = "credit_decision_dataset_with_decline_reasons.xlsx"
EXCEL_RESULTS_JSON = "excel_decision_eval_results.json"

CUSTOM_CSS = """
#blocking-overlay {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(11, 15, 25, 0.80) !important;
    backdrop-filter: blur(5px) !important;
    z-index: 99999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: all !important;
}
.overlay-card {
    background: #161b26;
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 32px 40px;
    max-width: 460px;
    width: 90%;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65);
    color: #f8fafc;
}
.overlay-spinner {
    width: 48px;
    height: 48px;
    margin: 0 auto 18px auto;
    border: 4px solid rgba(249, 115, 22, 0.25);
    border-top-color: #f97316;
    border-radius: 50%;
    animation: copilotSpin 0.85s linear infinite;
}
@keyframes copilotSpin {
    to { transform: rotate(360deg); }
}
.overlay-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #ffffff;
}
.overlay-sub {
    font-size: 0.92rem;
    line-height: 1.5;
    color: #cbd5e1;
}
"""


def style_verdict_badge(verdict: str, header: bool = True) -> str:
    label = str(verdict).strip() or "FAIL"
    v = label.upper()
    font_size = "1.05rem" if header else "0.9rem"
    padding = "3px 12px" if header else "2px 10px"
    if v.startswith("PASS") or v == "TRUE":
        return (
            f'<span style="display:inline-block; font-size:{font_size}; font-weight:700; '
            f'font-style:italic; color:#22c55e; background:rgba(34,197,94,0.14); '
            f'border:1px solid rgba(34,197,94,0.45); border-radius:6px; '
            f'padding:{padding}; letter-spacing:0.4px; white-space:nowrap;">{label}</span>'
        )
    return (
        f'<span style="display:inline-block; font-size:{font_size}; font-weight:700; '
        f'font-style:italic; color:#ef4444; background:rgba(239,68,68,0.14); '
        f'border:1px solid rgba(239,68,68,0.45); border-radius:6px; '
        f'padding:{padding}; letter-spacing:0.4px; white-space:nowrap;">{label}</span>'
    )


def render_overlay_html(title: str, subtitle: str) -> str:
    return f"""
    <div class="overlay-card">
        <div class="overlay-spinner"></div>
        <div class="overlay-title">{title}</div>
        <div class="overlay-sub">{subtitle}</div>
    </div>
    """


AGENT = None
llm = None
STARTUP_ERROR = None

STATE = StateStore()
CACHE = TTLCache(ttl=int(os.getenv("CACHE_TTL_SECONDS", "300")))
LIMITER = RateLimiter(
    limit=int(os.getenv("RATE_LIMIT_REQUESTS", "12")),
    window=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)

try:
    llm = QwenClient()
    rag = HybridRAG(folder=settings.index_dir)
    rag.load()
    AGENT = CreditAgents(llm, rag, tracer=Tracer())
except Exception as exc:
    STARTUP_ERROR = classify_exception(exc)


# ---------------------------------------------------------------------------
# TAB 1: Live Credit Card Acquisition Risk Copilot
# ---------------------------------------------------------------------------
def run_assessment(message, history, task, profile_json, session_id):
    if STARTUP_ERROR:
        return "Startup error: " + json.dumps(STARTUP_ERROR)
    if not LIMITER.allow(session_id):
        return "RATE_LIMITED: Too many requests. Please wait and retry."
    try:
        profile = json.loads(profile_json or "{}")
        key = CACHE.key(task, profile, message)
        cached = CACHE.get(key)
        if cached is not None:
            result = cached
        else:
            state = STATE.get(session_id)
            result, state = AGENT.run(
                task,
                profile,
                session_id=session_id,
                state=state,
                user_request=message or ""
            )
            STATE.save(session_id, state)
            CACHE.set(key, result)

        critic_info = result.get("critic", {})
        invalid_cites = result.get("invalid_citation_ids", [])
        critic_ok = critic_info.get("passed", True) if isinstance(critic_info, dict) else True
        raw_verdict = "PASS" if (critic_ok and len(invalid_cites) == 0) else "FAIL (Needs Review)"
        gate_badge = style_verdict_badge(raw_verdict, header=True)

        sources = "\n".join(
            f"- [{s['id']}] {s['title']} p.{s['page']} ({s['license']}) {s['url']}"
            for s in result.get("sources", [])
        )
        audit_data = {
            k: result.get(k)
            for k in ["trace_id", "revisions", "invalid_citation_ids", "human_review_required", "security_flags"]
        }

        return (
            f"### LLM Quality Gate Verdict: &nbsp;{gate_badge}\n"
            f"*(Verified: Financial metrics grounded, policy citations `[S#]` checked, human review flag evaluated)*\n\n"
            f"---\n\n"
            f"{result['report']}\n\n"
            f"---\n"
            f"### Agent Plan\n```json\n{json.dumps(result['plan'], indent=2)}\n```\n"
            f"### Metrics\n```json\n{json.dumps(result['metrics'], indent=2)}\n```\n"
            f"### Critic\n```json\n{json.dumps(result['critic'], indent=2)}\n```\n"
            f"### Sources\n{sources}\n\n"
            f"### Audit\n```json\n{json.dumps(audit_data, indent=2)}\n```"
        )
    except json.JSONDecodeError as exc:
        return "INVALID_JSON_INPUT: " + str(exc)
    except Exception as exc:
        failure = classify_exception(exc)
        return f"{failure['code']}: {failure['message']}"


def run_numeric_assessment_with_overlay(
    message, monthly_net_income, monthly_debt_obligations,
    estimated_new_card_minimum_payment, monthly_expenses,
    requested_or_proposed_limit, payment_missed_or_late, session_id
):
    yield (
        gr.update(
            value=render_overlay_html(
                "Generating Credit Risk Assessment...",
                "Running Hybrid RAG policy retrieval and Planner → Executor → Critic verification. Please wait..."
            ),
            visible=True
        ),
        gr.update()
    )
    try:
        profile = {
            "monthly_net_income": monthly_net_income,
            "monthly_debt_obligations": monthly_debt_obligations,
            "estimated_new_card_minimum_payment": estimated_new_card_minimum_payment,
            "monthly_expenses": monthly_expenses,
            "requested_or_proposed_limit": requested_or_proposed_limit,
            "payment_missed_or_late": bool(payment_missed_or_late),
            "transactions": []
        }
        final_output = run_assessment(message, [], "acquisition", json.dumps(profile), session_id)
    finally:
        yield (gr.update(visible=False), final_output)


# ---------------------------------------------------------------------------
# TAB 2: Gold Standard Report Evaluation (15 Pre-Verified Human Reports)
# ---------------------------------------------------------------------------
def _get_gold_records():
    if os.path.exists(GOLD_RESULTS_JSON):
        with open(GOLD_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(LEGACY_RESULTS_JSON):
        with open(LEGACY_RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data, list) and "coverage_score" in data[0]:
            with open(GOLD_RESULTS_JSON, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2)
            return data
    return []


def inspect_case(selected_id: str, records: list = None):
    if records is None:
        records = _get_gold_records()
    if not records or not selected_id:
        return "_No report selected._", "_No report selected._"

    gold_map = {}
    if os.path.exists(GOLD_FILE):
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            gold_map = {c["case_id"]: c.get("gold_standard_report", "") for c in json.load(f)}

    for r in records:
        if r.get("case_id") == selected_id:
            gold_text = r.get("gold_standard_report") or gold_map.get(selected_id, "_Not available_")
            agent_text = r.get("agent_report", "_Not available_")
            return (
                f"#### Human Gold Standard Report ({selected_id})\n\n{gold_text}",
                f"#### AI Copilot Generated Report ({selected_id})\n\n{agent_text}"
            )
    return "_Case not found._", "_Case not found._"


def _resolve_row_verdict(row: pd.Series) -> str:
    for key in ["verdict", "result", "status", "pass_fail", "final_verdict", "gate"]:
        val = str(row.get(key, "")).strip().upper()
        if val in ["PASS", "FAIL"]:
            return val
    try:
        cov = float(row.get("coverage_score", 0))
        reas = float(row.get("reasonableness_score", 0))
        return "PASS" if (cov >= 4 and reas >= 4) else "FAIL"
    except Exception:
        return "FAIL"


def load_saved_gold_eval():
    records = _get_gold_records()
    display_cols = ["case_id", "expected_outcome", "coverage_score", "reasonableness_score", "verdict", "judge_reason"]
    if not records:
        return (
            "### End-to-End Report Quality Gate: **NO SAVED RESULTS FOUND**",
            pd.DataFrame(columns=display_cols),
            gr.update(choices=[], value=None),
            "_No data available._",
            "_No data available._"
        )

    df = pd.DataFrame(records)
    for col in ["case_id", "expected_outcome", "judge_reason"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["coverage_score", "reasonableness_score"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["verdict"] = df.apply(_resolve_row_verdict, axis=1)
    valid = df[df["verdict"].isin(["PASS", "FAIL"])].copy()
    total = len(valid)
    passed = int((valid["verdict"] == "PASS").sum()) if total > 0 else 0
    pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0
    avg_cov = round(float(valid["coverage_score"].mean()), 2) if total > 0 else 0.0
    avg_reas = round(float(valid["reasonableness_score"].mean()), 2) if total > 0 else 0.0

    overall_label = (
        f"PASS — {pass_rate}% PASS ({passed} / {total} Gold Reports Passed)"
        if pass_rate >= 80.0
        else f"FAIL — {pass_rate}% PASS ({passed} / {total} Gold Reports Passed)"
    )
    banner = (
        f"### End-to-End Report Quality Gate Verdict: &nbsp;{style_verdict_badge(overall_label, header=True)}\n"
        f"**Mean Coverage Score:** `{avg_cov} / 5.0` &nbsp;|&nbsp; "
        f"**Mean Reasonableness / Accuracy Score:** `{avg_reas} / 5.0` &nbsp;|&nbsp; "
        f"*(Pass Rule: Coverage ≥ 4/5 AND Reasonableness ≥ 4/5)*"
    )

    table_df = df[display_cols].copy()
    table_df["verdict"] = table_df["verdict"].apply(lambda v: style_verdict_badge(v, header=False))

    case_ids = df["case_id"].tolist()
    first_id = case_ids[0] if case_ids else None
    first_gold, first_agent = inspect_case(first_id, records)
    return banner, table_df, gr.update(choices=case_ids, value=first_id), first_gold, first_agent


# ---------------------------------------------------------------------------
# TAB 3: Full Excel Dataset Evaluation (Pre-Computed Offline)
# ---------------------------------------------------------------------------
def _get_excel_records():
    for path in [EXCEL_RESULTS_JSON, LEGACY_RESULTS_JSON]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data, list) and "gold_decision" in data[0]:
                if path != EXCEL_RESULTS_JSON:
                    with open(EXCEL_RESULTS_JSON, "w", encoding="utf-8") as out_f:
                        json.dump(data, out_f, indent=2)
                return data
    return []


def load_saved_excel_eval():
    records = _get_excel_records()
    display_cols = ["row_id", "gold_decision", "predicted_decision", "verdict", "judge_score", "judge_reason"]
    if not records:
        return (
            f"### Full Excel Decision Benchmark (`{EXCEL_DATASET}`): **NO SAVED RESULTS YET**\n"
            f"*(Run `!python evaluate_agent_decisions_llm_judge.py` once before starting `app.py` to populate this benchmark.)*",
            pd.DataFrame(columns=display_cols)
        )

    df = pd.DataFrame(records)
    valid = df.dropna(subset=["gold_decision", "predicted_decision"]).copy()
    total = len(valid)
    matches = int((valid["gold_decision"].astype(str).str.lower() == valid["predicted_decision"].astype(str).str.lower()).sum()) if total > 0 else 0
    accuracy = round((matches / total) * 100, 1) if total > 0 else 0.0
    avg_judge = round(float(pd.to_numeric(valid.get("judge_score", 0), errors="coerce").fillna(0).mean()), 2) if total > 0 else 0.0

    valid["verdict"] = valid.apply(
        lambda r: "PASS" if str(r["gold_decision"]).lower() == str(r["predicted_decision"]).lower() else "FAIL",
        axis=1
    )

    overall_label = (
        f"PASS — {accuracy}% Accuracy ({matches} / {total} Decisions Matched)"
        if accuracy >= 80.0
        else f"FAIL — {accuracy}% Accuracy ({matches} / {total} Decisions Matched)"
    )

    banner = (
        f"### Full Excel Dataset Benchmark (`{EXCEL_DATASET}`): &nbsp;{style_verdict_badge(overall_label, header=True)}\n"
        f"**Total Records Evaluated:** `{total}` &nbsp;|&nbsp; "
        f"**Decision Accuracy:** `{accuracy}%` &nbsp;|&nbsp; "
        f"**Mean LLM-Judge Alignment Score:** `{avg_judge} / 1.0`"
    )

    for col in display_cols:
        if col not in valid.columns:
            valid[col] = ""
    table_df = valid[display_cols].copy()
    table_df["verdict"] = table_df["verdict"].apply(lambda v: style_verdict_badge(v, header=False))
    return banner, table_df


def load_all_on_startup():
    g_banner, g_table, g_dd, g_gold, g_agent = load_saved_gold_eval()
    e_banner, e_table = load_saved_excel_eval()
    return g_banner, g_table, g_dd, g_gold, g_agent, e_banner, e_table


# ---------------------------------------------------------------------------
# GRADIO UI LAYOUT (3 Tabs + Full-Screen Blocking Modal Overlay)
# ---------------------------------------------------------------------------
with gr.Blocks(title="Credit Card Acquisition Risk Copilot", css=CUSTOM_CSS) as demo:
    overlay = gr.HTML(
        value=render_overlay_html("Processing...", "Please wait..."),
        visible=False,
        elem_id="blocking-overlay"
    )

    gr.Markdown("# Credit Card Acquisition Risk Copilot")
    gr.Markdown("Planner → Executor → Critic • BM25 + FAISS + Hybrid RAG • End-to-End LLM-as-a-Judge Verified")
    gr.Markdown("Use synthetic or authorized minimized data only. PII-like input is blocked. This is decision support, not an automated lending decision.")

    sid = gr.State(value=lambda: uuid.uuid4().hex)

    with gr.Tabs():
        # --- TAB 1: Live Acquisition Risk Assessment ---
        with gr.Tab("1. Live Credit Risk Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        income = gr.Number(label="Monthly net income", value=75000)
                        debt = gr.Number(label="Monthly debt obligations", value=12000)
                    with gr.Row():
                        new_min = gr.Number(label="Estimated new card minimum payment", value=3000)
                        expenses = gr.Number(label="Monthly expenses", value=35000)
                    limit = gr.Number(label="Requested / proposed credit limit", value=100000)
                    missed = gr.Checkbox(label="Customer reports missed / late payment", value=False)
                    msg = gr.Textbox(
                        label="Question / additional context",
                        placeholder="Assess affordability, identify risk signals and missing information"
                    )
                    submit = gr.Button("Run agent workflow", variant="primary")

                with gr.Column(scale=2):
                    out = gr.Markdown(value="*Click **Run agent workflow** to generate the Credit Risk Assessment Report.*")

            submit.click(
                fn=run_numeric_assessment_with_overlay,
                inputs=[msg, income, debt, new_min, expenses, limit, missed, sid],
                outputs=[overlay, out],
                show_progress="hidden"
            )

        # --- TAB 2: Gold Standard Report Evaluation (15 Pre-Verified Reports) ---
        with gr.Tab("2. Gold Standard Evaluation (LLM-as-a-Judge)"):
            banner_md = gr.Markdown("### Loading Gold Standard Evaluation Results...")

            results_table = gr.Dataframe(
                label="End-to-End Report Validation Results (Coverage & Reasonableness)",
                datatype=["str", "str", "number", "number", "html", "str"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("### Side-by-Side Report Comparison (Human Gold Standard vs. AI Copilot)")
            case_dropdown = gr.Dropdown(label="Select Test Case ID to Inspect", choices=[])
            with gr.Row():
                gold_box = gr.Markdown()
                agent_box = gr.Markdown()

            case_dropdown.change(
                fn=lambda cid: inspect_case(cid),
                inputs=[case_dropdown],
                outputs=[gold_box, agent_box],
                show_progress="hidden"
            )

        # --- TAB 3: Full Excel Dataset Evaluation (Pre-Computed Offline) ---
        with gr.Tab("3. Full Excel Decision Evaluation"):
            excel_banner_md = gr.Markdown("### Loading Full Excel Decision Benchmark...")

            excel_results_table = gr.Dataframe(
                label="Full Dataset Decision & Reasoning Benchmark (credit_decision_dataset_with_decline_reasons.xlsx)",
                datatype=["number", "str", "str", "html", "number", "str"],
                interactive=False,
                wrap=True
            )

    demo.load(
        fn=load_all_on_startup,
        outputs=[banner_md, results_table, case_dropdown, gold_box, agent_box, excel_banner_md, excel_results_table],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=True
    )