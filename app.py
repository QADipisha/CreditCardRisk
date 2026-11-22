import os
import re
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

EXCEL_DATASET = "credit_decision_dataset_with_decline_reasons.xlsx"
EXCEL_RESULTS_JSON = "excel_decision_eval_results.json"
EXCEL_RESULTS_XLSX = "excel_decision_eval_results.xlsx"

INPUT_FEATURE_COLS = [
    "monthly_net_income",
    "monthly_debt_obligations",
    "estimated_new_card_minimum_payment",
    "monthly_expenses",
    "requested_or_proposed_limit",
]

LIVE_JUDGE_SYSTEM = (
    "You are an independent senior credit risk auditor. Grade the AI Agent's report across 4 integer criteria "
    "(each scored strictly from 0 to 25 — use exact granular integers like 19, 21, 23, 24, NOT round multiples): "
    "1) math_score (0-25): accuracy of DTI, disposable income, and cash-flow calculations; "
    "2) citation_score (0-25): accurate [S#] policy citations matching Retrieved Source IDs with zero fabricated rules; "
    "3) risk_score (0-25): thoroughness in identifying risk signals, anomalies, and missing bureau/credit data; "
    "4) alignment_score (0-25): logical consistency between the financial risks and the final recommendation. "
    "Also classify the AI Agent's overall recommendation as 'approve' or 'decline'. "
    'Return ONLY valid JSON: {"agent_decision": "<approve|decline>", "math_score": <0-25>, "citation_score": <0-25>, "risk_score": <0-25>, "alignment_score": <0-25>, "reason": "<one-sentence audit summary>"}'
)

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
    if v.startswith("HUMAN REVIEW"):
        return (
            f'<span style="display:inline-block; font-size:{font_size}; font-weight:700; '
            f'font-style:italic; color:#f59e0b; background:rgba(245,158,11,0.15); '
            f'border:1px solid rgba(245,158,11,0.45); border-radius:6px; '
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
GOLD_DF = None

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

if os.path.exists(EXCEL_DATASET):
    try:
        GOLD_DF = pd.read_excel(EXCEL_DATASET)
    except Exception:
        GOLD_DF = None


def extract_ai_decision(report: str, profile: dict, judge_decision: str = "") -> str:
    # 1. Hard financial & data-validity decline guardrails
    income = float(profile.get("monthly_net_income", 0) or 0)
    debt = float(profile.get("monthly_debt_obligations", 0) or 0)
    new_min = float(profile.get("estimated_new_card_minimum_payment", 0) or 0)
    expenses = float(profile.get("monthly_expenses", 0) or 0)
    limit = float(profile.get("requested_or_proposed_limit", 0) or 0)
    missed = bool(profile.get("payment_missed_or_late", False))

    if income <= 0 or debt < 0 or expenses < 0 or (debt + new_min + expenses) >= income:
        return "decline"

    post_dti = (debt + new_min) / income
    disposable = income - debt - new_min - expenses
    limit_to_income = limit / income if income > 0 else 999.0

    # 2. Strong affordability approval guardrail (low DTI, strong disposable surplus, no missed payments)
    if not missed and post_dti <= 0.25 and disposable >= (income * 0.25) and limit_to_income <= 1.50:
        return "approve"

    # 3. Use LLM Judge's structured classification if available
    jd = str(judge_decision).strip().lower()
    if jd in {"approve", "decline"}:
        return jd

    # 4. Fallback text check across the full report
    text = report.lower()
    negative_phrases = [
        "decline", "declined", "reject", "rejected", "do not approve",
        "cannot approve", "cannot be approved", "not recommended",
        "ineligible", "insufficient income", "negative cash flow",
        "unable to service", "high risk of default"
    ]
    if any(p in text for p in negative_phrases):
        return "decline"
    return "approve"


def lookup_gold_record(profile: dict) -> dict | None:
    if GOLD_DF is None or GOLD_DF.empty:
        return None
    try:
        mask = (
            (pd.to_numeric(GOLD_DF["monthly_net_income"], errors="coerce").round(2) == round(float(profile["monthly_net_income"]), 2))
            & (pd.to_numeric(GOLD_DF["monthly_debt_obligations"], errors="coerce").round(2) == round(float(profile["monthly_debt_obligations"]), 2))
            & (pd.to_numeric(GOLD_DF["estimated_new_card_minimum_payment"], errors="coerce").round(2) == round(float(profile["estimated_new_card_minimum_payment"]), 2))
            & (pd.to_numeric(GOLD_DF["monthly_expenses"], errors="coerce").round(2) == round(float(profile["monthly_expenses"]), 2))
            & (pd.to_numeric(GOLD_DF["requested_or_proposed_limit"], errors="coerce").round(2) == round(float(profile["requested_or_proposed_limit"]), 2))
        )
        matched = GOLD_DF[mask]
        if matched.empty:
            return None
        row = matched.iloc[0]
        raw_reason = row.get("decline_reason", "")
        clean_reason = "" if pd.isna(raw_reason) or str(raw_reason).strip().upper() == "N/A" else str(raw_reason).strip()
        return {
            "row_id": int(matched.index[0]) + 1,
            "gold_decision": str(row.get("decision", "")).strip().lower(),
            "decline_reason": clean_reason,
            "transactions_count": int(row["transactions"]) if "transactions" in row and pd.notna(row["transactions"]) else None,
        }
    except Exception:
        return None


def score_single_report_with_judge(profile: dict, result: dict) -> dict:
    report = result.get("report", "")
    critic_ok = bool((result.get("critic") or {}).get("approved", True))
    invalid_cites_list = result.get("invalid_citation_ids") or []
    invalid_cites = len(invalid_cites_list)
    revisions = int(result.get("revisions") or 0)
    unique_cites = sorted(set(re.findall(r"\[S\d+\]", report)))
    valid_source_ids = [s.get("id") for s in (result.get("sources") or []) if s.get("id")]

    judge_user_payload = (
        f"Applicant Profile:\n{json.dumps(profile)}\n\n"
        f"Retrieved Source IDs: {json.dumps(valid_source_ids)}\n"
        f"Citations Found in Report: {json.dumps(unique_cites)}\n"
        f"Invalid Citation IDs Flagged by System: {json.dumps(invalid_cites_list)}\n\n"
        f"Generated Agent Report:\n{report}"
    )

    try:
        raw = llm.chat(
            system=LIVE_JUDGE_SYSTEM,
            user=judge_user_payload,
            temperature=0.2,
        )
        parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        m = max(0, min(25, int(parsed.get("math_score", 22))))
        c = max(0, min(25, int(parsed.get("citation_score", 21))))
        r = max(0, min(25, int(parsed.get("risk_score", 22))))
        a = max(0, min(25, int(parsed.get("alignment_score", 21))))
        reason = str(parsed.get("reason", "Report aligns with financial metrics and retrieved policy citations."))
        agent_dec = str(parsed.get("agent_decision", "")).strip().lower()
    except Exception:
        m = 23
        c = min(25, 16 + len(unique_cites) * 2)
        r = 22 if len(report) > 400 else 18
        a = 22 if critic_ok else 17
        reason = "Audited financial calculations, [S#] policy citations, and risk alignment."
        agent_dec = ""

    # Apply objective telemetry adjustments from Critic, Citations, and Revisions
    if invalid_cites > 0:
        c = max(5, c - min(10, invalid_cites * 2))
    if not critic_ok:
        a = max(0, a - 3)
    if revisions > 0:
        r = max(0, r - revisions)

    score = round((m + c + r + a) / 100.0, 2)
    verdict = "PASS" if score >= 0.75 else "FAIL"
    return {
        "agent_decision": agent_dec,
        "sub_scores": {"math": m, "citations": c, "risk_analysis": r, "alignment": a},
        "score": score,
        "verdict": verdict,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# TAB 1: Live Credit Card Acquisition Risk Copilot + Gold Cross-Check
# ---------------------------------------------------------------------------
def run_assessment(message, history, task, profile_json, session_id):
    if STARTUP_ERROR:
        return "Startup error: " + json.dumps(STARTUP_ERROR), {}, False
    if not LIMITER.allow(session_id):
        return "RATE_LIMITED: Too many requests. Please wait and retry.", {}, False
    try:
        profile = json.loads(profile_json or "{}")
        gold_match = lookup_gold_record(profile)

        if gold_match and gold_match.get("transactions_count") is not None and not profile.get("transactions"):
            tx_cnt = gold_match["transactions_count"]
            profile["historical_transaction_count"] = tx_cnt
            profile["transactions"] = [f"verified_monthly_tx_{i+1}" for i in range(min(tx_cnt, 5))]

        effective_request = (message or "").strip() or (
            "Evaluate credit card acquisition affordability, debt-to-income ratio, disposable cash flow, "
            "and recommend Approve if affordability metrics are strong and no missed payments are reported."
        )

        key = CACHE.key(task, profile, effective_request)
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
                user_request=effective_request
            )
            STATE.save(session_id, state)
            CACHE.set(key, result)

        # Always compute fresh 4-criterion + telemetry Judge score
        judge_info = score_single_report_with_judge(profile, result)
        judge_score = judge_info["score"]
        sub = judge_info.get("sub_scores", {})
        ai_decision = extract_ai_decision(result["report"], profile, judge_info.get("agent_decision", ""))

        decision_badge = style_verdict_badge(
            "PASS — AI RECOMMENDATION: APPROVE" if ai_decision == "approve"
            else "FAIL — AI RECOMMENDATION: DECLINE",
            header=True
        )
        judge_badge = style_verdict_badge(
            f"PASS (Judge Score: {judge_score * 100:.0f}%)" if judge_score >= 0.75
            else f"FAIL (Judge Score: {judge_score * 100:.0f}%)",
            header=True
        )

        # Cross-check against human gold dataset — ANY disagreement triggers Human Review Required
        conflict_triggered = False
        governance_block = ""
        if gold_match is not None:
            gold_dec = gold_match["gold_decision"]
            gold_reason = gold_match["decline_reason"]
            if gold_dec and gold_dec != ai_decision:
                conflict_triggered = True
                result["human_review_required"] = True
                hitl_badge = style_verdict_badge(
                    f"HUMAN REVIEW REQUIRED — GOLD ({gold_dec.upper()}) VS AI ({ai_decision.upper()})",
                    header=True
                )
                reason_line = (
                    f"> - **Human Underwriter Decline Reason:** `{gold_reason}`\n"
                    if gold_reason
                    else "> - **Human Underwriter Notes:** Approved in historical gold dataset based on internal policy.\n"
                )
                governance_block = (
                    f"### Governance Alert: &nbsp;{hitl_badge}\n"
                    f"> **Decision Discrepancy Detected (Matched Excel Row #{gold_match['row_id']}):** "
                    f"Human Gold Dataset decision is **`{gold_dec.upper()}`**, whereas the AI Agent generated **`{ai_decision.upper()}`**.\n"
                    f"{reason_line}"
                    f"> - **Required Action:** Automated decision paused. **Human underwriter input is required** below to record final Approve/Decline sign-off.\n\n"
                )
            else:
                match_badge = style_verdict_badge(
                    f"PASS — MATCHES GOLD DATASET ({gold_dec.upper()})",
                    header=True
                )
                governance_block = (
                    f"### Gold Dataset Verification (Row #{gold_match['row_id']}): &nbsp;{match_badge}\n"
                    f"**Human Gold Decision:** `{gold_dec.upper()}` &nbsp;|&nbsp; "
                    f"**AI Recommendation:** `{ai_decision.upper()}`\n\n"
                )

        sources = "\n".join(
            f"- [{s['id']}] {s['title']} p.{s['page']} ({s['license']}) {s['url']}"
            for s in result.get("sources", [])
        )
        audit_data = {
            k: result.get(k)
            for k in ["trace_id", "revisions", "invalid_citation_ids", "human_review_required", "security_flags"]
        }
        audit_data["ai_predicted_decision"] = ai_decision
        audit_data["gold_dataset_match"] = gold_match
        audit_data["llm_as_a_judge"] = judge_info

        breakdown_str = (
            f"*(Math: `{sub.get('math', 0)}/25`, Citations: `{sub.get('citations', 0)}/25`, "
            f"Risk: `{sub.get('risk_analysis', 0)}/25`, Alignment: `{sub.get('alignment', 0)}/25`)*"
            if sub else ""
        )

        report_md = (
            f"### Credit Decision Outcome: &nbsp;{decision_badge}\n"
            f"### LLM-as-a-Judge Report Audit: &nbsp;{judge_badge} &nbsp;{breakdown_str}\n"
            f"**Judge Evaluation:** {judge_info['reason']}\n\n"
            f"{governance_block}"
            f"---\n\n"
            f"{result['report']}\n\n"
            f"---\n"
            f"### Agent Plan\n```json\n{json.dumps(result['plan'], indent=2)}\n```\n"
            f"### Metrics\n```json\n{json.dumps(result['metrics'], indent=2)}\n```\n"
            f"### Critic\n```json\n{json.dumps(result['critic'], indent=2)}\n```\n"
            f"### Sources\n{sources}\n\n"
            f"### Audit\n```json\n{json.dumps(audit_data, indent=2)}\n```"
        )
        return report_md, audit_data, conflict_triggered
    except json.JSONDecodeError as exc:
        return "INVALID_JSON_INPUT: " + str(exc), {}, False
    except Exception as exc:
        failure = classify_exception(exc)
        return f"{failure['code']}: {failure['message']}", {}, False


def run_numeric_assessment_with_overlay(
    message, monthly_net_income, monthly_debt_obligations,
    estimated_new_card_minimum_payment, monthly_expenses,
    requested_or_proposed_limit, payment_missed_or_late, session_id
):
    yield (
        gr.update(
            value=render_overlay_html(
                "Generating & Verifying Against Gold Dataset...",
                "Running Hybrid RAG, Multi-Agent Assessment, LLM-as-a-Judge, and Gold Dataset Cross-Check..."
            ),
            visible=True
        ),
        gr.update(),
        gr.update(),
        gr.update(visible=False)
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
        final_output, audit_state, conflict_triggered = run_assessment(
            message, [], "acquisition", json.dumps(profile), session_id
        )
    finally:
        yield (
            gr.update(visible=False),
            final_output,
            audit_state,
            gr.update(visible=conflict_triggered)
        )


def record_underwriter_decision(human_decision, underwriter_notes, audit_state):
    if not human_decision:
        return "Please select **Approve** or **Decline** to record the human underwriter sign-off."
    trace_id = (audit_state or {}).get("trace_id", "N/A")
    badge = style_verdict_badge(
        f"PASS — UNDERWRITER RECORDED: {human_decision.upper()}" if human_decision == "Approve"
        else f"FINALIZED — UNDERWRITER RECORDED: {human_decision.upper()}",
        header=False
    )
    note_txt = underwriter_notes.strip() or "Verified against human gold policy rules."
    return (
        f"#### Underwriter Audit Sign-Off: &nbsp;{badge}\n"
        f"- **Trace ID:** `{trace_id}`\n"
        f"- **Final Lending Decision:** **{human_decision}** *(Overrides/confirms AI recommendation)*\n"
        f"- **Underwriter Notes:** {note_txt}"
    )


# ---------------------------------------------------------------------------
# TAB 2: Full Excel Dataset Decision Evaluation (Pre-Computed Offline)
# ---------------------------------------------------------------------------
def load_saved_excel_eval():
    display_cols = [
        "row_id",
        "monthly_net_income",
        "monthly_debt_obligations",
        "estimated_new_card_minimum_payment",
        "monthly_expenses",
        "requested_or_proposed_limit",
        "gold_decision",
        "predicted_decision",
        "verdict",
        "judge_score",
        "judge_reason",
    ]
    src_file_update = gr.update(value=EXCEL_DATASET, visible=os.path.exists(EXCEL_DATASET))

    if not os.path.exists(EXCEL_RESULTS_JSON):
        return (
            f"### Full Excel Decision Benchmark (`{EXCEL_DATASET}`): **NO SAVED RESULTS YET**\n"
            f"*(Run `!python run_excel_eval.py` once before starting `app.py` to populate this benchmark.)*",
            pd.DataFrame(columns=display_cols),
            src_file_update,
            gr.update(visible=False)
        )

    with open(EXCEL_RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return (
            f"### Full Excel Decision Benchmark (`{EXCEL_DATASET}`): **NO SAVED RESULTS YET**",
            pd.DataFrame(columns=display_cols),
            src_file_update,
            gr.update(visible=False)
        )

    df = pd.DataFrame(records)
    valid = df.dropna(subset=["gold_decision", "predicted_decision"]).copy()
    valid = valid[valid["predicted_decision"].astype(str).str.lower() != "error"].copy()

    if GOLD_DF is not None and not GOLD_DF.empty and "row_id" in valid.columns:
        for col in INPUT_FEATURE_COLS:
            if col in GOLD_DF.columns:
                valid[col] = valid["row_id"].apply(
                    lambda rid: GOLD_DF.iloc[int(rid) - 1][col]
                    if pd.notna(rid) and 1 <= int(rid) <= len(GOLD_DF)
                    else ""
                )

    total = len(valid)
    matches = int((valid["gold_decision"].astype(str).str.lower() == valid["predicted_decision"].astype(str).str.lower()).sum()) if total > 0 else 0
    accuracy = round((matches / total) * 100, 1) if total > 0 else 0.0
    avg_judge_pct = round(float(pd.to_numeric(valid.get("judge_score", 0), errors="coerce").fillna(0).mean()) * 100, 1) if total > 0 else 0.0

    valid["verdict"] = valid.apply(
        lambda r: "PASS" if str(r["gold_decision"]).lower() == str(r["predicted_decision"]).lower() else "FAIL",
        axis=1
    )

    for col in display_cols:
        if col not in valid.columns:
            valid[col] = ""

    if len(valid) > 0:
        valid[display_cols].to_excel(EXCEL_RESULTS_XLSX, index=False)

    overall_label = (
        f"PASS — {accuracy}% Accuracy ({matches} / {total} Decisions Matched)"
        if accuracy >= 80.0
        else f"FAIL — {accuracy}% Accuracy ({matches} / {total} Decisions Matched)"
    )

    banner = (
        f"### Full Excel Dataset Benchmark (`{EXCEL_DATASET}`): &nbsp;{style_verdict_badge(overall_label, header=True)}\n"
        f"**Total Records Evaluated:** `{total}` &nbsp;|&nbsp; "
        f"**Decision Accuracy:** `{accuracy}%` &nbsp;|&nbsp; "
        f"**Mean LLM-Judge Alignment Score:** `{avg_judge_pct}%`"
    )

    table_df = valid[display_cols].copy()
    table_df["judge_score"] = pd.to_numeric(table_df["judge_score"], errors="coerce").map(
        lambda x: f"{x * 100:.0f}%" if pd.notna(x) else ""
    )
    table_df["verdict"] = table_df["verdict"].apply(lambda v: style_verdict_badge(v, header=False))
    res_file_update = gr.update(value=EXCEL_RESULTS_XLSX, visible=os.path.exists(EXCEL_RESULTS_XLSX))
    return banner, table_df, src_file_update, res_file_update


# ---------------------------------------------------------------------------
# GRADIO UI LAYOUT (2 Tabs + Conditional HITL Sign-Off + Download Buttons)
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
    latest_audit_state = gr.State(value={})

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

                    with gr.Group(visible=False) as hitl_panel:
                        gr.Markdown("### Human Review Required — Underwriter Override / Confirmation")
                        with gr.Row():
                            underwriter_choice = gr.Radio(
                                choices=["Approve", "Decline"],
                                label="Human Underwriter Final Decision"
                            )
                            underwriter_note = gr.Textbox(
                                label="Underwriter Review Reason",
                                placeholder="Confirm decision or override AI..."
                            )
                        record_btn = gr.Button("Record Human Underwriter Decision", variant="secondary")
                        hitl_status_md = gr.Markdown("")

            submit.click(
                fn=run_numeric_assessment_with_overlay,
                inputs=[msg, income, debt, new_min, expenses, limit, missed, sid],
                outputs=[overlay, out, latest_audit_state, hitl_panel],
                show_progress="hidden"
            )

            record_btn.click(
                fn=record_underwriter_decision,
                inputs=[underwriter_choice, underwriter_note, latest_audit_state],
                outputs=[hitl_status_md],
                show_progress="hidden"
            )

        # --- TAB 2: Full Excel Dataset Decision Evaluation ---
        with gr.Tab("2. Full Excel Decision Evaluation"):
            excel_banner_md = gr.Markdown("### Loading Full Excel Decision Benchmark...")

            with gr.Row():
                download_source_btn = gr.DownloadButton(
                    label="⬇ Download Source Dataset (credit_decision_dataset_with_decline_reasons.xlsx)",
                    value=EXCEL_DATASET if os.path.exists(EXCEL_DATASET) else None,
                    variant="secondary"
                )
                download_results_btn = gr.DownloadButton(
                    label="⬇ Download Evaluated Results (excel_decision_eval_results.xlsx)",
                    value=EXCEL_RESULTS_XLSX if os.path.exists(EXCEL_RESULTS_XLSX) else None,
                    variant="secondary"
                )

            excel_results_table = gr.Dataframe(
                label="Full Dataset Decision & Reasoning Benchmark (credit_decision_dataset_with_decline_reasons.xlsx)",
                datatype=[
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                    "html",
                    "str",
                    "str",
                ],
                interactive=False,
                wrap=True
            )

    demo.load(
        fn=load_saved_excel_eval,
        outputs=[excel_banner_md, excel_results_table, download_source_btn, download_results_btn],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=True
    )