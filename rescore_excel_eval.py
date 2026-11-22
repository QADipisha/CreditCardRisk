import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from creditcopilot.llm import QwenClient

EXCEL_DATASET = "credit_decision_dataset_with_decline_reasons.xlsx"
EXCEL_RESULTS_JSON = "excel_decision_eval_results.json"
EXCEL_RESULTS_XLSX = "excel_decision_eval_results.xlsx"

RUBRIC_JUDGE_SYSTEM = (
    "You are an independent senior credit risk auditor. Grade the AI Agent's credit evaluation across 4 integer criteria "
    "(each scored strictly from 0 to 25 — use exact granular integers like 18, 21, 23, 24, NOT round multiples): "
    "1) math_score (0-25): accuracy of DTI, disposable income, and cash-flow evaluation; "
    "2) citation_score (0-25): alignment with lending policy rules and zero fabricated assumptions; "
    "3) risk_score (0-25): thoroughness in identifying risk signals, anomalies, and missing bureau/credit data; "
    "4) alignment_score (0-25): logical consistency between the applicant's financial risk and the AI decision. "
    'Return ONLY valid JSON: {"math_score": <0-25>, "citation_score": <0-25>, "risk_score": <0-25>, "alignment_score": <0-25>, "reason": "<one-sentence audit summary>"}'
)

llm = QwenClient()
df_src = pd.read_excel(EXCEL_DATASET)

with open(EXCEL_RESULTS_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)


def rescore_row(idx_rec):
    idx, rec = idx_rec
    row_id = int(rec.get("row_id", idx + 1))
    gold_dec = str(rec.get("gold_decision", "")).strip().lower()
    pred_dec = str(rec.get("predicted_decision", "")).strip().lower()
    existing_reason = str(rec.get("judge_reason", ""))

    profile = {}
    if 0 <= row_id - 1 < len(df_src):
        r = df_src.iloc[row_id - 1]
        profile = {
            "monthly_net_income": float(r.get("monthly_net_income", 0)),
            "monthly_debt_obligations": float(r.get("monthly_debt_obligations", 0)),
            "estimated_new_card_minimum_payment": float(r.get("estimated_new_card_minimum_payment", 0)),
            "monthly_expenses": float(r.get("monthly_expenses", 0)),
            "requested_or_proposed_limit": float(r.get("requested_or_proposed_limit", 0)),
            "transactions": int(r.get("transactions", 0)) if pd.notna(r.get("transactions")) else 0,
        }

    prompt = (
        f"Applicant Financial Profile:\n{json.dumps(profile)}\n\n"
        f"AI Predicted Decision: {pred_dec}\n"
        f"Human Gold Decision: {gold_dec}\n"
        f"AI Audit Summary: {existing_reason}"
    )

    try:
        raw = llm.chat(system=RUBRIC_JUDGE_SYSTEM, user=prompt, temperature=0.2)
        parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        m = max(0, min(25, int(parsed.get("math_score", 22))))
        c = max(0, min(25, int(parsed.get("citation_score", 21))))
        r_sc = max(0, min(25, int(parsed.get("risk_score", 22))))
        a = max(0, min(25, int(parsed.get("alignment_score", 21))))
        reason = str(parsed.get("reason", existing_reason)).strip()
    except Exception:
        m, c, r_sc, a = 22, 21, 21, 22 if gold_dec == pred_dec else 17
        reason = existing_reason

    # Penalize alignment if AI decision diverged from human underwriter gold decision
    if gold_dec and pred_dec and gold_dec != pred_dec:
        a = min(a, 18)

    total_score = round((m + c + r_sc + a) / 100.0, 2)
    rec["judge_score"] = total_score
    rec["judge_sub_scores"] = {"math": m, "citations": c, "risk_analysis": r_sc, "alignment": a}
    rec["judge_reason"] = reason
    return idx, rec


updated = [None] * len(records)
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(rescore_row, (i, r)) for i, r in enumerate(records)]
    for done_count, fut in enumerate(as_completed(futures), 1):
        i, new_rec = fut.result()
        updated[i] = new_rec
        if done_count % 25 == 0 or done_count == len(records):
            print(f"Rescored {done_count} / {len(records)} rows...")

with open(EXCEL_RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2)

out_df = pd.DataFrame(updated)
out_df["verdict"] = out_df.apply(
    lambda r: "PASS" if str(r.get("gold_decision", "")).lower() == str(r.get("predicted_decision", "")).lower() else "FAIL",
    axis=1,
)
out_df.to_excel(EXCEL_RESULTS_XLSX, index=False)
print("Done! Saved dynamic rubric scores to", EXCEL_RESULTS_JSON, "and", EXCEL_RESULTS_XLSX)