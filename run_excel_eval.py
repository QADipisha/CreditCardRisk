import os
import re
import time
import json
import pandas as pd

from creditcopilot.llm import QwenClient
from creditcopilot.retrieval import HybridRAG
from creditcopilot.agents import CreditAgents
from creditcopilot.observability import Tracer

DATASET = "credit_decision_dataset_with_decline_reasons.xlsx"
OUTPUT_JSON = "excel_decision_eval_results.json"
OUTPUT_XLSX = "excel_decision_eval_results.xlsx"
TARGET_ROWS = 300  # College capstone benchmark target

SYSTEM_DECISION_JUDGE = (
    "You are an independent senior credit risk reviewer. "
    "Compare: 1. Gold human credit decision and decline reason 2. AI Agent generated assessment. "
    "Also extract the AI Agent's final recommendation as either 'approve' or 'decline'. "
    'Return ONLY valid JSON: {"agent_decision": "approve", "agreement": true, "score": 0.92, "reason": "short explanation"}'
)


def classify_agent_output(report: str, judge_decision: str = "") -> str:
    jd = str(judge_decision).strip().lower()
    if jd in {"approve", "decline"}:
        return jd

    rec_match = re.search(
        r"(?:recommendation|final decision|decision|conclusion|human review)(.*?)(?:\n#{2,}|\Z)",
        report,
        flags=re.IGNORECASE | re.DOTALL,
    )
    target = rec_match.group(1).lower() if rec_match else report[-600:].lower()

    if any(w in target for w in ["decline", "declined", "reject", "rejected", "do not approve", "ineligible"]):
        return "decline"
    return "approve"


def run_with_retry(fn, max_retries: int = 5, base_delay: float = 5.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as ex:
            if "429" in str(ex) and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"   429 rate/concurrency limit hit. Waiting {wait:.0f}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def main():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"Missing dataset: {DATASET}")

    rag = HybridRAG(folder="index")
    rag.load()
    llm = QwenClient()
    agent = CreditAgents(llm, rag, tracer=Tracer())

    df = pd.read_excel(DATASET).head(TARGET_ROWS)
    total_rows = len(df)
    print(f"Loaded {total_rows} target rows from {DATASET}.")

    # Resume from existing progress if available
    results = []
    completed_ids = set()
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list) and existing and "gold_decision" in existing[0]:
                results = [
                    r for r in existing
                    if r.get("predicted_decision") != "error" and int(r.get("row_id", 0)) <= TARGET_ROWS
                ]
                completed_ids = {int(r["row_id"]) for r in results}
                if completed_ids:
                    print(f"Resuming: {len(completed_ids)}/{total_rows} rows already completed, {total_rows - len(completed_ids)} remaining.")
        except Exception:
            results = []
            completed_ids = set()

    for idx, row in df.iterrows():
        row_id = int(idx) + 1
        if row_id in completed_ids:
            continue

        profile = {
            "monthly_net_income": float(row["monthly_net_income"]),
            "monthly_debt_obligations": float(row["monthly_debt_obligations"]),
            "estimated_new_card_minimum_payment": float(row["estimated_new_card_minimum_payment"]),
            "monthly_expenses": float(row["monthly_expenses"]),
            "requested_or_proposed_limit": float(row["requested_or_proposed_limit"]),
            "payment_missed_or_late": bool(row.get("payment_missed_or_late", False)),
            "transactions": [],
        }
        try:
            res, _ = run_with_retry(lambda: agent.run(task="acquisition", profile=profile))
            rep = res["report"]
            gold = str(row["decision"]).strip().lower()
            decline_reason = str(row.get("decline_reason", ""))

            time.sleep(1.5)
            raw = run_with_retry(
                lambda: llm.chat(
                    system=SYSTEM_DECISION_JUDGE,
                    user=f"Gold decision: {gold}\nGold decline reason: {decline_reason}\nAgent report:\n{rep}",
                    temperature=0.0,
                )
            )
            j = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            pred = classify_agent_output(rep, j.get("agent_decision", ""))

            results.append(
                {
                    "row_id": row_id,
                    "gold_decision": gold,
                    "predicted_decision": pred,
                    "judge_agreement": j.get("agreement", False),
                    "judge_score": j.get("score", 0.0),
                    "judge_reason": j.get("reason", ""),
                }
            )
            print(f"Row {row_id}/{total_rows} -> Gold: {gold} | Pred: {pred} | Score: {j.get('score')}")
        except Exception as ex:
            results.append(
                {
                    "row_id": row_id,
                    "gold_decision": str(row.get("decision", "")).lower(),
                    "predicted_decision": "error",
                    "judge_agreement": False,
                    "judge_score": 0.0,
                    "judge_reason": str(ex),
                }
            )
            print(f"Row {row_id}/{total_rows} -> ERROR: {ex}")

        results.sort(key=lambda r: int(r.get("row_id", 0)))
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        time.sleep(2.5)

    results.sort(key=lambda r: int(r.get("row_id", 0)))
    pd.DataFrame(results).to_excel(OUTPUT_XLSX, index=False)
    print(f"DONE! Saved {len(results)} rows to {OUTPUT_JSON} and {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()