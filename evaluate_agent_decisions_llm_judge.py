"""
evaluate_agent_decisions_llm_judge.py
Evaluates end-to-end CreditAgents reports against human-written
Gold Standard reports across Coverage and Reasonableness/Accuracy.
"""
import json
import pandas as pd
from creditcopilot.llm import QwenClient
from creditcopilot.retrieval import HybridRAG
from creditcopilot.agents import CreditAgents
from creditcopilot.observability import Tracer

GOLD_DATASET = "gold_standard_reports.json"

rag = HybridRAG(folder="index")
rag.load()
llm = QwenClient()
agent = CreditAgents(llm, rag, tracer=Tracer())

with open(GOLD_DATASET, "r", encoding="utf-8") as f:
    gold_cases = json.load(f)

SYSTEM_JUDGE = """
You are an independent senior credit risk reviewer evaluating an AI-generated
credit risk report against a human domain expert's Gold Standard Report.

Score the Agent Report on TWO dimensions from 1 to 5:

1. coverage_score (1-5):
   - 5: Covers all required financial metrics (DTI, post-card DTI, cash buffer, limit check),
        all required_checks, missing-info caveats, and policy citations.
   - 3: Covers main decision and basic ratios, but misses 1-2 required checks or caveats.
   - 1: Omits most required financial calculations and policy checks.

2. reasonableness_score (1-5):
   - 5: Financial math, threshold comparisons, risk logic, and final recommendation
        align completely with the Gold Standard human analyst report.
   - 3: Directionally reasonable conclusion, but contains minor threshold or reasoning gaps.
   - 1: Contradicts the Gold Standard decision or contains flawed financial reasoning.

Return ONLY valid JSON:
{
  "coverage_score": 4,
  "reasonableness_score": 5,
  "missing_elements": ["list any missed checks, or empty list"],
  "reason": "concise 1-2 sentence justification"
}
"""

def extract_agent_outcome(result: dict) -> str:
    report_lower = result.get("report", "").lower()
    if any(w in report_lower for w in ["recommend decline", "decision: decline", "should be declined", "reject"]):
        return "decline"
    if result.get("human_review_required") or "human review" in report_lower:
        return "human_review"
    return "approve"

results = []

for case in gold_cases:
    case_id = case["case_id"]
    try:
        result, _ = agent.run(
            task=case.get("task", "acquisition"),
            profile=case["profile"]
        )
        agent_report = result["report"]
        predicted_outcome = extract_agent_outcome(result)

        judge_prompt = f"""
Applicant Profile:
{json.dumps(case["profile"], indent=2)}

Required Domain Checkpoints:
{json.dumps(case["required_checks"], indent=2)}

Human Gold Standard Report:
{case["gold_standard_report"]}

AI Agent Generated Report:
{agent_report}
"""
        raw = llm.chat(system=SYSTEM_JUDGE, user=judge_prompt, temperature=0)
        start, end = raw.find("{"), raw.rfind("}")
        judge = json.loads(raw[start:end + 1])

        results.append({
            "case_id": case_id,
            "expected_outcome": case["expected_outcome"],
            "predicted_outcome": predicted_outcome,
            "coverage_score": judge.get("coverage_score"),
            "reasonableness_score": judge.get("reasonableness_score"),
            "missing_elements": "; ".join(judge.get("missing_elements", [])),
            "judge_reason": judge.get("reason"),
            "agent_report": agent_report
        })
    except Exception as ex:
        results.append({"case_id": case_id, "error": str(ex)})

out = pd.DataFrame(results)
out.to_excel("agent_llm_judge_results.xlsx", index=False)
out.to_json("agent_llm_judge_results.json", orient="records", indent=2)

valid = out.dropna(subset=["coverage_score", "reasonableness_score"])
print("=" * 60)
print("END-TO-END GOLD REPORT EVALUATION COMPLETE")
print("=" * 60)
print("Evaluated Cases:", len(valid))
print("Mean Coverage Score (out of 5):      ", round(valid["coverage_score"].mean(), 2))
print("Mean Reasonableness Score (out of 5):", round(valid["reasonableness_score"].mean(), 2))
# Add single PASS / FAIL verdict per report
valid["verdict"] = valid.apply(
    lambda r: "PASS" if (r["coverage_score"] >= 4 and r["reasonableness_score"] >= 4) else "FAIL",
    axis=1
)

pass_count = (valid["verdict"] == "PASS").sum()
total_count = len(valid)
pass_rate = round((pass_count / total_count) * 100, 1)

# Save clean summary table
valid[["case_id", "expected_outcome", "coverage_score", "reasonableness_score", "verdict", "judge_reason"]].to_excel(
    "agent_llm_judge_results.xlsx", index=False
)

print("=" * 60)
print(f"FINAL LLM-JUDGE RESULT: {pass_rate}% PASS ({pass_count}/{total_count} Reports Passed)")
print("=" * 60)