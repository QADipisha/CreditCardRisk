import json
import pandas as pd

# Match your exact filename in Colab
EXCEL_FILE = "credit_decision_dataset_with_decline_reasons.xlsx"
df = pd.read_excel(EXCEL_FILE)

# Pick 15 diverse cases: 5 approve, 5 decline (DTI), 5 decline (Limit)
approves = df[df["decision"] == "approve"].head(5)
dti_declines = df[
    (df["decision"] == "decline") &
    (df["decline_reason"].str.contains("15%", na=False))
].head(5)
limit_declines = df[
    (df["decision"] == "decline") &
    (~df["decline_reason"].str.contains("15%", na=False))
].head(5)

sample_df = pd.concat([approves, dti_declines, limit_declines]).reset_index(drop=True)

gold_cases = []
for idx, row in sample_df.iterrows():
    inc = float(row["monthly_net_income"])
    debt = float(row["monthly_debt_obligations"])
    min_pay = float(row["estimated_new_card_minimum_payment"])
    exp = float(row["monthly_expenses"])
    limit = float(row["requested_or_proposed_limit"])

    curr_dti = round((debt / inc) * 100, 2)
    post_dti = round(((debt + min_pay) / inc) * 100, 2)
    free_cash = round(inc - debt - min_pay - exp, 2)
    inc_to_limit = round((inc / limit) * 100, 2)
    decision = str(row["decision"]).lower()
    reason = "" if pd.isna(row["decline_reason"]) else str(row["decline_reason"])

    if decision == "approve":
        checks = [
            f"Current DTI ({curr_dti}%) and post-card DTI ({post_dti}%) are within acceptable limits",
            f"Healthy post-card net free cash buffer ({free_cash})",
            f"Income ({inc}) meets >= 80% of proposed credit limit ({limit}) at {inc_to_limit}%",
            "Flags missing bureau score / income verification for human underwriter review"
        ]
        draft_gold = (
            f"Applicant shows manageable leverage with current DTI of {curr_dti}% and post-card DTI of {post_dti}%, "
            f"leaving a positive monthly cash buffer of {free_cash}. Monthly net income ({inc}) is {inc_to_limit}% "
            f"of the requested {limit} limit. Affordability supports approval subject to human review of bureau score and income proof."
        )
    else:
        checks = [
            f"Identifies policy breach: {reason}",
            f"Calculates current DTI ({curr_dti}%) and post-card DTI ({post_dti}%)",
            f"Checks income ({inc}) against proposed limit ({limit}) ({inc_to_limit}%)",
            "Recommends decline at requested limit and flags for human review"
        ]
        draft_gold = (
            f"Recommend decline at the requested limit of {limit}. Policy breach triggered: {reason}. "
            f"Current DTI is {curr_dti}%, rising to {post_dti}% post-card, while income covers {inc_to_limit}% of the proposed limit. "
            f"Flagged for human reviewer sign-off or lower counter-offer consideration."
        )

    gold_cases.append({
        "case_id": f"GOLD-{idx + 1:02d}",
        "task": "acquisition",
        "profile": {
            "monthly_net_income": inc,
            "monthly_debt_obligations": debt,
            "estimated_new_card_minimum_payment": min_pay,
            "monthly_expenses": exp,
            "requested_or_proposed_limit": limit,
            "transactions": []
        },
        "expected_outcome": decision,
        "required_checks": checks,
        "gold_standard_report": draft_gold
    })

with open("gold_standard_reports.json", "w", encoding="utf-8") as f:
    json.dump(gold_cases, f, indent=2)

print(f"Created gold_standard_reports.json with {len(gold_cases)} cases.")
