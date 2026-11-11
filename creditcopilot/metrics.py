import math

def num(x):
    try:
        if x is None or x=="": return None
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError): return None

def acquisition_metrics(p):
    income=num(p.get("monthly_net_income"))
    obligations=num(p.get("monthly_debt_obligations"))
    proposed_min=num(p.get("estimated_new_card_minimum_payment"))
    expenses=num(p.get("monthly_expenses"))
    limit=num(p.get("requested_or_proposed_limit"))
    out={}
    if income and income>0 and obligations is not None:
        out["existing_obligations_to_income_pct"]=round(obligations/income*100,2)
    if income and income>0 and obligations is not None and proposed_min is not None:
        out["total_obligations_including_card_to_income_pct"]=round((obligations+proposed_min)/income*100,2)
    if income and expenses is not None and obligations is not None:
        out["estimated_monthly_free_cash_before_new_card"]=round(income-expenses-obligations,2)
    if limit is not None: out["proposed_credit_limit"]=limit
    return out

def portfolio_metrics(txns):
    """Transactions expected: date, amount, category, merchant, txn_type, status."""
    import pandas as pd
    if not txns: return {"transaction_count":0}
    df=pd.DataFrame(txns)
    if "amount" not in df: return {"error":"Transactions need amount"}
    df["amount"]=pd.to_numeric(df["amount"],errors="coerce")
    df=df.dropna(subset=["amount"])
    if df.empty: return {"transaction_count":0}
    spend=df[df.get("txn_type", "purchase")!="payment"]
    metrics={"transaction_count":int(len(df)),
             "total_transaction_value":round(float(df["amount"].abs().sum()),2)}
    if "date" in df:
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        df["month"]=df["date"].dt.to_period("M").astype(str)
        monthly=spend.groupby("month")["amount"].sum().sort_index()
        if len(monthly)>=2 and monthly.iloc[-2]!=0:
            metrics["month_on_month_spend_change_pct"]=round((monthly.iloc[-1]/monthly.iloc[-2]-1)*100,2)
    if "category" in spend:
        cat=spend.groupby("category")["amount"].sum().sort_values(ascending=False)
        metrics["top_spend_categories"]=cat.head(5).round(2).to_dict()
    if "status" in df:
        failed=df["status"].astype(str).str.lower().isin(["declined","failed","returned"])
        metrics["failed_transaction_count"]=int(failed.sum())
    return metrics

def early_warning_triggers(txns, profile=None):
    m=portfolio_metrics(txns); triggers=[]
    change=m.get("month_on_month_spend_change_pct")
    if change is not None and change>=40:
        triggers.append({"trigger":"Rapid spend acceleration","severity":"medium",
                         "evidence":f"Spend increased {change}% month over month.",
                         "action":"Review whether change is seasonal, planned, or potentially stressful."})
    if m.get("failed_transaction_count",0)>=3:
        triggers.append({"trigger":"Repeated failed transactions","severity":"medium",
                         "evidence":f"{m['failed_transaction_count']} failed/declined/returned transactions in supplied data.",
                         "action":"Check operational causes and payment capacity; do not infer delinquency without account records."})
    if profile and profile.get("payment_missed_or_late") is True:
        triggers.append({"trigger":"Reported missed/late payment","severity":"high",
                         "evidence":"Profile indicates a missed or late payment.",
                         "action":"Verify statement/payment records and contact through approved servicing workflow."})
    if not triggers:
        triggers.append({"trigger":"No configured trigger detected","severity":"informational",
                         "evidence":"Supplied data did not cross the prototype's illustrative trigger rules.",
                         "action":"Continue normal monitoring; absence of a trigger is not proof of low risk."})
    return m,triggers
