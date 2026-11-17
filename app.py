import os, json, gradio as gr
from creditcopilot.config import settings
from creditcopilot.llm import QwenClient
from creditcopilot.retrieval import HybridRAG
from creditcopilot.agents import CreditAgents
from creditcopilot.state import StateStore
from creditcopilot.cache import TTLCache, RateLimiter
from creditcopilot.observability import Tracer
from creditcopilot.failures import classify_exception

AGENT=None; STARTUP_ERROR=None
STATE=StateStore(); CACHE=TTLCache(ttl=int(os.getenv('CACHE_TTL_SECONDS','45')))
LIMITER=RateLimiter(limit=int(os.getenv('RATE_LIMIT_REQUESTS','8')),window=int(os.getenv('RATE_LIMIT_WINDOW','60')))
try:
    llm=QwenClient(); rag=HybridRAG(folder=settings.index_dir); rag.load()
    AGENT=CreditAgents(llm,rag,tracer=Tracer())
except Exception as exc: STARTUP_ERROR=classify_exception(exc)

def run_assessment(message,history,task,profile_json,session_id):
    if STARTUP_ERROR: return 'Startup error: '+json.dumps(STARTUP_ERROR)
    if not LIMITER.allow(session_id): return 'RATE_LIMITED: Too many requests. Please wait and retry.'
    try:
        profile=json.loads(profile_json or '{}')
        key=CACHE.key(task,profile,message)
        cached=CACHE.get(key)
        if cached is not None: result=cached
        else:
            state=STATE.get(session_id)
            # Message is a task clarification/context string; profile is structured data.
            task_text=(message or task).strip()
            result,state=AGENT.run(task,profile,session_id=session_id,state=state,user_request=message or '')
            STATE.save(session_id,state); CACHE.set(key,result)
        sources='\n'.join(f"- [{s['id']}] {s['title']} p.{s['page']} ({s['license']}) {s['url']}" for s in result['sources'])
        return (result['report']+'\n\n### Agent plan\n```json\n'+json.dumps(result['plan'],indent=2)+'\n```\n### Metrics\n```json\n'+json.dumps(result['metrics'],indent=2)+'\n```\n### Critic\n```json\n'+json.dumps(result['critic'],indent=2)+'\n```\n### Sources\n'+sources+'\n\n### Audit\n```json\n'+json.dumps({k:result[k] for k in ['trace_id','revisions','invalid_citation_ids','human_review_required','security_flags']},indent=2)+'\n```')
    except json.JSONDecodeError as exc: return 'INVALID_JSON_INPUT: '+str(exc)
    except Exception as exc:
        failure=classify_exception(exc); return f"{failure['code']}: {failure['message']}"

def run_numeric_assessment(message, task, monthly_net_income, monthly_debt_obligations,
                           estimated_new_card_minimum_payment, monthly_expenses,
                           requested_or_proposed_limit, transactions_json, payment_missed_or_late, session_id):
    profile = {
        "monthly_net_income": monthly_net_income,
        "monthly_debt_obligations": monthly_debt_obligations,
        "estimated_new_card_minimum_payment": estimated_new_card_minimum_payment,
        "monthly_expenses": monthly_expenses,
        "requested_or_proposed_limit": requested_or_proposed_limit,
        "payment_missed_or_late": bool(payment_missed_or_late)
    }
    try:
        profile["transactions"] = json.loads(transactions_json or "[]")
        if not isinstance(profile["transactions"], list):
            raise ValueError("Transactions must be a JSON list.")
    except Exception as exc:
        return "Invalid transactions JSON: " + str(exc)
    return run_assessment(message, [], task, json.dumps(profile), session_id)

with gr.Blocks(title="Credit Card Risk Copilot") as demo:
    gr.Markdown("# Credit Card Acquisition & Portfolio Risk Copilot")
    gr.Markdown("Planner → Executor → Critic • BM25 + FAISS + Hybrid RAG • Numeric user inputs")
    gr.Markdown("Use synthetic or authorized minimized data only. PII-like input is blocked. This is decision support, not an automated lending decision.")
    sid = gr.State(value=lambda: __import__("uuid").uuid4().hex)
    task = gr.Radio(["acquisition", "portfolio"], value="acquisition", label="Workflow")
    with gr.Row():
        income = gr.Number(label="Monthly net income", value=75000)
        debt = gr.Number(label="Monthly debt obligations", value=12000)
    with gr.Row():
        new_min = gr.Number(label="Estimated new card minimum payment", value=3000)
        expenses = gr.Number(label="Monthly expenses", value=35000)
    limit = gr.Number(label="Requested / proposed credit limit", value=100000)
    missed = gr.Checkbox(label="Customer reports missed / late payment", value=False)
    transactions = gr.Textbox(label="Transactions JSON (portfolio workflow; use [] if none)", lines=6, value="[]")
    msg = gr.Textbox(label="Question / additional context", placeholder="Assess affordability, identify risk signals and missing information")
    out = gr.Markdown()
    submit = gr.Button("Run agent workflow", variant="primary")
    submit.click(run_numeric_assessment,
        [msg, task, income, debt, new_min, expenses, limit, transactions, missed, sid], out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=True)
