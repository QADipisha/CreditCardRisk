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

with gr.Blocks(title='Credit Card Risk Copilot') as demo:
    gr.Markdown('# Credit Card Acquisition & Portfolio Risk Copilot\nPlanner → Executor → Critic • Hybrid RAG • SQLite conversation state')
    gr.Markdown('Use synthetic or authorized minimized data only. PII-like input is blocked. Output is decision support and requires human review.')
    sid=gr.State(value=lambda: __import__('uuid').uuid4().hex)
    task=gr.Radio(['acquisition','portfolio'],value='acquisition',label='Workflow')
    profile=gr.Textbox(label='Customer profile / portfolio JSON',lines=12,value=json.dumps({'monthly_net_income':75000,'monthly_debt_obligations':12000,'estimated_new_card_minimum_payment':3000,'monthly_expenses':35000,'requested_or_proposed_limit':100000,'transactions':[]},indent=2))
    msg=gr.Textbox(label='Task / additional context',placeholder='Assess affordability and identify missing information')
    out=gr.Markdown()
    submit=gr.Button('Run agent workflow',variant='primary')
    submit.click(run_assessment,[msg,gr.State([]),task,profile,sid],out)
if __name__=='__main__': demo.launch(server_name='0.0.0.0',server_port=int(os.getenv('PORT','7860')))
