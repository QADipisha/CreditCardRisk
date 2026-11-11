import pytest
from creditcopilot.agents import parse_json_object
from creditcopilot.cache import TTLCache, RateLimiter
from creditcopilot.failures import classify_exception

def test_json_parser_handles_fenced_json():
    assert parse_json_object('```json\n{"passed": true}\n```')['passed'] is True

def test_json_parser_rejects_invalid():
    with pytest.raises(ValueError): parse_json_object('not json')

def test_cache_roundtrip():
    c=TTLCache(); c.set('x',{'v':1}); assert c.get('x')=={'v':1}

def test_rate_limiter():
    r=RateLimiter(limit=1,window=60); assert r.allow('a'); assert not r.allow('a')

def test_failure_taxonomy():
    assert classify_exception(TimeoutError())['code']=='LLM_TIMEOUT'

def test_planner_executor_critic_loop_and_state(tmp_path):
    from creditcopilot.agents import CreditAgents
    from creditcopilot.observability import Tracer
    from creditcopilot.state import StateStore
    class FakeLLM:
        def __init__(self): self.calls=[]
        def chat(self, system, user, temperature=0.1):
            self.calls.append(system)
            if 'Planner' in system:
                return '{"objective":"assess","steps":["calculate","retrieve","review"],"tools":["metrics","rag"],"missing_fields":[]}'
            if 'Critic' in system: return '{"passed":true,"issues":[],"revision_instructions":""}'
            return 'Assessment based on supplied facts. Refer for human review.'
    class FakeRAG:
        def search(self, query, k=7):
            return [{'title':'Policy guide','url':'https://example.org/policy','license':'CC-BY','page':1,'chunk_id':1,'text':'Affordability should be assessed.'}]
    llm=FakeLLM(); agent=CreditAgents(llm,FakeRAG(),tracer=Tracer(tmp_path/'trace.jsonl'))
    store=StateStore(str(tmp_path/'state.sqlite3')); state=store.get('session-1')
    result,state=agent.run('acquisition',{'monthly_net_income':75000,'monthly_debt_obligations':12000,'estimated_new_card_minimum_payment':3000,'monthly_expenses':35000},session_id='session-1',state=state)
    store.save('session-1',state)
    assert len(llm.calls)==3
    assert result['critic']['passed'] is True
    assert result['human_review_required'] is True
    assert len(store.get('session-1')['turns'])==1
    assert (tmp_path/'trace.jsonl').exists()
