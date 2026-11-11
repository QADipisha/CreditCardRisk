"""Lightweight JSONL tracing. Avoid logging raw profiles, prompts, secrets or PII."""
import json, os, time, uuid
from pathlib import Path
class Tracer:
    def __init__(self,path=None): self.path=Path(path or os.getenv('TRACE_PATH','logs/traces.jsonl')); self.path.parent.mkdir(parents=True,exist_ok=True)
    def emit(self,event,**fields):
        row={"trace_id":fields.pop('trace_id',None) or str(uuid.uuid4()),"timestamp":time.time(),"event":event,**fields}
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
        return row['trace_id']
