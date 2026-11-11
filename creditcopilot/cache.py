"""Small TTL cache and process-local rate limiter; use Redis/API gateway for multi-worker production."""
import time, hashlib, json, threading
class TTLCache:
    def __init__(self,ttl=60,max_items=256): self.ttl=ttl; self.max_items=max_items; self.data={}; self.lock=threading.Lock()
    def get(self,key):
        with self.lock:
            v=self.data.get(key)
            if not v:return None
            if v[0]<time.time(): self.data.pop(key,None); return None
            return v[1]
    def set(self,key,value):
        with self.lock:
            if len(self.data)>=self.max_items:self.data.pop(next(iter(self.data)))
            self.data[key]=(time.time()+self.ttl,value)
    @staticmethod
    def key(*parts): return hashlib.sha256(json.dumps(parts,sort_keys=True,default=str).encode()).hexdigest()
class RateLimiter:
    def __init__(self,limit=10,window=60): self.limit=limit; self.window=window; self.calls={}; self.lock=threading.Lock()
    def allow(self,identity):
        now=time.time()
        with self.lock:
            recent=[t for t in self.calls.get(identity,[]) if now-t<self.window]
            if len(recent)>=self.limit:self.calls[identity]=recent; return False
            recent.append(now); self.calls[identity]=recent; return True
