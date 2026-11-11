"""Input safety helpers. Heuristic screening is defense-in-depth, not a guarantee."""
import re
INJECTION_PATTERNS=[r'ignore (all )?(previous|prior) instructions',r'reveal (the )?(system prompt|developer message)',r'you are now (an? )?(admin|system)',r'disregard (your|the) (rules|instructions)',r'execute .* instead of']
PII_PATTERNS={"email":r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',"phone":r'(?<!\d)(?:\+?\d[ .()-]?){9,14}(?!\d)',"pan_like":r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)',"aadhaar_like":r'(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)'}
def detect_prompt_injection(text):
    t=(text or '').lower(); return [p for p in INJECTION_PATTERNS if re.search(p,t)]
def detect_pii(text):
    out=[]
    for name,pat in PII_PATTERNS.items():
        if re.search(pat,text or '',re.I): out.append(name)
    return out
def redact_pii(text):
    out=text or ''
    for name,pat in PII_PATTERNS.items(): out=re.sub(pat,f'[REDACTED_{name.upper()}]',out,flags=re.I)
    return out
