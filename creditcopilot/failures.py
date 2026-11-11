"""Central failure taxonomy and safe exception classification."""
from enum import Enum
class FailureCode(str, Enum):
    CONFIG="CONFIG_ERROR"; LLM_TIMEOUT="LLM_TIMEOUT"; LLM_PROVIDER="LLM_PROVIDER_ERROR"
    RATE_LIMIT="RATE_LIMITED"; INVALID_JSON="INVALID_JSON"; TOOL="TOOL_EXECUTION_ERROR"
    RETRIEVAL="RETRIEVAL_ERROR"; STATE="STATE_ERROR"; INPUT="INPUT_VALIDATION_ERROR"
    PROMPT_INJECTION="PROMPT_INJECTION_DETECTED"; PII="PII_DETECTED"; CRITIC="CRITIC_REJECTED"
    UNKNOWN="UNKNOWN_ERROR"
def classify_exception(exc):
    name=type(exc).__name__.lower(); msg=str(exc).lower()
    if 'timeout' in name or 'timed out' in msg: code=FailureCode.LLM_TIMEOUT
    elif 'ratelimit' in name or 'rate' in msg and ('limit' in msg or '429' in msg): code=FailureCode.RATE_LIMIT
    elif 'apierror' in name or 'apistatus' in name: code=FailureCode.LLM_PROVIDER
    elif 'json' in name or 'json' in msg: code=FailureCode.INVALID_JSON
    elif 'retriev' in msg or 'faiss' in msg or 'index' in msg: code=FailureCode.RETRIEVAL
    elif 'config' in msg or 'missing' in msg and 'key' in msg: code=FailureCode.CONFIG
    else: code=FailureCode.UNKNOWN
    return {"code":code.value,"exception_type":type(exc).__name__,"message":str(exc)[:500]}
