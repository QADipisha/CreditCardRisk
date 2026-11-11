import ssl, httpx
from openai import OpenAI
from .config import settings

class QwenClient:
    """OpenAI-compatible client for user's cloud Qwen endpoint; TLS verification remains enabled."""
    def __init__(self):
        settings.validate()
        verify = ssl.create_default_context(cafile=settings.ca_cert_path) if settings.ca_cert_path else True
        self.http = httpx.Client(verify=verify, timeout=httpx.Timeout(90, connect=20))
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url.rstrip("/")+"/",
                             http_client=self.http, max_retries=2)
    def chat(self, system, user, temperature=0.1, max_tokens=1800):
        r=self.client.chat.completions.create(
            model=settings.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content or ""
