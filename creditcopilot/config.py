import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
    ca_cert_path: str = os.getenv("CA_CERT_PATH", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    index_dir: str = os.getenv("INDEX_DIR", "index")
    docs_dir: str = os.getenv("DOCS_DIR", "data/pdfs")
    def validate(self):
        missing=[k for k,v in {"LLM_API_KEY":self.api_key,"LLM_BASE_URL":self.base_url}.items() if not v]
        if missing: raise ValueError("Missing configuration: "+", ".join(missing))
        if self.ca_cert_path and not os.path.isfile(self.ca_cert_path):
            raise FileNotFoundError(f"CA certificate not found: {self.ca_cert_path}")
settings=Settings()
