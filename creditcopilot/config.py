import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    # LLM Configs
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
    llm_cert_path: str = os.getenv("LLM_CERT_PATH", "")
    
    # Embedding Configs
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5")
    embedding_cert_path: str = os.getenv("EMBEDDING_CERT_PATH", "")
    
    # General Paths & Fallback
    ca_cert_path: str = os.getenv("CA_CERT_PATH", "")
    index_dir: str = os.getenv("INDEX_DIR", "index")
    docs_dir: str = os.getenv("DOCS_DIR", "data/pdfs")
    
    def validate(self):
        missing = [k for k, v in {
            "LLM_API_KEY": self.api_key, 
            "LLM_BASE_URL": self.base_url,
            "EMBEDDING_API_KEY": self.embedding_api_key,
            "EMBEDDING_BASE_URL": self.embedding_base_url
        }.items() if not v]
        
        if missing: 
            raise ValueError("Missing configuration: " + ", ".join(missing))
            
        # Validate certificate paths if provided
        for name, path in [("LLM_CERT_PATH", self.llm_cert_path), 
                           ("EMBEDDING_CERT_PATH", self.embedding_cert_path), 
                           ("CA_CERT_PATH", self.ca_cert_path)]:
            if path and not os.path.isfile(path):
                raise FileNotFoundError(f"{name} not found: {path}")

settings = Settings()