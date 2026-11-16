import os
import certifi

# 1. Reset certifi to clean default state for public downloads
certifi_path = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi_path
os.environ["SSL_CERT_FILE"] = certifi_path

print("Certifi certificate bundle reset to default for public downloads.")

# 2. Load your Colab Secrets safely
try:
    from google.colab import userdata
    for k in ['LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL', 'EMBEDDING_API_KEY', 'EMBEDDING_BASE_URL']:
        try:
            os.environ[k] = userdata.get(k)
        except Exception:
            pass
    print("Colab secrets loaded.")
except ImportError:
    pass

os.environ['LLM_MODEL'] = 'Qwen/Qwen3-8B'

# 3. Run your knowledge base builder with local check
from creditcopilot.knowledge import download_and_extract
from creditcopilot.retrieval import HybridRAG
from creditcopilot.config import settings

if __name__ == "__main__":
    os.makedirs(settings.docs_dir, exist_ok=True)
    
    # Check if PDFs are already present locally
    existing_pdfs = [f for f in os.listdir(settings.docs_dir) if f.endswith('.pdf')]
    
    if len(existing_pdfs) > 0:
        print(f"Found {len(existing_pdfs)} existing PDFs in '{settings.docs_dir}'. Skipping download to preserve corpus consistency.")
        pages, manifest = download_and_extract(target=2000, folder=settings.docs_dir)
    else:
        print("No local PDFs found. Starting document download and extraction...")
        pages, manifest = download_and_extract(target=2000, folder=settings.docs_dir)
        
    print("Unique source PDFs:", len(manifest), "Extracted pages:", len(pages))
    
    rag = HybridRAG(folder=settings.index_dir)
    rag.build(pages)
    print("Hybrid FAISS + BM25 + reranker index saved.")