from creditcopilot.knowledge import download_and_extract
from creditcopilot.retrieval import HybridRAG
from creditcopilot.config import settings

if __name__=="__main__":
    pages,manifest=download_and_extract(target=700,folder=settings.docs_dir)
    print("Unique source PDFs:",len(manifest),"Extracted pages:",len(pages))
    rag=HybridRAG(folder=settings.index_dir); rag.build(pages)
    print("Hybrid FAISS + BM25 + reranker index saved.")
