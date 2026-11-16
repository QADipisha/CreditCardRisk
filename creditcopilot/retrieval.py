import json
import re
import os
import ssl
import time
import httpx
from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from openai import OpenAI
from .config import settings

# Load the exact tokenizer for bge-small-en-v1.5 to enforce exact token counts
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunks_from_pages(pages, max_tokens=450, overlap_tokens=50):
    """
    Splits pages into chunks by exact model token count (strictly <= 450 tokens),
    guaranteeing zero 422 token-limit errors regardless of numbers, tables, or symbols.
    """
    out = []
    step = max_tokens - overlap_tokens
    for p in pages:
        raw_text = re.sub(r'\s+', ' ', p.get("text", "")).strip()
        if not raw_text:
            continue
            
        # Encode full page text into exact token IDs (without special tokens yet)
        token_ids = TOKENIZER.encode(raw_text, add_special_tokens=False)
        
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start:start + max_tokens]
            if len(chunk_ids) < 30:  # Skip tiny trailing noise snippets
                continue
                
            t = TOKENIZER.decode(chunk_ids, skip_special_tokens=True).strip()
            if t:
                out.append({k: p.get(k, "") for k in ["title", "url", "license", "page", "file"]} | {"text": t})
    return out

def tok(s): 
    return re.findall(r"[a-z0-9]+", s.lower())

class HybridRAG:
    def __init__(self, embedding=None, reranker="cross-encoder/ms-marco-MiniLM-L-6-v2", folder=None):
        self.model_name = embedding or settings.embedding_model
        
        # Configure TLS verification for embeddings
        cert_path = settings.embedding_cert_path or settings.ca_cert_path
        verify = ssl.create_default_context(cafile=cert_path) if cert_path else True
        self.http_client = httpx.Client(verify=verify, timeout=httpx.Timeout(90, connect=20))
        
        # Initialize OpenAI-compatible embedding client
        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            http_client=self.http_client,
            max_retries=2
        )
        
        self.reranker = CrossEncoder(reranker)
        self.folder = Path(folder or settings.index_dir)
        self.folder.mkdir(exist_ok=True)
        self.chunks = []
        self.faiss = None
        self.bm25 = None

    def _safe_truncate(self, text, max_tokens=480):
        """Hard safety check: ensures any string sent to the API is <= 480 tokens."""
        ids = TOKENIZER.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            return TOKENIZER.decode(ids[:max_tokens], skip_special_tokens=True)
        return text

    def build(self, pages, batch_size=16):
        """
        Builds dense FAISS and sparse BM25 indices with batch size strictly capped at 16
        and exact token-verified chunks.
        """
        self.chunks = chunks_from_pages(pages, max_tokens=450, overlap_tokens=50)
        if not self.chunks:
            print("No chunks generated from pages.")
            return

        print(f"Generating embeddings for {len(self.chunks)} token-verified chunks using API model {self.model_name}...")
        texts = [c["text"] for c in self.chunks]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            safe_batch = [self._safe_truncate(t, max_tokens=480) for t in batch]
            
            retries = 4
            response = None
            for attempt in range(retries):
                try:
                    response = self.client.embeddings.create(
                        input=safe_batch,
                        model=self.model_name
                    )
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = 2 * (attempt + 1)
                        print(f"⚠️ API error on batch {i // batch_size}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e

            # Sort by index just in case the API returns items out of order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity using Inner Product
        faiss.normalize_L2(embeddings_np)
        dimension = embeddings_np.shape[1]
        self.faiss = faiss.IndexFlatIP(dimension)
        self.faiss.add(embeddings_np)

        # Build BM25 index
        tokenized_corpus = [tok(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.save()
        print(f"✔ Hybrid FAISS + BM25 index built and saved successfully to {self.folder}")

    def save(self):
        with open(self.folder / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        if self.faiss is not None:
            faiss.write_index(self.faiss, str(self.folder / "dense.faiss"))

    def load(self):
        chunks_path = self.folder / "chunks.json"
        faiss_path = self.folder / "dense.faiss"
        
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            tokenized_corpus = [tok(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        if faiss_path.exists():
            self.faiss = faiss.read_index(str(faiss_path))
            
        print(f"Loaded {len(self.chunks)} chunks and indices from {self.folder}")

    def search(self, query, k=15, candidate_k=100):
        if not self.chunks or self.faiss is None or self.bm25 is None:
            raise ValueError("RAG index is not loaded or built. Run build() or load() first.")

        safe_query = self._safe_truncate(query, max_tokens=480)

        # 1. Dense Retrieval
        retries = 3
        response = None
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(input=[safe_query], model=self.model_name)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1 + attempt)
                else:
                    raise e

        q_emb = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(q_emb)
        
        dense_scores, dense_indices = self.faiss.search(q_emb, min(candidate_k, len(self.chunks)))
        
        # 2. BM25 Retrieval
        tokenized_query = tok(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:min(candidate_k, len(self.chunks))]

        # Combine candidates
        candidate_indices = list(dict.fromkeys(list(dense_indices[0]) + list(bm25_indices)))
        candidate_indices = [idx for idx in candidate_indices if idx < len(self.chunks)]

        if not candidate_indices:
            return []

        # 3. Rerank candidates
        pairs = [[query, self.chunks[idx]["text"]] for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)

        scored_candidates = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in scored_candidates[:k]:
            chunk = self.chunks[idx].copy()
            chunk["chunk_id"] = idx
            chunk["score"] = float(score)
            results.append(chunk)

        return results
  
def retrieval_recall_at_k(gold, rag, k=15):
    """
    Computes average Recall@k across all labeled queries in the gold dataset.
    Skips any query where relevant_chunk_ids is empty.
    """
    recalls = []
    for item in gold:
        relevant = set(item.get("relevant_chunk_ids", []))
        if not relevant:
            continue
        results = rag.search(item["query"], k=k)
        retrieved = {int(r["chunk_id"]) for r in results}
        recalls.append(len(relevant & retrieved) / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0