import json,re
from pathlib import Path
import numpy as np, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

def chunks_from_pages(pages,words=420,overlap=70):
    out=[]
    for p in pages:
        w=p["text"].split()
        for start in range(0,len(w),words-overlap):
            t=" ".join(w[start:start+words])
            if len(t)>220:
                out.append({k:p.get(k,"") for k in ["title","url","license","page","file"]}|{"text":t})
    return out

def tok(s): return re.findall(r"[a-z0-9]+",s.lower())

class HybridRAG:
    def __init__(self,embedding="sentence-transformers/all-MiniLM-L6-v2",
                 reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",folder="index"):
        self.encoder=SentenceTransformer(embedding)
        self.reranker=CrossEncoder(reranker)
        self.folder=Path(folder); self.folder.mkdir(exist_ok=True)
        self.chunks=[]; self.faiss=None; self.bm25=None

    def build(self,pages):
        self.chunks=chunks_from_pages(pages)
        if not self.chunks: raise ValueError("No chunks to index")
        vectors=self.encoder.encode([x["text"] for x in self.chunks],normalize_embeddings=True,
                                    convert_to_numpy=True,show_progress_bar=True).astype("float32")
        self.faiss=faiss.IndexFlatIP(vectors.shape[1]); self.faiss.add(vectors)
        self.bm25=BM25Okapi([tok(x["text"]) for x in self.chunks])
        faiss.write_index(self.faiss,str(self.folder/"dense.faiss"))
        (self.folder/"chunks.json").write_text(json.dumps(self.chunks,ensure_ascii=False),encoding="utf-8")

    def load(self):
        self.faiss=faiss.read_index(str(self.folder/"dense.faiss"))
        self.chunks=json.loads((self.folder/"chunks.json").read_text(encoding="utf-8"))
        self.bm25=BM25Okapi([tok(x["text"]) for x in self.chunks])

    def search(self,query,k=6,candidate_k=25):
        if self.faiss is None:self.load()
        q=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
        scores,ids=self.faiss.search(q,min(candidate_k,len(self.chunks)))
        dense={int(i):float(s) for s,i in zip(scores[0],ids[0]) if i>=0}
        bm=self.bm25.get_scores(tok(query))
        bm_ids=np.argsort(bm)[::-1][:candidate_k]
        # Reciprocal-rank fusion avoids mixing incomparable raw scores.
        fused={}
        for rank,i in enumerate(dense.keys(),1): fused[i]=fused.get(i,0)+1/(60+rank)
        for rank,i in enumerate(bm_ids,1): fused[int(i)]=fused.get(int(i),0)+1/(60+rank)
        cand=sorted(fused,key=fused.get,reverse=True)[:candidate_k]
        pairs=[[query,self.chunks[i]["text"]] for i in cand]
        rr=self.reranker.predict(pairs)
        order=sorted(zip(cand,rr),key=lambda x:float(x[1]),reverse=True)[:k]
        return [{"chunk_id":i,"rerank_score":float(s),**self.chunks[i]} for i,s in order]

def retrieval_recall_at_k(test_queries,rag,k=5):
    """Each test item: {'query': str, 'relevant_chunk_ids': [int,...]}."""
    vals=[]
    for row in test_queries:
        got={x["chunk_id"] for x in rag.search(row["query"],k=k)}
        rel=set(row["relevant_chunk_ids"])
        vals.append(len(got&rel)/len(rel) if rel else 0.0)
    return sum(vals)/len(vals) if vals else None
