"""Create a gold set with relevant chunk IDs after inspecting the corpus.
Run after build_knowledge_base.py; evaluation is deliberately separate from answer scoring."""
import json
from creditcopilot.retrieval import HybridRAG,retrieval_recall_at_k
rag=HybridRAG(); rag.load()
# Replace examples with manually judged relevant chunk IDs from index/chunks.json.
gold=[
 {"query":"credit card underwriting affordability and income verification","relevant_chunk_ids":[]},
 {"query":"early warning indicators for rising credit card spend and payment stress","relevant_chunk_ids":[]},
]
print("Recall@5:",retrieval_recall_at_k(gold,rag,k=5))
print("Populate relevant_chunk_ids using expert-labelled chunk IDs; empty labels are not a valid evaluation.")
