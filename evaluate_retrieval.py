"""Evaluate retrieval performance using the generated gold retrieval set."""
import json
import os
from creditcopilot.retrieval import HybridRAG, retrieval_recall_at_k

rag = HybridRAG(folder="index")
rag.load()

gold_path = "data/gold_retrieval_set.json"
if not os.path.exists(gold_path):
    raise FileNotFoundError(f"Could not find {gold_path}. Run your judge evaluation script first.")

with open(gold_path, "r", encoding="utf-8") as f:
    gold = json.load(f)

valid_gold = [g for g in gold if g.get("relevant_chunk_ids")]
print(f"Loaded {len(valid_gold)} labeled queries from {gold_path}.")

for k in [5, 10, 15]:
    score = retrieval_recall_at_k(valid_gold, rag, k=k)
    print(f"Recall@{k}: {score:.4f} ({score * 100:.2f}%)")