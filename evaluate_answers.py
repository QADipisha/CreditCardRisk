"""Answer evaluation scaffold. Populate with gold questions and human-reviewed expected facts."""
import json
from pathlib import Path
from creditcopilot.llm import QwenClient
from creditcopilot.retrieval import HybridRAG
from creditcopilot.agents import CreditAgents

gold=json.loads(Path("data/gold_answer_set.json").read_text()) if Path("data/gold_answer_set.json").exists() else []
if not gold:
    raise SystemExit("Create data/gold_answer_set.json with human-reviewed test cases first.")
agent=CreditAgents(QwenClient(),HybridRAG())
rows=[]
for case in gold:
    result=agent.run(case["task"],case["profile"])
    rows.append({"case_id":case["id"],"report":result["report"],"critic":result["critic"],
                 "invalid_citation_ids":result["invalid_citation_ids"]})
Path("data/answer_eval_outputs.json").write_text(json.dumps(rows,indent=2))
print("Generated outputs. Human-score correctness, citation support, completeness, fairness and abstention.")
