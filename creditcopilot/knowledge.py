"""Discover open scholarly documents through OpenAlex. Licence metadata must be verified at source."""
import os,re,json,hashlib,requests,fitz
from pathlib import Path

QUERIES=["credit card risk","credit card default","credit card acquisition","credit scoring",
"consumer credit","credit card delinquency","credit card portfolio management",
"payment behaviour","credit card fraud risk","responsible lending","consumer protection finance",
"banking regulation","financial inclusion","credit risk early warning","household debt"]

ALLOWED=("cc-by","cc0","cc-by-sa")

def discover(limit=700):
    results=[]; seen=set()
    for q in QUERIES:
        params={"search":q,"filter":"open_access.is_oa:true","per-page":100,
                "select":"id,title,doi,publication_year,locations"}
        try:
            data=requests.get("https://api.openalex.org/works",params=params,timeout=40).json()
        except Exception as e:
            print("OpenAlex discovery issue:",e); continue
        for item in data.get("results",[]):
            for loc in item.get("locations") or []:
                lic=(loc.get("license") or "").lower(); url=loc.get("pdf_url")
                if url and lic.startswith(ALLOWED) and url not in seen:
                    seen.add(url)
                    results.append({"title":item.get("title") or "Untitled","doi":item.get("doi"),
                                    "year":item.get("publication_year"),"url":url,"license":lic,"query":q})
                    break
            if len(results)>=limit: return results[:limit]
    return results[:limit]

def download_and_extract(target=700, folder="data/pdfs"):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    pages=[]; hashes=set(); manifest=[]
    candidates=discover(target)
    for i,m in enumerate(candidates):
        try:
            r=requests.get(m["url"],timeout=45,headers={"User-Agent":"CreditCardRiskResearchCopilot/1.0"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"): continue
            digest=hashlib.sha256(r.content).hexdigest()
            if digest in hashes: continue
            hashes.add(digest)
            name=re.sub(r"[^A-Za-z0-9_-]+","_",m["title"][:65]) or f"doc{i}"
            path=folder/f"{i:04d}_{name}.pdf"; path.write_bytes(r.content)
            m.update({"file":str(path),"sha256":digest}); manifest.append(m)
            doc=fitz.open(path)
            for pno,page in enumerate(doc,1):
                txt=page.get_text().strip()
                if len(txt)>100:
                    pages.append({"text":txt,"title":m["title"],"url":m["url"],
                                  "license":m["license"],"page":pno,"file":str(path)})
        except Exception as e: print("Skipped",m.get("title"),str(e)[:160])
    Path("data").mkdir(exist_ok=True)
    Path("data/manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Downloaded {len(manifest)} unique PDFs. Target: {target}.")
    if len(manifest)<500:
        print("WARNING: fewer than 500 eligible downloadable documents. Add verified licensed sources; do not fabricate count.")
    return pages,manifest
