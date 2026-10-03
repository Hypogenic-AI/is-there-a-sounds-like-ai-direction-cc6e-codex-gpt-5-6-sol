#!/usr/bin/env python3
"""Data-quality audit and descriptive figures for the local corpora."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results"; FIG=ROOT/"figures"
rows=[]
for p in sorted((ROOT/"datasets/human-ai-parallel-corpus/text_data").glob("*.parquet")):
    d=pd.read_parquet(p); wc=d.text.str.split().str.len()
    rows.append({"dataset":"HAP-E","file":p.name,"rows":len(d),"missing_text":int(d.text.isna().sum()),
                 "duplicate_ids":int(d.doc_id.duplicated().sum()),"mean_words":float(wc.mean()),
                 "median_words":float(wc.median()),"p95_words":float(wc.quantile(.95))})
for p in sorted((ROOT/"datasets/MAGE").glob("*.csv")):
    d=pd.read_csv(p); wc=d.text.fillna("").str.split().str.len()
    rows.append({"dataset":"MAGE","file":p.name,"rows":len(d),"missing_text":int(d.text.isna().sum()),
                 "duplicate_ids":int(d.text.duplicated().sum()),"mean_words":float(wc.mean()),
                 "median_words":float(wc.median()),"p95_words":float(wc.quantile(.95))})
audit=pd.DataFrame(rows); audit.to_csv(OUT/"data_audit.csv",index=False)
hape=audit[audit.dataset=="HAP-E"].sort_values("file")
fig,ax=plt.subplots(figsize=(10,4.5)); ax.barh(hape.file.str.replace("hape-text_","",regex=False).str.replace(".parquet","",regex=False),hape.mean_words)
ax.set(xlabel="Mean word count",title="HAP-E continuation length by author/model"); fig.tight_layout(); fig.savefig(FIG/"hape_lengths.png",dpi=180)
print(audit.to_string(index=False))
