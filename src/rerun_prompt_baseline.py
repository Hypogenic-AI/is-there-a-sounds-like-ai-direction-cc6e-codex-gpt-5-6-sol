#!/usr/bin/env python3
"""Regenerate only the corrected prompt baseline after truncation-order validation."""
from pathlib import Path
import pandas as pd
from src.experiment import load_model, generate_one

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/"results/qwen25_7b_instruct"
d=pd.read_parquet(out/"generations.parquet")
split=pd.read_parquet(out/"hape_split.parquet").query("split=='test'").head(16)
layer=int(__import__('numpy').load(out/"vectors.npz")["layer"][0])
model,tok=load_model(ROOT/"artifacts/models/Qwen2.5-7B-Instruct")
new=[]
for _,r in split.iterrows():
    prompt="Write in a natural, personal human style. Avoid polished assistant-like phrasing.\n\n"+r.prefix
    text=generate_one(model,tok,prompt,layer,None,0,True,80)
    new.append({"base_id":r.base_id,"domain":r.domain,"condition":"prompt_human","strength":0.,"prompt":r.prefix,"generation":text})
d=d[d.condition!="prompt_human"]
d=pd.concat([d,pd.DataFrame(new)],ignore_index=True)
d.to_parquet(out/"generations.parquet",index=False)
print("corrected prompt rows",len(new),"total",len(d))
