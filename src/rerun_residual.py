#!/usr/bin/env python3
"""Regenerate instruct residual controls with joint-span orthogonalization."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.experiment import load_model, generate_one, orthogonalize

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/"results/qwen25_7b_instruct"
d=pd.read_parquet(out/"generations.parquet")
split=pd.read_parquet(out/"hape_split.parquet").query("split=='test'").head(16)
z=np.load(out/"vectors.npz"); layer=int(z["layer"][0])
residual=orthogonalize(z["ai"],[z["assistant"],z["formality"]])
payload={k:z[k] for k in z.files}; payload["residual"]=residual; np.savez(out/"vectors.npz",**payload)
model,tok=load_model(ROOT/"artifacts/models/Qwen2.5-7B-Instruct")
new=[]
for _,r in split.iterrows():
    for s in [-2.,2.]:
        text=generate_one(model,tok,r.prefix,layer,residual,s,True,80)
        new.append({"base_id":r.base_id,"domain":r.domain,"condition":"residual","strength":s,"prompt":r.prefix,"generation":text})
d=d[d.condition!="residual"]
d=pd.concat([d,pd.DataFrame(new)],ignore_index=True); d.to_parquet(out/"generations.parquet",index=False)
u=lambda x:x/np.linalg.norm(x)
print("residual rows",len(new),"cos assistant",u(residual)@u(z["assistant"]),"cos formality",u(residual)@u(z["formality"]))
