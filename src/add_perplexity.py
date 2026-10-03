#!/usr/bin/env python3
"""Add base-Qwen continuation perplexity as a coherence/fluency proxy."""
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"results/evaluated_generations.parquet"
d=pd.read_parquet(path)
model_path=ROOT/"artifacts/models/Qwen2.5-7B"
tok=AutoTokenizer.from_pretrained(model_path,padding_side="right")
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map="cuda",low_cpu_mem_usage=True).eval()
ppls=[]
with torch.inference_mode():
    for st in range(0,len(d),8):
        x=tok(d.generation.iloc[st:st+8].tolist(),return_tensors="pt",padding=True,truncation=True,max_length=192).to("cuda")
        logits=model(**x,use_cache=False).logits[:,:-1].float()
        labels=x.input_ids[:,1:]; mask=x.attention_mask[:,1:].float()
        loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,reduction="none")
        nll=(loss*mask).sum(1)/mask.sum(1).clamp_min(1)
        ppls.extend(torch.exp(nll.clamp(max=20)).cpu().tolist())
d["perplexity"]=ppls
d.to_parquet(path,index=False)
print(d.groupby(["checkpoint","condition","strength"]).perplexity.mean().to_string())
