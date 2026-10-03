#!/usr/bin/env python3
"""Fail-fast scientific and artifact validation for the completed run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"
checks={}
required=[ROOT/"REPORT.md",ROOT/"README.md",ROOT/"planning.md",R/"evaluated_generations.parquet",
          R/"endpoint_effects.csv",R/"dose_response_stats.csv",ROOT/"figures/dose_response_detector_ai.png",
          ROOT/"figures/dose_response_public_detector_ai.png",ROOT/"figures/control_effects.png"]
checks["required_files"]=all(x.exists() and x.stat().st_size>0 for x in required)

splits=[]
for tag in ["qwen25_7b_instruct","qwen25_7b_base"]:
    d=pd.read_parquet(R/tag/"hape_split.parquet"); splits.append(d)
    ids={s:set(d.loc[d.split==s,"base_id"]) for s in ["train","val","test"]}
    checks[f"{tag}_split_sizes"]=[len(ids[s]) for s in ["train","val","test"]]==[240,80,80]
    checks[f"{tag}_no_split_leakage"]=not (ids["train"]&ids["val"] or ids["train"]&ids["test"] or ids["val"]&ids["test"])
checks["same_checkpoint_split"]=splits[0][["base_id","split"]].equals(splits[1][["base_id","split"]])

d=pd.read_parquet(R/"evaluated_generations.parquet")
checks["generation_rows"]=len(d)==292
checks["prompt_counts"]=(d.query("checkpoint=='qwen25_7b_instruct'").base_id.nunique()==16 and d.query("checkpoint=='qwen25_7b_base'").base_id.nunique()==12)
checks["complete_metrics"]=not d[["detector_ai","public_detector_ai","perplexity","semantic_to_baseline"]].isna().any().any()
checks["score_ranges"]=(d.detector_ai.between(0,1).all() and d.public_detector_ai.between(0,1).all() and (d.perplexity>0).all())

z=np.load(R/"qwen25_7b_instruct/vectors.npz"); u=lambda x:x/np.linalg.norm(x)
checks["residual_orthogonal"]=(abs(float(u(z["residual"])@u(z["assistant"])))<1e-6 and abs(float(u(z["residual"])@u(z["formality"])))<1e-6)
norms=[np.linalg.norm(z[k]) for k in ["ai","residual","assistant","formality","random","shuffled"]]
checks["matched_vector_norms"]=bool(np.allclose(norms,norms[0],rtol=1e-5,atol=1e-6))

e=pd.read_csv(R/"endpoint_effects.csv")
checks["valid_confidence_intervals"]=bool(((e.ci_low<=e.mean_diff_plus2_minus2)&(e.mean_diff_plus2_minus2<=e.ci_high)).all())
report=(ROOT/"REPORT.md").read_text()
checks["report_sections"]=all(x in report for x in ["Executive Summary","Methodology","Results","Discussion","Limitations","Conclusions","References"])
checks={k:(bool(v) if isinstance(v,(bool,np.bool_)) else v) for k,v in checks.items()}
checks["all_passed"]=all(v is True for k,v in checks.items() if k!="all_passed")
(R/"validation.json").write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
if not checks["all_passed"]: raise SystemExit(1)
