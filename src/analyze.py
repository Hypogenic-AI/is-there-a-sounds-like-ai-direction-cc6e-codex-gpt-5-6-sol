#!/usr/bin/env python3
"""Final paired statistical analysis from cached experimental outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower
from src.evaluate import endpoint_effects, boot_ci, signflip_p, make_figures

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"
d=pd.read_parquet(R/"evaluated_generations.parquet")
effects=endpoint_effects(d); effects.to_csv(R/"endpoint_effects.csv",index=False)

summary=d.groupby(["checkpoint","condition","strength"]).agg(
    n=("base_id","size"), detector_ai_mean=("detector_ai","mean"), detector_ai_sd=("detector_ai","std"),
    public_ai_mean=("public_detector_ai","mean"), public_ai_sd=("public_detector_ai","std"),
    perplexity_mean=("perplexity","mean"), words_mean=("word_count","mean"),
    repetition_mean=("repeat_3gram","mean"), semantic_mean=("semantic_to_baseline","mean")).reset_index()
summary.to_csv(R/"condition_summary.csv",index=False)

# Per-prompt linear slopes across the full five-dose raw AI sweep.
rows=[]
for checkpoint,g in d[d.condition=="ai"].groupby("checkpoint"):
    for metric in ["detector_ai","public_detector_ai","perplexity","word_count","repeat_3gram"]:
        slopes=[]
        for _,h in g.groupby("base_id"):
            if h.strength.nunique()>=3: slopes.append(np.polyfit(h.strength,h[metric],1)[0])
        if not slopes: continue
        means=g.groupby("strength")[metric].mean().sort_index()
        rho,p_rho=stats.spearmanr(means.index,means.values)
        ci=boot_ci(slopes)
        rows.append({"checkpoint":checkpoint,"metric":metric,"n_prompts":len(slopes),
                     "mean_within_prompt_slope":float(np.mean(slopes)),"ci_low":ci[0],"ci_high":ci[1],
                     "p_signflip":signflip_p(slopes),"aggregate_spearman_rho":float(rho),"spearman_p":float(p_rho)})
pd.DataFrame(rows).to_csv(R/"dose_response_stats.csv",index=False)

# Explicit prompt-only comparison against unsteered generation.
ig=d[d.checkpoint.eq("qwen25_7b_instruct")]
prompt_rows=[]
for metric in ["detector_ai","public_detector_ai","perplexity","word_count","repeat_3gram","semantic_to_baseline"]:
    a=ig[(ig.condition=="ai")&(ig.strength==0)].set_index("base_id")[metric]
    b=ig[ig.condition=="prompt_human"].set_index("base_id")[metric]
    x=(b-a).dropna().values; ci=boot_ci(x)
    prompt_rows.append({"metric":metric,"n":len(x),"prompt_human_minus_unsteered":float(x.mean()),
                        "ci_low":ci[0],"ci_high":ci[1],"p_signflip":signflip_p(x)})
pd.DataFrame(prompt_rows).to_csv(R/"prompt_baseline_effects.csv",index=False)

# Sensitivity, not post-hoc observed power: minimum paired dz detectable at 80% power.
sens={str(n):float(TTestPower().solve_power(nobs=n,alpha=.05,power=.8,alternative="two-sided")) for n in [12,16]}
json.dump({"paired_dz_80pct_power":sens,"alpha":.05,"note":"two-sided sensitivity analysis"},open(R/"power_sensitivity.json","w"),indent=2)

# Representative largest raw-vector changes for transparent error inspection.
raw=ig[ig.condition.eq("ai") & ig.strength.isin([-2,2])]
w=raw.pivot(index="base_id",columns="strength",values=["detector_ai","public_detector_ai","generation"])
lexical_delta=w[("detector_ai",2)]-w[("detector_ai",-2)]
public_delta=w[("public_detector_ai",2)]-w[("public_detector_ai",-2)]
lines=["# Endpoint error analysis",""]
for bid,row in w.reindex(lexical_delta.abs().sort_values(ascending=False).index[:6]).iterrows():
    lines += [f"## {bid}",f"MAGE-detector Δ={lexical_delta[bid]:.3f}; public-RoBERTa Δ={public_delta[bid]:.3f}",
              f"- α=-2: {str(row[('generation',-2)])[:500]}",f"- α=+2: {str(row[('generation',2)])[:500]}",""]
(R/"error_analysis.md").write_text("\n".join(lines))
make_figures(d,effects)
print(pd.DataFrame(rows).to_string(index=False))
