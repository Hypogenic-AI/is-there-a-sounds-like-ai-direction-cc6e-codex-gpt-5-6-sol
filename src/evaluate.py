#!/usr/bin/env python3
"""Independent detectors, blinded LLM judging, statistics, and figures."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import FeatureUnion
from statsmodels.stats.multitest import multipletests
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/"results"; DATA=ROOT/"datasets"; FIG=ROOT/"figures"
SEED=42


def train_detector():
    """Train a lexical detector on MAGE only; HAP-E never enters its training."""
    rng=np.random.default_rng(SEED)
    train=pd.read_csv(DATA/"MAGE"/"train.csv").dropna(subset=["text","src"])
    train["machine"]=(~train.src.str.lower().str.contains("human")).astype(int)
    chunks=[]
    for label,g in train.groupby("machine"):
        chunks.append(g.iloc[rng.permutation(len(g))[:12000]])
    fit=pd.concat(chunks).iloc[rng.permutation(24000)]
    features=FeatureUnion([
        ("word",TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=35000,sublinear_tf=True)),
        ("char",TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=4,max_features=25000,sublinear_tf=True)),
    ])
    x=features.fit_transform(fit.text)
    clf=LogisticRegression(C=2,max_iter=1000,random_state=SEED,n_jobs=-1).fit(x,fit.machine)
    # Held-out in-domain and RAID no-attack validation.
    test=pd.read_csv(DATA/"MAGE"/"test.csv").dropna(subset=["text","src"])
    test["machine"]=(~test.src.str.lower().str.contains("human")).astype(int)
    test=test.iloc[rng.permutation(len(test))[:12000]]
    raid=pd.read_parquet(DATA/"raid"/"train_sample.parquet")
    raid=raid.query("attack=='none'").dropna(subset=["generation"])
    raid["machine"]=(raid.model!="human").astype(int)
    vals={}
    for name,d,col in [("mage_test",test,"text"),("raid_sample",raid,"generation")]:
        pred=clf.predict_proba(features.transform(d[col]))[:,1]
        vals[name]={"n":len(d),"auroc":float(roc_auc_score(d.machine,pred)),
                    "accuracy":float(accuracy_score(d.machine,pred>=.5)),
                    "prevalence_machine":float(d.machine.mean())}
    return features,clf,vals


def text_metrics(s):
    words=str(s).split(); n=len(words)
    return pd.Series({"word_count":n,"unique_ratio":len(set(w.lower() for w in words))/max(n,1),
                      "repeat_3gram":1-len(set(zip(words,words[1:],words[2:])))/max(n-2,1)})


@torch.inference_mode()
def public_detector_scores(texts, path=ROOT/"artifacts/models/roberta-base-openai-detector"):
    """Score with OpenAI's GPT-2 RoBERTa detector (label 0/Fake = machine)."""
    tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForSequenceClassification.from_pretrained(path,dtype=torch.float16).cuda().eval()
    scores=[]
    for st in range(0,len(texts),32):
        x=tok(list(texts[st:st+32]),return_tensors="pt",padding=True,truncation=True,max_length=510).to("cuda")
        scores.extend(model(**x).logits.softmax(-1)[:,0].float().cpu().tolist())
    del model; torch.cuda.empty_cache()
    return np.asarray(scores)


def score_generations(tags):
    vec,clf,valid=train_detector(); json.dump(valid,open(RESULTS/"detector_validation.json","w"),indent=2)
    # Validate the second, independently trained detector on fixed subsets.
    rng=np.random.default_rng(SEED)
    mt=pd.read_csv(DATA/"MAGE"/"test.csv").dropna(subset=["text","src"])
    mt["machine"]=(~mt.src.str.lower().str.contains("human")).astype(int)
    mt=mt.iloc[rng.permutation(len(mt))[:1000]]
    raid=pd.read_parquet(DATA/"raid"/"train_sample.parquet").query("attack=='none'").dropna(subset=["generation"])
    raid["machine"]=(raid.model!="human").astype(int)
    for name,dd,col in [("public_roberta_mage",mt,"text"),("public_roberta_raid",raid,"generation")]:
        pr=public_detector_scores(dd[col].tolist())
        valid[name]={"n":len(dd),"auroc":float(roc_auc_score(dd.machine,pr)),
                     "accuracy":float(accuracy_score(dd.machine,pr>=.5)),"prevalence_machine":float(dd.machine.mean())}
    json.dump(valid,open(RESULTS/"detector_validation.json","w"),indent=2)
    frames=[]
    for tag in tags:
        d=pd.read_parquet(RESULTS/tag/"generations.parquet"); d["checkpoint"]=tag
        d["detector_ai"]=clf.predict_proba(vec.transform(d.generation.fillna("")))[:,1]
        d=pd.concat([d,d.generation.apply(text_metrics)],axis=1); frames.append(d)
    out=pd.concat(frames,ignore_index=True)
    out["public_detector_ai"]=public_detector_scores(out.generation.fillna("").tolist())
    out.to_parquet(RESULTS/"scored_generations.parquet",index=False)
    return out


JUDGE_SYSTEM="""You are a blinded evaluator of text continuations. Score each continuation independently.
AI_LIKENESS: 0 definitely naturally human-written, 100 definitely sounds machine-generated. Judge surface writing style, not topic.
COHERENCE: 0 incoherent/broken, 100 fluent and internally coherent.
RELEVANCE: 0 unrelated/contradictory to passage, 100 a natural content continuation.
Return strict JSON: {\"items\":[{\"id\":integer,\"ai_likeness\":integer,\"coherence\":integer,\"relevance\":integer}]}.
Do not infer experimental condition; none is provided."""


@retry(wait=wait_exponential(min=1,max=20),stop=stop_after_attempt(5))
def judge_call(client,items,model):
    payload=[]
    for i,r in items:
        payload.append({"id":int(i),"passage":str(r.prompt)[-1800:],"continuation":str(r.generation)[:1600]})
    resp=client.chat.completions.create(model=model,temperature=0,seed=SEED,max_tokens=1200,
        response_format={"type":"json_object"},messages=[{"role":"system","content":JUDGE_SYSTEM},
        {"role":"user","content":json.dumps(payload,ensure_ascii=False)}])
    return json.loads(resp.choices[0].message.content), resp.usage


def run_judge(d,model="openai/gpt-5.6-luna",batch=6):
    cache=RESULTS/"judge_scores.jsonl"; done={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            x=json.loads(line); done[int(x["id"])]=x
    client=OpenAI(api_key=os.environ["OPENROUTER_KEY"],base_url="https://openrouter.ai/api/v1")
    usage={"prompt_tokens":0,"completion_tokens":0,"calls":0,"model":model}
    ids=[i for i in d.index if i not in done]
    random.Random(SEED).shuffle(ids)
    for st in range(0,len(ids),batch):
        group=[(i,d.loc[i]) for i in ids[st:st+batch]]
        obj,u=judge_call(client,group,model)
        got={int(x["id"]):x for x in obj["items"]}
        for i,_ in group:
            if i not in got: raise ValueError(f"Judge omitted id {i}")
            done[i]=got[i]
            with cache.open("a") as f: f.write(json.dumps(got[i])+"\n")
        usage["prompt_tokens"]+=getattr(u,"prompt_tokens",0); usage["completion_tokens"]+=getattr(u,"completion_tokens",0); usage["calls"]+=1
        time.sleep(.15)
    json.dump(usage,open(RESULTS/"judge_usage.json","w"),indent=2)
    scores=pd.DataFrame.from_dict(done,orient="index").drop(columns="id",errors="ignore")
    for c in scores: scores[c]=pd.to_numeric(scores[c]).clip(0,100)
    return d.join(scores)


def semantic_scores(d):
    """Cosine similarity to each prompt's unsteered generation (descriptive)."""
    try:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cuda")
        emb=model.encode(d.generation.tolist(),batch_size=64,normalize_embeddings=True,show_progress_bar=False)
        d=d.copy(); d["semantic_to_baseline"]=np.nan
        for (_,bid),idx in d.groupby(["checkpoint","base_id"]).groups.items():
            ids=list(idx); base=[i for i in ids if d.loc[i,"condition"]=="ai" and d.loc[i,"strength"]==0]
            if base:
                d.loc[ids,"semantic_to_baseline"]=emb[ids]@emb[base[0]]
        return d
    except Exception as e:
        (RESULTS/"semantic_error.txt").write_text(repr(e)); d["semantic_to_baseline"]=np.nan; return d


def signflip_p(x,n_perm=20000):
    x=np.asarray(x,float); obs=abs(x.mean()); rng=np.random.default_rng(SEED)
    null=np.array([(x*rng.choice([-1,1],len(x))).mean() for _ in range(n_perm)])
    return float((1+(np.abs(null)>=obs).sum())/(n_perm+1))


def boot_ci(x,n=5000):
    x=np.asarray(x,float); rng=np.random.default_rng(SEED)
    means=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def endpoint_effects(d):
    rows=[]
    metrics=[c for c in ["detector_ai","public_detector_ai","ai_likeness","coherence","relevance","word_count","repeat_3gram","semantic_to_baseline","perplexity"] if c in d]
    for (checkpoint,cond),g in d.groupby(["checkpoint","condition"]):
        if not {-2.,2.}.issubset(set(g.strength)): continue
        wide=g[g.strength.isin([-2.,2.])].pivot(index="base_id",columns="strength",values=metrics)
        for metric in metrics:
            try: diff=(wide[metric][2.]-wide[metric][-2.]).dropna().values
            except KeyError: continue
            if not len(diff): continue
            rows.append({"checkpoint":checkpoint,"condition":cond,"metric":metric,"n":len(diff),
                         "mean_diff_plus2_minus2":float(diff.mean()),"ci_low":boot_ci(diff)[0],"ci_high":boot_ci(diff)[1],
                         "sd_diff":float(diff.std(ddof=1)) if len(diff)>1 else np.nan,
                         "cohens_dz":float(diff.mean()/diff.std(ddof=1)) if len(diff)>1 and diff.std(ddof=1)>0 else np.nan,
                         "p_perm":signflip_p(diff)})
    out=pd.DataFrame(rows)
    # Holm within the two primary AI-likeness outcomes across causal controls/checkpoints.
    mask=out.metric.isin(["detector_ai","public_detector_ai"])
    if mask.any(): out.loc[mask,"p_holm"]=multipletests(out.loc[mask,"p_perm"],method="holm")[1]
    return out


def make_figures(d,effects):
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    for metric,label in [("detector_ai","MAGE lexical detector P(AI)"),("public_detector_ai","Public RoBERTa detector P(AI)")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        sub=d[d.condition=="ai"]
        for checkpoint,g in sub.groupby("checkpoint"):
            z=g.groupby("strength")[metric].agg(["mean","sem"]).reset_index()
            ax.errorbar(z.strength,z["mean"],yerr=1.96*z["sem"],marker="o",capsize=3,label=checkpoint)
        ax.set(xlabel="Steering strength α",ylabel=label,title="Signed AI-direction dose response")
        ax.axvline(0,color="grey",lw=.8); ax.legend(); fig.tight_layout(); fig.savefig(FIG/f"dose_response_{metric}.png",dpi=180); plt.close(fig)
    sub=effects.query("metric=='detector_ai'").copy()
    if len(sub):
        fig,ax=plt.subplots(figsize=(8,4.8)); labels=sub.checkpoint+" / "+sub.condition
        y=np.arange(len(sub)); ax.errorbar(sub.mean_diff_plus2_minus2,y,
            xerr=[sub.mean_diff_plus2_minus2-sub.ci_low,sub.ci_high-sub.mean_diff_plus2_minus2],fmt="o",capsize=3)
        ax.set_yticks(y,labels); ax.axvline(0,color="black",lw=.8); ax.set_xlabel("P(AI) difference: α=+2 minus α=−2")
        ax.set_title("Specificity controls with prompt-cluster bootstrap 95% CIs"); fig.tight_layout(); fig.savefig(FIG/"control_effects.png",dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--tags",nargs="+",required=True); p.add_argument("--skip-judge",action="store_true")
    args=p.parse_args(); d=score_generations(args.tags)
    if not args.skip_judge: d=run_judge(d)
    d=semantic_scores(d); d.to_parquet(RESULTS/"evaluated_generations.parquet",index=False)
    effects=endpoint_effects(d); effects.to_csv(RESULTS/"endpoint_effects.csv",index=False)
    summaries=d.groupby(["checkpoint","condition","strength"]).agg(
        n=("base_id","size"),detector_ai_mean=("detector_ai","mean"),detector_ai_sd=("detector_ai","std"),
        public_ai_mean=("public_detector_ai","mean"),words_mean=("word_count","mean"),
        repetition_mean=("repeat_3gram","mean"),semantic_mean=("semantic_to_baseline","mean")).reset_index()
    summaries.to_csv(RESULTS/"condition_summary.csv",index=False); make_figures(d,effects)

if __name__=="__main__": main()
