#!/usr/bin/env python3
"""Residual-stream AI-direction experiment.

The script is intentionally stage-based: expensive activations and generations are
checkpointed, while analysis can be rerun without loading the language model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"
SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def base_id(s: str) -> str:
    return s.split("@")[0]


def load_hape(ai_file: str) -> pd.DataFrame:
    """Join HAP-E shared prefixes, human continuations, and one AI continuation."""
    folder = DATA / "human-ai-parallel-corpus" / "text_data"
    frames = {}
    for key, name in {
        "prefix": "hape-text_human-chunk-1.parquet",
        "human": "hape-text_human-chunk-2.parquet",
        "ai": ai_file,
    }.items():
        d = pd.read_parquet(folder / name)
        d["base_id"] = d.doc_id.map(base_id)
        frames[key] = d[["base_id", "text"]].rename(columns={"text": key})
    out = frames["prefix"].merge(frames["human"], on="base_id").merge(frames["ai"], on="base_id")
    out["domain"] = out.base_id.str.split("_").str[0]
    return out


def split_hape(df: pd.DataFrame, n_train=240, n_val=80, n_test=80) -> dict[str, pd.DataFrame]:
    """Deterministic domain-stratified split with disjoint document IDs."""
    rng = np.random.default_rng(SEED)
    parts = []
    total = n_train + n_val + n_test
    for _, g in df.groupby("domain"):
        take = max(1, round(total * len(g) / len(df)))
        parts.append(g.iloc[rng.permutation(len(g))[:take]])
    sample = pd.concat(parts).drop_duplicates("base_id")
    if len(sample) < total:
        rest = df.loc[~df.base_id.isin(sample.base_id)]
        sample = pd.concat([sample, rest.iloc[rng.permutation(len(rest))[:total-len(sample)] ]])
    sample = sample.iloc[rng.permutation(len(sample))[:total]].reset_index(drop=True)
    # Global shuffle keeps allocation deterministic; report domain mix afterward.
    return {"train": sample.iloc[:n_train].copy(),
            "val": sample.iloc[n_train:n_train+n_val].copy(),
            "test": sample.iloc[n_train+n_val:total].copy()}


def model_layers(model):
    """Return decoder block list for Qwen/Llama-like Transformers models."""
    for path in [("model", "layers"), ("transformer", "h")]:
        obj = model
        try:
            for p in path: obj = getattr(obj, p)
            return obj
        except AttributeError:
            pass
    raise AttributeError("Unsupported decoder layout")


class Capture:
    """Capture attention-mask-weighted layer output means without retaining graphs."""
    def __init__(self, layers, indices):
        self.mask = None; self.data = {i: [] for i in indices}; self.handles = []
        for i in indices:
            self.handles.append(layers[i].register_forward_hook(self._hook(i)))
    def _hook(self, idx):
        def fn(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            mask = self.mask[:, :x.shape[1]].to(x.device).unsqueeze(-1)
            pooled = (x.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
            self.data[idx].append(pooled.detach().cpu().numpy())
        return fn
    def close(self):
        for h in self.handles: h.remove()


@torch.inference_mode()
def extract(model, tokenizer, texts, layer_ids, batch_size=2, max_length=384):
    cap = Capture(model_layers(model), layer_ids)
    for start in range(0, len(texts), batch_size):
        toks = tokenizer(texts[start:start+batch_size], return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(model.device)
        cap.mask = toks["attention_mask"]
        model(**toks, use_cache=False)
    cap.close()
    return {i: np.concatenate(cap.data[i]) for i in layer_ids}


def fit_directions(features: dict[int, np.ndarray], labels: np.ndarray,
                   pair_count: int, seed=SEED):
    """Fit logistic, paired mean-difference, and shuffled-label directions."""
    out = {}
    rng = np.random.default_rng(seed)
    for layer, x in features.items():
        clf = LogisticRegression(C=1, max_iter=2000, random_state=seed).fit(x, labels)
        logistic = clf.coef_[0]
        paired = x[pair_count:].mean(0) - x[:pair_count].mean(0)
        shuffled = labels.copy(); rng.shuffle(shuffled)
        shuf = LogisticRegression(C=1, max_iter=2000, random_state=seed).fit(x, shuffled).coef_[0]
        out[layer] = {"logistic": logistic, "paired": paired, "shuffled": shuf, "clf": clf}
    return out


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def orthogonalize(v, axes):
    # Project off the joint nuisance span. Sequential subtraction is insufficient
    # when nuisance axes are correlated (as Assistant and formality are here).
    a = np.column_stack([unit(x) for x in axes])
    q, _ = np.linalg.qr(a)
    z = v - q @ (q.T @ v)
    return unit(z) * np.linalg.norm(v)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt, layer_idx, vector=None, strength=0.0,
                 instruct=True, max_new_tokens=96):
    """Greedy generation with intervention only on newly decoded token states."""
    if instruct:
        msgs = [{"role":"user", "content": "Continue the passage naturally and only provide the continuation.\n\nPASSAGE:\n" + prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(model.device)
    handle = None
    if vector is not None and strength != 0:
        steer = torch.as_tensor(vector * strength, device=model.device, dtype=model.dtype)
        def hook(_m, _inp, output):
            x = output[0] if isinstance(output, tuple) else output
            if x.shape[1] == 1:  # cached decoding step: generation token only
                y = x + steer
                return (y,) + output[1:] if isinstance(output, tuple) else y
            return output
        handle = model_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        seq = model.generate(**toks, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, use_cache=True)
    finally:
        if handle: handle.remove()
    return tokenizer.decode(seq[0, toks.input_ids.shape[1]:], skip_special_tokens=True).strip()


def build_axis(model, tokenizer, layer, kind, instruct):
    topics = [
        "Explain why leaves change color.", "Describe a good morning routine.",
        "Discuss whether cities need more parks.", "Explain how bread rises.",
        "Give advice for learning a language.", "Describe a rainy afternoon.",
        "Explain why sleep matters.", "Discuss the value of public libraries.",
    ]
    if kind == "assistant":
        neg = ["Write a spontaneous first-person diary note about: " + t for t in topics]
        pos = ["Give a polished, helpful AI assistant response to: " + t for t in topics]
    else:
        neg = ["Answer in very casual slang, like a quick text to a friend: " + t for t in topics]
        pos = ["Answer in highly formal professional prose: " + t for t in topics]
    ng = [generate_one(model, tokenizer, x, layer, instruct=instruct, max_new_tokens=64) for x in neg]
    pg = [generate_one(model, tokenizer, x, layer, instruct=instruct, max_new_tokens=64) for x in pos]
    fx = extract(model, tokenizer, ng + pg, [layer], batch_size=2, max_length=160)[layer]
    return fx[len(ng):].mean(0) - fx[:len(ng)].mean(0), {"negative": ng, "positive": pg}


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def run_checkpoint(args):
    seed_all(); RESULTS.mkdir(exist_ok=True, parents=True)
    outdir = RESULTS / args.tag; outdir.mkdir(exist_ok=True)
    df = load_hape(args.ai_file); splits = split_hape(df, args.n_train, args.n_val, args.n_test)
    pd.concat([g.assign(split=k) for k,g in splits.items()]).to_parquet(outdir/"hape_split.parquet")
    model, tok = load_model(args.model)
    nl = len(model_layers(model)); layer_ids = sorted(set([max(0, round(x*(nl-1))) for x in (.2,.35,.5,.65,.8)]))
    all_features = {}
    for split, g in splits.items():
        texts = g.human.tolist() + g.ai.tolist()
        feats = extract(model, tok, texts, layer_ids, args.batch_size)
        np.savez_compressed(outdir/f"activations_{split}.npz", **{str(k):v for k,v in feats.items()})
        all_features[split] = feats
    y = {k:np.r_[np.zeros(len(g)),np.ones(len(g))] for k,g in splits.items()}
    dirs = fit_directions(all_features["train"], y["train"], len(splits["train"]))
    rows=[]
    for layer,d in dirs.items():
        for split in ("train","val","test"):
            score=all_features[split][layer]@unit(d["logistic"])
            rows.append({"layer":layer,"split":split,"auroc":roc_auc_score(y[split],score),
                         "accuracy":accuracy_score(y[split],d["clf"].predict(all_features[split][layer]))})
    metrics=pd.DataFrame(rows); metrics.to_csv(outdir/"readout_metrics.csv",index=False)
    best=int(metrics.query("split=='val'").sort_values("auroc",ascending=False).iloc[0].layer)
    ai_vec=dirs[best]["logistic"]
    # Give the unit direction the empirical paired-difference magnitude for interpretable alpha.
    ai_vec=unit(ai_vec)*np.linalg.norm(dirs[best]["paired"])
    assist, assist_text=build_axis(model,tok,best,"assistant",args.instruct)
    formal, formal_text=build_axis(model,tok,best,"formality",args.instruct)
    residual=orthogonalize(ai_vec,[assist,formal])
    rng=np.random.default_rng(SEED); random_vec=unit(rng.normal(size=len(ai_vec)))*np.linalg.norm(ai_vec)
    shuffled=unit(dirs[best]["shuffled"])*np.linalg.norm(ai_vec)
    np.savez(outdir/"vectors.npz",ai=ai_vec,residual=residual,assistant=unit(assist)*np.linalg.norm(ai_vec),
             formality=unit(formal)*np.linalg.norm(ai_vec),random=random_vec,shuffled=shuffled,
             layer=np.array([best]))
    (outdir/"axis_generations.json").write_text(json.dumps(assist_text|{"formality":formal_text},indent=2))
    cos=lambda a,b:float(np.dot(unit(a),unit(b)))
    json.dump({"best_layer":best,"n_layers":nl,"ai_norm":float(np.linalg.norm(ai_vec)),
               "cos_ai_assistant":cos(ai_vec,assist),"cos_ai_formality":cos(ai_vec,formal),
               "cos_assistant_formality":cos(assist,formal)},open(outdir/"direction_geometry.json","w"),indent=2)

    # Causal prompts are held-out prefixes only, truncated by the tokenizer in generate_one.
    prompts=splits["test"].head(args.n_prompts)
    strengths=[-2.,-1.,0.,1.,2.] if args.full_conditions else [-2.,0.,2.]
    conditions=[]
    for s in strengths: conditions.append(("ai",s,ai_vec))
    if args.full_conditions:
        for name,v in [("residual",residual),("random",random_vec),("shuffled",shuffled),
                       ("formality",unit(formal)*np.linalg.norm(ai_vec)),
                       ("assistant",unit(assist)*np.linalg.norm(ai_vec))]:
            for s in [-2.,2.]: conditions.append((name,s,v))
    generations=[]
    start=time.time()
    for _,row in prompts.iterrows():
        for name,s,v in conditions:
            text=generate_one(model,tok,row.prefix,best,v,s,args.instruct,args.max_new_tokens)
            generations.append({"base_id":row.base_id,"domain":row.domain,"condition":name,
                                "strength":s,"prompt":row.prefix,"generation":text})
        if args.full_conditions:
            # Put the behavioral instruction before the long HAP-E prefix so truncation
            # cannot silently remove it (a validation run caught the opposite ordering).
            human_prompt="Write in a natural, personal human style. Avoid polished assistant-like phrasing.\n\n" + row.prefix
            text=generate_one(model,tok,human_prompt,best,None,0,args.instruct,args.max_new_tokens)
            generations.append({"base_id":row.base_id,"domain":row.domain,"condition":"prompt_human",
                                "strength":0.,"prompt":row.prefix,"generation":text})
        pd.DataFrame(generations).to_parquet(outdir/"generations.parquet",index=False)
    json.dump({"seconds":time.time()-start,"model":args.model,"dtype":"bfloat16","batch_size":args.batch_size,
               "max_new_tokens":args.max_new_tokens,"seed":SEED,"instruct":args.instruct},
              open(outdir/"run_metadata.json","w"),indent=2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True); p.add_argument("--tag",required=True)
    p.add_argument("--ai-file",default="hape-text_llama-3-8B-Instruct.parquet")
    p.add_argument("--instruct",action="store_true"); p.add_argument("--full-conditions",action="store_true")
    p.add_argument("--n-train",type=int,default=240); p.add_argument("--n-val",type=int,default=80)
    p.add_argument("--n-test",type=int,default=80); p.add_argument("--n-prompts",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--max-new-tokens",type=int,default=96)
    run_checkpoint(p.parse_args())

if __name__ == "__main__": main()
