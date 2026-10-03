# Phase 0–1: Motivation, Novelty, and Preregistered Plan

## Motivation & Novelty Assessment

### Why This Research Matters

A causally steerable “sounds like AI” representation would connect machine-text detection to the mechanisms that produce detectable prose, with implications for detector robustness, evasion, and interpretability. Conversely, a clean negative result would show that linearly readable authorship information can be merely diagnostic rather than a generative control variable.

### Gap in Existing Work

Quaremba et al. and SV-Detect establish residual-stream linear readout; Kuznetsov et al. causally steer selected SAE style features; and Lu et al. establish a causal Assistant persona axis. None performs the decisive matched experiment: fit a human-versus-machine direction, intervene on it during free generation, score outputs with independent detectors, preserve content/coherence, and test whether effects survive persona/style controls.

### Our Novel Contribution

We test the dense readout direction as a causal intervention with a signed dose response, equal-norm random and confound controls, a prompt-only baseline, and two independent outcome measures. We separately test raw and nuisance-orthogonal directions and compare pretrained base versus instruction-tuned checkpoints; claims are limited to the tested Qwen family and layer/sample budget.

### Experiment Justification

- **Experiment 1 — readout/localization:** required as a manipulation check and to identify layers without assuming that published Llama layer bands transfer to Qwen.
- **Experiment 2 — causal dose response:** directly tests whether adding/removing the fitted direction changes independent AI-likeness scores for the same prompts.
- **Experiment 3 — specificity/disentanglement:** random, shuffled-label, formality, Assistant-persona, and prompt-only controls test alternative causal explanations.
- **Experiment 4 — base versus instruct:** tests whether a comparable direction exists before chat tuning and whether tuning amplifies its readout or causal effect.

## Research Question

Does a residual-stream direction fitted to distinguish content-matched human and machine continuations causally change how AI-like new generations appear to independent evaluators while preserving meaning and coherence, and is any effect distinct from formality and Assistant persona?

## Hypothesis Decomposition and Variables

- **H1 (readout):** a linear direction decodes authorship above chance on held-out document IDs and transfers across generator/domain subsets.
- **H2 (causality):** signed intervention strength has a monotonic, direction-specific effect on independent detector scores.
- **H3 (preservation):** a detector shift occurs before material degradation in coherence, prompt relevance, repetition, or length.
- **H4 (distinctness):** the effect remains for a direction orthogonalized against measured formality and same-model Assistant-persona axes.
- **H5 (post-training):** instruction tuning increases readout separability and/or causal leverage relative to the base model.

Independent variables are checkpoint, layer, vector estimator/control, and signed steering coefficient. Dependent variables are held-out AUROC, detector scores, judge AI-likeness/coherence/relevance, semantic preservation, length, and repetition. Prompt/document ID is the paired experimental unit.

## Proposed Methodology

### Approach

Use HAP-E shared-prefix human and model continuations with document-level splits to fit per-layer mean-difference and logistic directions inside Qwen2.5-7B and Qwen2.5-7B-Instruct. Choose layers using validation data only. Generate from held-out prefixes under raw, orthogonalized, random, shuffled-label, and prompt-only conditions. Evaluate using a MAGE-trained TF–IDF detector and a live-catalog-verified OpenRouter judge, neither of which uses the fitted activations.

### Experimental Steps

1. Validate and stratify HAP-E; fit an independent MAGE lexical detector.
2. Extract mean-pooled hidden states from paired continuations at a preregistered sparse layer grid and fit directions on training document IDs.
3. Select the best layer by validation AUROC; reserve test IDs for a single confirmatory readout.
4. Estimate same-model formality and Assistant axes from contrastive prompts; residualize the AI direction and renormalize.
5. Generate paired continuations at signed strengths, including equal-norm random/shuffled/confound controls and “write like a human” prompt baseline.
6. Score every output using the lexical detector, a blinded LLM judge, and quality/content proxies; run paired clustered bootstrap/permutation analyses.
7. Repeat a reduced strongest-setting protocol on the base checkpoint.

### Baselines

No intervention; equal-norm isotropic random vectors; shuffled-label authorship vector; formality-axis steering; Assistant-axis steering; explicit “write like a human” prompting. The primary comparison uses identical prompts and decoding parameters.

### Evaluation Metrics

Readout AUROC and accuracy; independent detector probability; LLM-judge 0–100 AI-likeness, coherence, and relevance; generation length; distinct-token repetition; and semantic similarity to the unsteered output. Direction overlaps are cosine similarities. A causal result requires a signed monotonic detector response, specificity over random/shuffled controls, and matched-quality support.

### Statistical Analysis Plan

The primary estimand is the within-prompt slope of detector score versus signed strength. Report cluster bootstrap 95% CIs over prompt IDs, paired permutation p-values, standardized paired effects, and Holm correction across primary direction/control contrasts. Spearman dose-response is descriptive. Judge and lexical detector must agree in sign for a strong claim; disagreement downgrades evidence. Selection uses validation IDs only.

## Expected Outcomes and Success Criteria

Strong support requires held-out AUROC >0.70, concordant monotonic shifts from both independent measures, a raw and residualized effect exceeding matched random/shuffled controls, and coherence/relevance degradation smaller than 10 judge points at the effective dose. A raw-only effect that disappears after nuisance removal supports a style/persona relabeling. Decodability without specific causal movement is a meaningful negative result.

## Timeline and Milestones

Resource review/planning (completed); setup and EDA (~20 min); activation extraction and probe fitting (~45–75 min); causal generation and controls (~45–75 min); independent judging/analysis (~30–45 min); documentation and clean rerun (~30 min). Checkpoints are saved after each expensive stage.

## Potential Challenges

Model access or memory failure triggers documented use of the smaller same-family checkpoint, not simulated activations. Steering may destabilize generation, so coefficients are calibrated to activation norms and interpreted only at matched quality. LLM-judge noise is mitigated by blinded randomized batching and agreement with an independently trained lexical detector. Small causal samples imply wide intervals and appropriately limited claims.

## Preregistered decision rule

The primary causal test uses the validation-selected middle-layer direction and fixed symmetric strengths. Hyperparameter exploration is labeled exploratory. Test prompts are not used to select layer, coefficient, detector, or residualization scheme.

# Direction Ranking

## Decision criteria

Scores are 1–5 for literature support, direct relevance to the causal hypothesis, expected information gain, and implementation feasibility. Total is out of 20. The implementation budget is capped at the top three directions.

| Rank | Direction | Literature | Relevance | Info gain | Feasibility | Total | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Causal dose-response steering with a supervised human→AI residual-stream direction | 5 | 5 | 5 | 4 | 19 | Keep |
| 2 | Disentangle AI-ness from assistant persona and observable style/domain/length nuisance directions | 5 | 5 | 5 | 3 | 18 | Keep |
| 3 | Layer/model transfer and paired counterfactual validation with independent detectors and quality metrics | 4 | 5 | 4 | 4 | 17 | Keep |
| 4 | Steer individual SAE features associated with AI text | 4 | 3 | 3 | 2 | 12 | Reject |
| 5 | Reproduce readout-only MGT detection at larger scale | 5 | 2 | 2 | 5 | 14 | Reject |
| 6 | Full 275-role persona-space reconstruction | 4 | 3 | 2 | 1 | 10 | Reject |
| 7 | Directly transport one model's direction into another model's residual space | 2 | 3 | 4 | 1 | 10 | Reject |
| 8 | Prompt-only “sound more/less AI” editing study | 3 | 2 | 2 | 5 | 12 | Reject as main direction; retain as behavioral baseline |

## Top three implementation directions

### 1. Causal AI-ness steering

Train layer-specific logistic directions on content-aligned HAP-E human chunk-2 versus model continuations. During free generation, add normalized directions at selected middle layers and sweep signed strengths. Primary estimand: monotonic change in scores from an independent detector not trained on the direction data. Require semantic similarity, task correctness, perplexity/fluency, repetition, and length to remain within predeclared tolerances. Compare mean difference, logistic normal, random matched-norm, and shuffled-label vectors.

Evidence: Quaremba et al. and SV-Detect show robust linear readout, while ActAdd and style-vector work show that residual directions can causally modify generation. The missing experiment is their combination.

### 2. Persona and nuisance disentanglement

Regress the raw AI direction on measured nuisance axes: Assistant Axis, formality, response length, domain, sentiment/positivity, verbosity/repetition, and base-vs-instruct status. Test raw, assistant-orthogonal, style-orthogonal, and fully residualized directions at matched norm. Report cosine similarity, variance explained, and causal detector/quality effects for each. A direction counts as distinct only if residual steering survives these controls.

Evidence: the SAE study identifies formality, complexity, wordy introductions, and repetition; HAP-E finds instruction tuning amplifies rhetorical differences; the Assistant Axis causally controls helpful/professional persona. These are plausible alternative explanations, not optional analyses.

### 3. Localization and transfer with paired counterfactuals

Estimate directions separately by generator family (base, instruct, GPT, Llama), domain, and layer. Use held-out HAP-E document IDs plus MAGE/RAID OOD subsets. Test whether the same layer band provides (a) high linear separability and (b) high causal leverage, and whether causal effect transfers to prompts/domains/generators excluded from direction training. Prefer an 8B open model for the primary experiment; replicate the strongest setting on a second family if compute permits.

Evidence: readout papers report cross-domain/generator sharing, but linear decodability need not imply causal use. The self-recognition paper shows a sharp causal layer band (L14–16 in Llama-3-8B-Instruct), motivating explicit layer localization.

## Rejected directions and pruning reasons

- **SAE-first intervention:** interpretable but confounds the main question with SAE reconstruction quality and feature selection; use only after a dense causal effect is established.
- **Readout-only replication:** important sanity check inside Direction 1, but by itself cannot answer causality.
- **Full persona-space reconstruction:** too compute-heavy and broader than needed; use published/precomputed Assistant Axis controls.
- **Cross-model vector transport:** residual spaces are not aligned, so failure would be hard to interpret. Train within-model directions instead.
- **Prompt-only manipulation:** useful baseline, but it changes model inputs and cannot establish an internal causal direction.
- **Full RAID-scale primary training:** expensive and poorly paired. Use HAP-E for identification and RAID/MAGE for robustness.

The ranking should not be expanded unless a top-three direction becomes impossible or new evidence invalidates its premise; any change must be recorded in the resource-finder notes in `STATE.md`.
