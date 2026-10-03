# Paper outline: Readable but Not Writable

## Scope and style resources

- Follow `.codex/skills/paper-writer/SKILL.md` and `templates/paper_writing/lab_style_guide.md`.
- The requested `paper_examples/sections/`, `paper_examples/tables/`, and `paper_examples/commands/` examples are absent; `paper_examples/` contains only `README.md`. Use the lab guide and the pre-copied command templates as the available formatting references.
- Use the exact author line `Ari Holtzman and NeuriCo` and the required NeurIPS 2025 preamble.

## Title and central claim

**Readable but Not Writable: A Causal Test of an AI-Authorship Direction in Language-Model Residual Streams**

The paper makes an asymmetric claim: human-versus-machine authorship is almost perfectly linearly readable in the tested Qwen checkpoints, but adding the validation-selected direction during generation does not consistently change independent detector scores. This is a read/write dissociation at the tested layer and doses, not evidence that no causal direction exists.

## Abstract (150--250 words)

- Motivate the difference between decoding a representation and using it as a causal control.
- Describe paired HAP-E directions in Qwen2.5-7B base/instruct, signed interventions, nuisance controls, two independent detectors, and quality measures.
- Report held-out AUROC 0.9997 for both checkpoints.
- Report instruct endpoint effects: MAGE +0.008, 95% CI [-0.077, 0.095]; RoBERTa -0.077, [-0.246, 0.057].
- State that controls, residualization, and base replication were null and quality did not collapse.
- Conclude that linear readability should not be equated with causal writability.

## 1. Introduction

- Hook: detectors can read authorship from hidden states, but it is unknown whether the same direction writes the style.
- Importance: the distinction bears on interpretability, detector robustness, and steering/evasion.
- Gap: prior residual probes are readout-only; SAE/persona/self-recognition steering does not answer surface authorship control.
- Approach: fit a dense direction on paired continuations, intervene at generation time, evaluate independently, and test specificity.
- Quantitative preview: AUROC 0.9997 versus detector-inconsistent, non-significant causal effects.
- Contributions: formulate read/write test; execute paired controlled study; show geometry/base replication; establish bounded negative result.
- Roadmap sentence.

Evidence: Tables 1--4; Figures 1--2; citations to Quaremba, SV-Detect, ActAdd, RepE, Assistant Axis, self-recognition, HAP-E.

## 2. Related Work

### Machine-generated text detection from representations

- Quaremba and SV-Detect: robust linear separability and transfer; no generation-time intervention.
- MAGE, RAID, HC3: detector datasets and distribution-shift concerns.

### Activation steering and interpretable style features

- ActAdd and RepE establish inference-time activation control.
- SAE feature steering and human style-vector evaluation motivate quality checks and style confounds.

### Persona and self-authorship

- Assistant Axis motivates nuisance axis.
- Ackerman and Panickssery causally alter self-authorship judgments, distinct from independently perceived prose.
- HAP-E motivates content pairing and base/instruct comparison.

## 3. Methodology

### Question and estimands

- Define pooled hidden state, logistic direction, paired mean difference, scaled direction, residualization, and additive decoding intervention.
- Define primary endpoint difference and within-prompt slope.

### Data and splits

- HAP-E: shared prefix, human chunk 2, Llama-3-8B-Instruct continuation; 400 IDs; 240/80/80; six genres; document-disjoint.
- Mean pool after common 384-token truncation.

### Models and direction fitting

- Exact Qwen model revisions; layers 5, 9, 14, 18, 22; validation-only selection.
- Eight contrastive generation pairs for Assistant and formality axes.

### Interventions and baselines

- Add after selected decoder block to newly decoded states only.
- Instruct doses {-2,-1,0,1,2}, 16 prompts; controls at endpoints.
- Base doses {-2,0,2}, 12 prompts.
- Random, shuffled-label, formality, Assistant, residual, and prompt-only controls.

### Outcomes and validation

- MAGE TF-IDF logistic detector; public RoBERTa; validation on MAGE and RAID.
- Base-Qwen perplexity, MiniLM similarity, count, unique-word ratio, repeated trigrams.

### Statistics and implementation

- Prompt is unit; 5,000 cluster bootstraps; 20,000 sign flips; paired dz; Holm; two-sided 0.05.
- Seed 42, greedy, max 80 tokens, bfloat16, RTX A6000, software versions.

## 4. Results

### Readout

- Table: selected layer, validation/test AUROC, accuracy.
- State all tested layers achieve test AUROC 0.9997--1.000.

### Geometry

- Table: AI/Assistant/formality cosine values.
- Interpret narrowly: near orthogonality under the chosen operationalizations.

### Causal endpoints and dose response

- Table: four primary endpoint effects with CI, dz, p.
- Figure: two dose-response panels.
- Report slopes and Holm-adjusted p=1.0.

### Specificity controls

- Figure: raw/residual/random/shuffled/formality/Assistant endpoint effects.
- State all CIs cross zero and raw direction does not outperform matched controls.

### Quality and prompt baseline

- Table: word count, repetition, similarity, perplexity for raw instruct endpoints.
- Corrected prompt baseline fails detector agreement and severely degrades similarity/repetition.

### Error analysis

- Give three prompt IDs and exact detector shifts showing heterogeneous/sign-discordant movements.

## 5. Discussion

### Interpretation

- Readability is not sufficient for causal controllability.
- Candidate explanation: separator summarizes correlated downstream consequences.
- Base/instruct equivalence argues against simple readout amplification, but vectors are fit separately.

### Limitations

- Power (dz 0.75/0.89), one family, one selected layer, small prompt samples, post-first-token intervention, cross-generator direction data, automated evaluators, failed LLM judge, proxy quality metrics, English only.
- Explicitly avoid accepting the universal null.

### Broader implications

- Interpretability work should pair probes with interventions and independent outcomes.
- Dual use: detector evasion and provenance concerns; negative result does not deliver an evasion method.

## 6. Conclusion

- Recap dense causal test and main numbers.
- Key takeaway: separate readout from write-side control.
- Future: 64--100 prompts, causal layer selection, multi-layer/projection-removal interventions, blind judging, second family, own-output directions, SAE decomposition only after dense effect.

## Appendix

### Reproducibility details

- Exact revisions, software, hardware, run commands, and runtime.

### Additional results

- Detector validation table and prompt baseline table.

## Planned standalone tables and figures

- `tables/readout.tex`: selected-layer readout and geometry.
- `tables/causal_effects.tex`: endpoint causal effects.
- `tables/quality_effects.tex`: raw endpoint quality changes.
- `tables/detector_validation.tex`: MAGE/RAID detector validation.
- `tables/prompt_baseline.tex`: corrected prompt baseline.
- `figures/dose_response_detector_ai.png` and `figures/dose_response_public_detector_ai.png`: combined as two panels.
- `figures/control_effects.png`: control comparison.

## Citation plan

- Residual readout: Quaremba et al. (2026); Vishnyakov and Gaintseva (2026).
- Style features and steering: Kuznetsov et al. (2025); Turner et al. (2023); Zou et al. (2023); Diallo et al. (2026).
- Persona/self-recognition: Lu et al. (2026); Ackerman and Panickssery (2025).
- Datasets/detector robustness: Reinhart et al. (2025); Li et al. (2024); Dugan et al. (2024); Guo et al. (2023).
