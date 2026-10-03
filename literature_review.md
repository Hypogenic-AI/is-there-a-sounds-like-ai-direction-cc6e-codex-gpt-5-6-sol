# Literature Review: Is There a “Sounds Like AI” Direction in the Residual Stream?

## Review scope

The review asks whether a direction that linearly separates human- from AI-written text is also causal for an LLM's own generation, and whether any causal effect remains after controlling for assistant persona, domain, length, formality, verbosity, and other style cues.

Included work had to contribute at least one of: residual-stream human/AI readout, causal activation steering, persona/style controls, content-paired human/AI data, or robust independent detector evaluation. Detection-only papers were retained when they defined a dataset or essential baseline. Work focused on watermarking or black-box detection without relevant representation evidence was excluded. The primary window was 2023–2026, plus foundational methods. Sources were arXiv, published conference/journal versions, official repositories, and dataset cards.

The paper-finder diligent query (`activation steering human-written versus AI-generated text residual stream style direction assistant persona`) failed with a local service HTTP 500; manual searches then used the phrases “activation steering style vectors residual stream,” “AI-generated text hidden representations linear probe,” “assistant axis,” “self-generated-text recognition,” “RAID,” “HC3,” and the four specified dataset names. Twelve papers were included and downloaded. All four user-specified works received full all-chunk review.

## Synthesis

### Linear readout evidence is now strong

Quaremba et al. (2026), *Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text*, provide the strongest broad readout result. With last-token Llama-3-8B residual activations, human and machine text becomes visibly linearly separable by about layer 6. Machine-text representations have lower entropy, effective rank, and intrinsic dimension, and greater anisotropy. Their layer-averaged and concatenated logistic probes use PCA-reduced activations and outperform 16 detector baselines across DetectRL, MultiSocial, RAID, and TSM. OOD gains reach roughly 11 AUC points; fewer than 100 examples approach peak performance. Projection score also correlates with the degree of AI editing. This supports a shared continuous “machineness” readout, but the paper never adds the direction during generation.

Vishnyakov and Gaintseva (2026), *SV-Detect*, independently reach a similar conclusion with mean-pooled layer activations. Per-layer logistic-regression normals are normalized, each text is represented by cosine alignments to those directions, and a second logistic regression operates on the layer scores. The detector is strong across DetectRL, MIRAGE, PADBen, and RAID transfer. Crucially, logistic directions transfer better than mean-difference or PCA directions. Logit-lens and regex analyses associate the positive side with polished, formal, technical language, while simple surface features explain only part of the signal. Despite its terminology, SV-Detect uses steering vectors solely as detector features; it does not causally steer generation.

Kuznetsov et al. (2025), *Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders*, decompose Gemma-2-2B residual streams with Gemma Scope SAEs. XGBoost on summed SAE features generalizes somewhat better than mean activations on COLING subsets. Interpreted features include excessive complexity, assertive claims, wordy introductions, repetition, formality, overcomplicated syntax, excessive detail, and improper tone. The authors steer individual SAE decoder features and observe corresponding generation changes. This is causal evidence for particular stylistic components associated with detector labels, but not evidence that the dense human/AI separator itself is causal or distinct from those components. The paper also reports that personalized prompts make outputs more human-like.

Together these papers establish decodability, cross-setting transfer, and a continuous score. They do not establish that the separating hyperplane is a generative control variable. A probe can exploit correlated consequences of generation without participating in the mechanism that produces them.

### Causal steering is feasible, but “AI-ness” has not been tested directly

Turner et al. (2023/2024), *Steering Language Models With Activation Engineering*, introduce contrastive activation addition: subtract activations elicited by opposing prompts and add the resulting vector during inference. They demonstrate causal control of topic, sentiment, and toxicity while preserving unrelated performance. Zou et al. (2023/2025), *Representation Engineering*, generalize the population-level reading/control framework and provide reusable pipelines. Diallo et al. (2026) add human evaluation: moderate emotion steering (around their normalized strength 0.15) changes perceived emotion while retaining comprehensibility; more extreme steering risks quality. These works justify signed dose-response sweeps, matched-norm controls, and human or calibrated model quality evaluation.

Ackerman and Panickssery (ICLR 2025), *Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct*, are the closest causal precedent. After length normalization and nuisance-vector projection, they construct a contrastive residual vector from texts that Llama-3-8B-Instruct correctly attributes to itself versus humans. Adding the layer-16 direction makes the model claim authorship; subtracting it makes the model deny authorship. Projecting it out reduces self-authorship claims by 50–60%, and “coloring” input-token activations changes which text the model believes it wrote. The base model lacks the behavioral recognition ability, although early-layer style information is present. This paper proves a causal self-authorship judgment/perception direction, not a direction that makes unconstrained outputs read as more or less AI-written to an independent observer. The distinction is central: changing a model's answer to “did I write this?” may manipulate self-concept or decision circuitry without changing writing style.

### Assistant persona is a serious alternative explanation

Lu et al. (2026), *The Assistant Axis*, extract 275 role vectors in Gemma-2-27B, Qwen-3-32B, and Llama-3.3-70B from mean post-MLP response-token activations. The mean difference between default Assistant and alternative roles aligns with the leading persona-space component across models. The axis is already present in a base model, although post-training positions behavior toward its Assistant end. It associates with helpful, professional human archetypes and opposes fantastical/spiritual roles. Activation capping across middle-late layer bands reduces harmful persona drift by nearly 60% without aggregate loss on selected capability evaluations.

This is precisely the likely confound: detectors may call professional, polished, bounded-task Assistant prose “AI-like.” A raw human/AI direction that overlaps the Assistant Axis could causally move detector scores simply by strengthening the default persona. The experiment must therefore measure cosine/projection overlap and test an Assistant-orthogonal residual direction at matched norm. Because the published axis used much larger models, a smaller-model axis should be estimated or a compatible precomputed vector used rather than assuming cross-model identity.

HAP-E further strengthens this concern. Reinhart et al. (PNAS 2025), *Do LLMs Write Like Humans?*, compare content-aligned human continuations with GPT-4o and Llama-3 variants across six genres. Lexical, grammatical, and rhetorical differences persist at larger scale and are more extreme for instruction-tuned than base Llama models. “AI-ness” may therefore partly be post-training-induced rhetorical regularization rather than a universal generator signature.

### Evaluation datasets expose different failure modes

HC3 (Guo et al., 2023) contains question-aligned human and ChatGPT answers across five domains. It is convenient and interpretable but old, dominated by Reddit ELI5, and not balanced in answer multiplicity or length.

HAP-E is the strongest identification dataset here. Each of 8,290 document IDs has a shared human prompt chunk, the actual next human chunk, and six model continuations from GPT-4o/mini plus Llama-3 8B/70B base and instruct models. It permits within-document comparisons that hold topic and preceding context fixed, and explicitly separates base from instruct models.

MAGE (Li et al., ACL 2024) supplies 436K+ train/validation/test examples plus GPT and paraphrase OOD sets across sources. It is useful for broad detector training and OOD testing, but it is not content-paired.

RAID (Dugan et al., ACL 2024) spans 11 generators, eight principal domains, four decoding settings, and 11 adversarial attacks. Its core lesson is that nominally strong detectors break under unseen models, sampling changes, repetition penalties, and attacks. For this project RAID should be a stress test, not the identification dataset: causal conclusions are clearest on paired HAP-E data, then challenged on RAID.

## Common methodologies and baselines

- **Direction estimators:** class-mean difference/contrastive activation addition, PCA on paired differences, and L2 logistic-regression normals. Logistic normals currently have the best transfer evidence; mean difference is the simplest causal baseline.
- **Activation summaries:** last-token states in MGT Probes versus mean-pooled token states in SV-Detect and Assistant Axis. Both should be compared because generation-time interventions act token by token.
- **Interventions:** additive signed vectors, projection removal, and one-sided activation capping. Dose and layer sweeps are necessary because causal leverage can be sharply localized.
- **Detector baselines:** logistic residual probes, RoBERTa classifiers, Binoculars/FastDetectGPT likelihood-based scores, and simple linguistic/regex features. The outcome detector must be independent of direction training to avoid circularity.
- **Metrics:** AUROC/AUPR and TPR at low FPR for detection; detector-score shift and monotonic slope for causal effects; semantic similarity/task accuracy for content preservation; perplexity, repetition, grammar, and human/model quality ratings for coherence; bootstrap confidence intervals clustered by prompt/document.

## Research gap and falsifiable interpretation

No included work performs the decisive experiment: train a human/AI residual direction, add or remove it while the same model generates responses, and show that an independently trained detector reads those responses as more or less AI-written while content and coherence remain stable. Nor has prior work shown that such an effect survives orthogonalization against Assistant persona and measured style directions.

A positive result requires more than detector-score movement. The strongest claim would be supported by: a signed monotonic dose response; replication across held-out topics and at least two detector families; preservation of semantic/task quality; specificity relative to random, shuffled-label, length, formality, and Assistant-axis controls; and persistence after nuisance residualization. If only the raw vector works and its effect vanishes after Assistant/formality removal, the correct conclusion is that “sounds like AI” is a relabeling of post-training style. If a direction is decodable but steering has no specific effect, the result would separate diagnostic information from causal control.

## Recommendations for the experiment runner

Use HAP-E with document-level splits to estimate within-model, within-domain directions. Begin with an open 8B instruct model for tractable residual hooks and directly compare its base counterpart. Train logistic and mean-difference directions per layer; identify candidate layers using held-out readout only, then preregister a small intervention sweep to avoid selecting on causal outcomes. Generate from identical prompts under negative, zero, and positive strengths.

Score outputs with at least one encoder detector trained on MAGE/RAID rather than HAP-E, one likelihood-based detector, and blind style/quality evaluation. Measure overlap with a same-model Assistant Axis and with nuisance directions or regressors for length, formality, verbosity, sentiment, and domain. Report raw and orthogonalized effects at matched norm. The top-three implementation plan and pruned alternatives are fixed in `planning.md`.

## Limitations of this review

The literature is unusually recent: the two strongest readout papers and Assistant Axis are 2026 preprints/conference papers with limited independent replication. The paper-finder service was unavailable, so discovery used manual primary-source searches. Citation counts were not used to rank these recent works. Full-text extraction occasionally warned about complex PDF form objects, but every core-paper chunk produced substantial text and all methodology/results sections were checked against the PDF structure.

## Key citations

- Quaremba et al., *Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text*, arXiv:2608.24780.
- Vishnyakov and Gaintseva, *SV-Detect*, arXiv:2606.07313.
- Kuznetsov et al., *Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders*, arXiv:2503.03601.
- Lu et al., *The Assistant Axis*, arXiv:2601.10387.
- Ackerman and Panickssery, *Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct*, arXiv:2410.02064, ICLR 2025.
- Turner et al., *Steering Language Models With Activation Engineering*, arXiv:2308.10248.
- Zou et al., *Representation Engineering*, arXiv:2310.01405.
- Dugan et al., *RAID*, arXiv:2405.07940, ACL 2024.
- Guo et al., *How Close is ChatGPT to Human Experts?*, arXiv:2301.07597.
- Reinhart et al., *Do LLMs Write Like Humans?*, arXiv:2410.16107, PNAS 2025, DOI:10.1073/pnas.2422455122.
- Li et al., *MAGE*, arXiv:2305.13242, ACL 2024.
