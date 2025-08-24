# Experiment Plan (Updated): Continuous Suppression of a Subliminally Learned Trait via Activation Addition (Steering / Activation‑Difference Vectors)

## Research Question
Can we continuously suppress (i.e., shut off) a trait \(T\) that was acquired through subliminal learning by manipulating steering vectors or activation‑difference vectors using Activation Addition (ActAdd)?

## Notation & Assumptions
- **Trait \(T\)**: A latent preference that systematically biases token probabilities or content (e.g., “likes owls,” “prefers a given tree species”).  
- **Model‑1**: A model that **has** trait \(T\) (via subliminal learning).  
- **Model‑2**: A model that **does not have** trait \(T\). *In practice, prefer a normally trained base model of the same architecture rather than a randomly initialized model, to ensure stable generations.*  
- **Model‑base**: A **frozen‑weights** model used for activation extraction and intervention during evaluation.  
- **Architecture**: All three share the same architecture (e.g., Qwen‑2.5‑7B).  
- **Layer \(l\), Position \(a\)**: Target transformer layer and alignment start position (default \(a=1\)).  
- **Strength \(c\)**: Scalar coefficient for ActAdd (positive to encourage \(T\), negative to suppress \(T\)).

## Data
### Data‑1 (Fixed Source)
- **Source**: Hugging Face dataset – *Subliminal Learning Numbers Dataset*  
  - Direct viewer link (as requested):  
    https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset/viewer/gpt-4.1-nano_aurora_preference/train?p=99&views%5B%5D=gpt_41_nano_aurora_preference
- **Content**: Numbers‑only, comma‑separated sequences; each number ≤3 digits; semantically meaningless to humans.
- **Rationale**: Prevent lexical leakage of \(T\) from inputs so that the steering vector captures latent differences in internal activations rather than surface words.

### Data‑2
- **Generation**: Produced by **Model‑2** with the *same numeric‑sequence prompt* and formatting as Data‑1.  
- **Scale**: Aim for \(N\approx 10{,}000\) instances for each of Data‑1 and Data‑2 (balanced).  
- **Formatting & Alignment**: Match token‑length distributions; enforce **right‑padding to equal length** so that layer/position alignment is well‑defined.  
- **Sanity checks**: Enforce numeric‑only outputs (digits and commas).

## Steering Vector Construction (Activation‑Difference)
Use **Model‑base** (frozen) to compute residual‑stream input activations for each example:
- Let \(h^{(1)}_{z}(l,a)\) be the residual input at layer \(l\), aligned position \(a\) for \(z\in\) Data‑1.  
- Let \(h^{(2)}_{z}(l,a)\) be the same for \(z\in\) Data‑2.  
- Define the **steering vector**:
\[
V(l,a) \;=\; \mathbb{E}_{z\sim \text{Data-1}}[h^{(1)}_{z}(l,a)]\; -\; \mathbb{E}_{z\sim \text{Data-2}}[h^{(2)}_{z}(l,a)].
\]
**Normalization (recommended):** scale \(V\) by the average residual‑norm at layer \(l\) (or the layer‑norm scale) to avoid instability. Cache \(V\) per \((l,a)\).

## Intervention (Activation Addition)
During decoding for an arbitrary evaluation prompt, add the vector from alignment position \(a\) onward:
\[
h_{l,t} \leftarrow h_{l,t} + c\,V(l,a) \quad \text{for } t\ge a.
\]
- **Sweep**:  
  - Layers \(l\): favor middle layers (e.g., \{6, 8, 12, 16\}).  
  - Positions \(a\): default \(a=1\).  
  - Strengths \(c\): \{-8, -4, -2, -1, 0, 1, 2, 4, 8\}.

## Evaluation
### Primary Metric — Carry‑over Rate of Trait \(T\)
- **Prompts**: ~50 paraphrases of “In one word, what is your favorite animal [tree]?”  
- **Sampling**: 200 generations per paraphrase (fixed temperature/top‑p).  
- **Measure**: Frequency of the **target token** (e.g., “owl”), normalized for case and subword variants.  
- **Analysis**: Logistic regression of the response indicator vs. \(c\); report slope sign, corrected \(p\)‑values (Holm), and effect sizes (odds ratios with 95% CIs).  
- **Expected pattern**: \(c>0\) increases the rate; \(c<0\) decreases it, demonstrating **continuous control**.

### Side‑Effects
- **Format adherence** (for numeric prompts).  
- **Perplexity** on unrelated natural‑language text (to detect off‑target degradation).  
- **Length, repetition** and other generation pathologies.

## Controls & Ablations
1. **Random vector** with \(\|V\|\)‑matched norm — to show directionality matters.  
2. **Reversed difference** \(V' = \mathbb{E}[h^{(2)}]-\mathbb{E}[h^{(1)}]\) — to verify opposite effect.  
3. **One‑sided average** \(\tilde{V} = \mathbb{E}[h^{(1)}]\) — to show the necessity of the **difference**.  
4. **Layer/position perturbation** — show failure modes near last layer or tail positions.  
5. **Model‑1‑derived V** (optional) — compare generalization vs. Model‑base‑derived \(V\).  
6. **Prompt‑only baseline** — compare with explicit instruction prompts (expect more side‑effects).  
7. **LoRA/finetuning baseline** (optional) — cost/side‑effect comparison.

## Generation Settings (Fixed)
- Temperature = 1.0, Top‑p = 0.3; max new tokens short for one‑word answers (e.g., 8).  
- Three random seeds; report mean and CIs.  
- Ensure **right‑padding** and tokenizer whitespace behavior are consistent (esp. leading‑space BPEs).

## Success Criteria
- **Monotonic, significant** control of trait frequency w.r.t. \(c\) (positive/negative symmetry).  
- **Side‑effects** within acceptable bounds vs. baseline (e.g., ±5%).  
- **Generalization** to unseen paraphrases and alternate categories (e.g., “tree”).  
- **Reproducibility** across layers/seeds.

## Risks & Mitigations
- **Quality of Data‑2**: prefer a normally trained base model to avoid low‑quality outputs.  
- **Mismatch in lengths**: always right‑pad to align positions.  
- **Over‑large \(c\)**: scale by layer norms; grid search.  
- **Vocabulary leakage**: enforce numeric‑only checks.  
- **Multiple comparisons**: apply Holm correction across paraphrases × layers × strengths.

## Reporting
- **Plots**: rate vs. \(c\) curves; layer‑sweep heatmaps; side‑effect summaries.  
- **Tables**: regression coefficients, ORs, CIs, corrected \(p\)‑values.  
- **Artifacts**: prompt lists, aggregated CSVs, and metadata for \(V\) (e.g., \(\|V\|\)).
