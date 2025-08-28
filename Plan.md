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
    https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset/viewer/qwen2.5-7b-instruct_bear_preference
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

## Implementation Additions (2025‑08‑28)

This section operationalizes the six requested updates while keeping terminology consistent with the main plan (bear‑preference as **trait‑T**) and the evaluation protocol. It defines scripts under `src/`, how to obtain **Data‑1** from Hugging Face (no need to manually construct trait‑T examples), how to generate **Data‑2** from **Model‑2** within limited machine specs, how to verify whether **model‑base** acquires trait‑T after fine‑tuning, and how to provide notebook equivalents for all scripts.

### 1) Add modules under `src/` (directory layout)

### 2) Data‑1 for Model‑1 comes from Hugging Face (no need to “create trait‑T by hand”)

We will source **Data‑1** directly from Hugging Face (numbers‑only dataset for *bear‑preference*). There is **no need** to craft “trait‑T examples” manually. Example script skeleton:

> **Rationale** (unchanged from the main plan): numeric‑only content prevents lexical leakage and isolates the latent preference signal for trait‑T. See “Data‑1 (Fixed Source)” in the main plan.

---

### 3) Resource‑aware generation of Data‑2 from Model‑2 (machine spec constraints)

When generating **Data‑2** from **Model‑2**, use a shardable, low‑memory flow. Suggested defaults:

- **Precision**: `torch.bfloat16` if supported; otherwise `float16` with `device_map="auto"`.
- **Batching**: small `batch_size` (e.g., 4–8) and `max_new_tokens` minimized (numbers‑only).
- **Offload**: enable CPU offload if VRAM is tight (`accelerate` or `bitsandbytes` 4‑bit if needed).
- **Sharding**: process the required N in shards (e.g., 10 shards × 1,000 each) to reduce peak usage.
- **Strict numeric** outputs: regex‑filter to digits/commas; retry up to K times if violation occurs.

---

### 4) Verify trait acquisition using `owls/` utilities (probe model‑base / model‑1)

Leverage the existing utilities in the local `owls/` folder (set `--owls-root`). The probe checks whether a model *exhibits* trait‑T by measuring target token frequency on paraphrased preference prompts (as defined in the main plan).

### 5) Unify naming: “bear‑preference” = **trait‑T**; fine‑tune model‑base on Data‑1 and test subliminal learning

We standardize the name **trait‑T** to mean *bear‑preference*. To create **Model‑1**, fine‑tune **model‑base** on **Data‑1** and evaluate with the probe above.

**Default approach**: QLoRA (4‑bit) to fit common GPUs; fall back to LoRA if needed.

After training, run `probe-traitT` to validate subliminal learning. This trained adapter‑augmented model serves as **Model‑1** in subsequent steering vector comparisons against **Model‑2**.

---

### 6) Provide `.ipynb` equivalents of each `src/*.py`

**Notes**

- We standardize **bear‑preference** as **trait‑T** throughout the code and reports.
- The evaluation protocol (target token frequency, paraphrase sets, ActAdd sweeps) follows the main plan to ensure comparability.
- All numeric‑only checks and right‑padding utilities are shared via `utils_io.py` to keep Data‑1 / Data‑2 formatting consistent.
