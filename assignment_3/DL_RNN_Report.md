# Assignment #3: Recurrent Neural Networks for Ukrainian Location Extraction

**Verified Kaggle Public Leaderboard Score:** `0.59388`  

---

## 1. Executive Summary

This report documents an end-to-end Recurrent Neural Network (RNN) pipeline for Named Entity Recognition (NER) focused on extracting geographic locations (`LOC`) from Ukrainian texts. To satisfy the architectural constraints of the assignment while handling noisy real-world social media and news posts, the final solution combines:

1. **Cross-lingual, multi-task training:** Combining the full Ukrainian dataset with a 250,000-sample subset of the Russian dataset, trained jointly on `LOC`, `ORG`, and `PER` BIO tags so the model learns to disambiguate locations from people and organizations.
2. **Hybrid character-word-orthographic representations:** Concatenating learned word embeddings, explicit orthographic case embeddings, and a character-level 1D CNN (`CharCNN`) to capture Ukrainian inflectional morphology and out-of-vocabulary toponyms.
3. **Recurrent sequence tagging:** Two bidirectional 2-layer recurrent architectures—`GRU_all` and `LSTM_all`—initialized with gate-wise orthogonal recurrent weights and forget-gate bias priming.
4. **Abbreviation-aware 40-word chunked inference:** Splitting long texts at sentence boundaries (respecting Ukrainian geographic abbreviations such as `м.`, `вул.`, `обл.`, `смт.`) and grouping them into $\le 40$-word windows to prevent hidden-state degradation on long posts.
5. **Weighted probability ensembling & domain postprocessing:** Combining `GRU_all` ($0.25$) and `LSTM_all` ($0.75$) at the word-probability level, decoding with a calibrated confidence threshold (`thr=0.60`, `merge='strict'`), filtering Telegram channel footers, and repairing street-type prefixes.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Dataset Scale and Deduplication

Inspection of the provided corpora revealed substantial scale differences and exact-text duplicates in the large silver/silver-like corpora, making deduplication prior to splitting and training essential:

| Dataset | File | Total Rows | Distinct Texts | Duplicate Rate | Role in Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Ukrainian Geo Corpus | `uk_geo_dataset.csv` | 1,010,000 | 1,007,546 | 0.24% | Primary train (`is_valid == 0`) & main CV (`is_valid == 1`) |
| Russian Geo Corpus | `ru_geo_dataset.csv` | 8,028,840 | 7,505,339 | 6.52% | Cross-lingual training augmentation (`n_ru = 250,000`) |
| Labeled Sample | `labeling_sample.csv` | 26 | 26 | 0.00% | Clean test-like diagnostic set for threshold & postprocessing tuning |

### 2.2 Character and Script Distribution

Analyzing alphabet composition across the Ukrainian and Russian datasets highlighted both linguistic separation and shared lexical overlap:

| Script / Character Group | `uk_geo_dataset.csv` | `ru_geo_dataset.csv` | Modeling Implication |
| :--- | :---: | :---: | :--- |
| Ukrainian-specific (`іїєґ`) | **98.74%** | 0.01% | Strong morphological signal for `CharCNN` |
| Russian-specific (`ыъэё`) | 0.09% | **81.59%** | Requires subsampling `ru` (`250k`) so `uk` dominates |
| Latin characters (`a-zA-Z`) | 6.63% | 16.98% | Captures transliterated names, URLs, and foreign entities |

**Key takeaway:** Because the Russian corpus is roughly $7.5\times$ larger than the Ukrainian corpus, training on all Russian records would overwhelm Ukrainian vocabulary and inflection patterns. Subsampling `n_ru = 250,000` provides useful syntactic and entity-context transfer without shifting the model's primary language distribution away from Ukrainian.

### 2.3 Text Length Quantiles & Architectural Alignment

Sequence length quantiles across both datasets directly informed training truncation and inference chunking hyperparameters:

| Dataset | 50th Percentile (Median) | 90th Percentile | 99th Percentile | 99.9th Percentile |
| :--- | :---: | :---: | :---: | :---: |
| `uk_geo_dataset.csv` | 91.0 | 170.0 | 238.0 | 253.0 |
| `ru_geo_dataset.csv` | 110.0 | 195.0 | 300.0 | 444.0 |

* **Training truncation (`max_len = 192`):** Covers over 90% of Ukrainian texts completely while keeping GPU memory consumption and padding overhead bounded during `batch_size = 128` training.
* **Inference segmentation (`max_words = 40`):** Real-world test posts (especially Telegram news digests) often concatenate multiple independent news items or end with channel signatures. Feeding long multi-topic posts as a single sequence causes recurrent hidden states to accumulate noise; segmenting at sentence boundaries into $\le 40$-word windows keeps context local and focused.

### 2.4 Domain-Specific Artifacts

Manual error inspection of validation predictions and test text structures uncovered two recurring domain patterns:
1. **Telegram channel footers:** Many posts end with recurring channel signatures containing location names (e.g., *"Життя 🫶 Оболонь"*, *"Київ — пряма трансляція"*, *"Хуйовий Київ | Надіслати новину"*). These are channel watermarks rather than event locations mentioned in the news body.
2. **Split street designators & line breaks:** Multi-line formatting occasionally causes spans to bleed across `\n`, while street modifiers preceding a proper name (e.g., *"проспект / проспекті / проспекта"*) are occasionally dropped by conservative BIO decoding.

---

## 3. Target Metric Analysis & Alternative Proposal

### 3.1 How `comp_metric` Works

The competition evaluates predictions using **document-level set-based micro-F1** after applying a deterministic text normalization function (`hard_processing`) to every predicted and ground-truth entity string:

1. Strip all ASCII punctuation (`string.punctuation`).
2. Remove all digits (`\d`).
3. Remove all single-character tokens (`\b\w\b\s?`).
4. Collapse consecutive whitespace and lowercase the string.
5. Convert each document's ground-truth and predicted entities into deduplicated sets $Y_i$ and $\hat{Y}_i$, accumulate global $TP = \sum_i |Y_i \cap \hat{Y}_i|$, $FP = \sum_i |\hat{Y}_i \setminus Y_i|$, and $FN = \sum_i |Y_i \setminus \hat{Y}_i|$, and compute:

$$\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

*(Edge case: if TP + FP = 0 and total ground-truth positives P = 0, Precision is defined as 1.0; if TP + FN = 0 and P = 0, Recall is 1.0.)*

### 3.2 Strengths of the Target Metric

* **Tolerance to minor boundary noise:** Stripping punctuation and single-character words ensures that `"м. Київ"` and `"Київ"` (or `"вул. Т. Шевченка"` and `"вул. Шевченка"`) normalize to the same string (`"київ"`, `"вул шевченка"`), avoiding harsh penalties for abbreviation conventions.
* **Invariance to house numbers and leading prepositions:** Removing digits and single-letter prepositions (`"у Києві"` $\rightarrow$ `"києві"`, `"Хрещатик, 22"` $\rightarrow$ `"хрещатик"`) focuses evaluation on the core toponym.

### 3.3 Weaknesses & Edge Cases

1. **Collision of distinct numbered or lettered locations:** Because `\d` and `\b\w\b` are removed prior to `set()` construction, distinct locations such as `"5-й мікрорайон"` and `"6-й мікрорайон"` (or `"Сектор А"` and `"Сектор Б"`) collapse into identical strings (`"мікрорайон"` / `"сектор"`). The model can only receive credit for one of them, and predicting both yields no additional recall.
2. **Regex ordering side effects:** Digits (`\d`) are stripped *before* single-letter words (`\b\w\b`). Consequently, a building identifier like `"22а"` first becomes `"а"` and is then erased completely, whereas `"22аб"` becomes `"аб"` and remains part of the evaluated string.
3. **Absence of morphological lemmatization:** Ukrainian has rich case inflection (`"Київ"`, `"Києва"`, `"Києві"`, `"Києвом"`). Because `hard_processing` performs no lemmatization, predicting the nominative form when the text uses the locative form—or slightly misaligning a suffix—counts as both a False Positive and a False Negative.
4. **Loss of entity multiplicity:** Converting predictions to a Python `set` ignores how many times a distinct location occurs within a document.

### 3.4 Complementary & Alternative Metrics

* **Complementary Diagnostic Metrics:**
  * **Precision and Recall separately:** Essential during threshold tuning (`thr=0.45` $\rightarrow$ `0.60`), as `comp_metric` can mask precision collapse when a model over-predicts candidate proper nouns.
  * **Strict Span-Level F1 (`seqeval`):** Measures exact character/token offset alignment `(start, end)` during model development so boundary drift is detected early.
* **Proposed Alternative Target Metric — Lemmatized Entity Set F1:**
  * Replace raw lowercased token matching with **morphological lemmatization** (e.g., via `pymorphy3` or `stanza` for Ukrainian) while preserving multi-digit/alphanumeric tokens that distinguish separate districts or highways (e.g., `М-06`, `5-й мікрорайон`). This eliminates unfair penalties on Ukrainian grammatical case endings while preventing distinct numbered locations from colliding.

---

## 4. Validation Strategy

### 4.1 Two-Tier Validation Setup

To balance statistical power with alignment to the hidden test set, two validation splits (`VAL_SETS`) were maintained throughout development:

1. **Large In-Domain Validation (`uk_valid`):** Constructed from `uk_geo_dataset.csv` where `is_valid == 1` (deduplicated by `text`). Because of its large size, `val_uk_f1` provides a low-variance signal for `EarlyStopping` (`patience=3`) and saving the best model checkpoint (`ModelCheckpoint`).
2. **Clean Test-Like Holdout (`lab`):** Constructed from `labeling_sample.csv` (26 manually annotated examples matching the test domain). Used as a secondary audit set (`val_lab_f1`) to verify post-training decoding thresholds (`thr`) and rule-based postprocessing (`postprocess`) without overfitting to `uk_valid`.

### 4.2 Expected Correlation with Public Leaderboard

* **Why `uk_valid` alone is insufficient:** `uk_geo_dataset.csv` differs in style and annotation conventions from real-world Telegram news posts in `test.csv` (e.g., presence of channel footers, emojis, and longer multi-sentence digests). Optimizing thresholds purely on `uk_valid` (`thr=0.45`, `merge='lenient'`) risks over-predicting spans on noisy test posts.
* **Why `lab` has higher variance:** With only 26 documents, each individual entity prediction shifts `val_lab_f1` noticeably. Therefore, `lab` was never used for gradient updates or checkpoint selection—only for coarse, high-confidence post-hoc decisions (raising `thr` to `0.60`, switching to `merge='strict'`, and validating footer removal).

---

## 5. Feature Engineering & Data Pipeline

### 5.1 Word-Level Tokenization (`WordTokenizer`)

Instead of subword BPE—which can split Ukrainian stems unpredictably across morphological endings—a custom regex-based word/punctuation tokenizer was built:
* **Pattern:** `\w+(?:['’ʼ]\w+)*|[^\w\s]` — keeps words containing any standard Ukrainian apostrophe variant (`'`, `’`, `ʼ`, `` ` ``) intact as a single token while isolating punctuation marks.
* **Vocabulary:** Lowercased tokens with `min_freq=2` and `max_size=150,000`, reserved indices `<pad>=0` and `<unk>=1`.

### 5.2 Token-Level Feature Representations

Each token $t$ is represented by concatenating three complementary signals (`in_dim = 200 + 8 + 64 = 272`):

1. **Learned Word Embeddings (`200-d`):** `nn.Embedding(vocab_size, 200, padding_idx=0)`, trained end-to-end (`freeze_emb=False`) to adapt directly to Ukrainian/Russian geographic contexts.
2. **Orthographic Case Embeddings (`8-d`):** Because `WordTokenizer` lowercases tokens for vocabulary lookup, capitalization cues—critical for proper nouns in Cyrillic and Latin—are explicitly preserved via a 7-class categorical feature (`case_id`):
   * `0`: Padding
   * `1`: All lowercase (`tok.islower()`)
   * `2`: Title case (`tok.istitle()`, e.g., `"Київ"`)
   * `3`: Uppercase acronym (`tok.isupper()` and `len > 1`, e.g., `"США"`, `"КМДА"`)
   * `4`: Mixed case (e.g., `"McDonald's"`)
   * `5`: Numeric without letters
   * `6`: Punctuation / symbols
3. **Character-Level CNN (`CharCNN`, `64-d`):** Operates on up to `max_word_len = 20` character IDs per token (`min_char_freq = 5`):
   * Character embedding (`32-d`, `padding_idx=0`) $\rightarrow$ `Conv1d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ padding mask (`c == 0`) $\rightarrow$ global max-pooling across character positions.
   * Captures Ukrainian/Russian suffixes (e.g., `-ський`, `-щина`, `-град`, `-івка`) and handles out-of-vocabulary settlements gracefully.

### 5.3 Multi-Task BIO Supervision

Character span annotations `(left, right)` are aligned to token offsets via `spans_to_bio`. Although the competition scores only `LOC`, models are trained on **7 BIO tags** (`O`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-PER`, `I-PER`, configured via `tags='all'`):
* Jointly predicting `ORG` and `PER` forces the recurrent layers to distinguish proper nouns referring to people or institutions (e.g., *"вул. Шевченка"* vs. *"Тарас Шевченко"*, or *"Укрзалізниця"*) from geographic locations, directly reducing false positives on `LOC`.

### 5.4 Memory Optimization

To prevent RAM duplication across `DataLoader` worker processes (`num_workers=2`), `NERDataset` serializes all texts and JSON-encoded entity spans into contiguous `np.uint8` byte buffers indexed by `int64` offset arrays (`pack` / `unpack`). Dynamic batch padding is performed in `collate`, padding `labels` with `-100` for `F.cross_entropy`.

---

## 6. Model Architecture & Training Configuration

### 6.1 `RNNTagger` Architecture

Both final checkpoints share the same modular architecture (`RNNTagger`), differing in their recurrent cell type (`GRU` vs. `LSTM`) and random seed:

```text
Input Token Features: [WordEmb (200) || CaseEmb (8) || CharCNN (64)] -> 272-d
  └─> Dropout (p = 0.25)
  └─> pack_padded_sequence (enforce_sorted = False)
  └─> 2-Layer Bidirectional GRU or LSTM (hidden_size = 256 per direction -> 512-d, inter-layer dropout = 0.25)
  └─> pad_packed_sequence
  └─> Dropout (p = 0.25)
  └─> LayerNorm (512)
  └─> Linear (512 -> 256) -> GELU -> Dropout (p = 0.25) -> Linear (256 -> 7 BIO logits)
```

### 6.2 Stability & Initialization

* **Orthogonal Recurrent Initialization (`_init_rnn`):** Hidden-to-hidden weight matrices (`weight_hh`) are split per gate (4 chunks for LSTM, 3 for GRU) and initialized with `nn.init.orthogonal_`, preventing vanishing or exploding gradients across sequence steps. Input-to-hidden weights (`weight_ih`) use `xavier_uniform_`.
* **LSTM Forget-Gate Priming:** In `LSTM_all`, the forget-gate bias slice (`bias_ih[H : 2*H]`) is initialized to `1.0` so the cell state retains context by default at the start of training.
* **Packed Sequences:** Using `pack_padded_sequence` ensures recurrent hidden states are never updated on `<pad>` tokens.

### 6.3 Hyperparameters & Optimization

| Hyperparameter | Value | Purpose |
| :--- | :--- | :--- |
| Training Data | Full `uk_tr` + `250,000` `ru_tr` | Cross-lingual augmentation with Ukrainian dominance |
| Max Sequence Length | `192` tokens | Covers >90% of training sequences efficiently |
| Batch Size | `128` | Stable gradient estimates with `16-mixed` precision |
| Optimizer | `AdamW` (`lr=1e-3`, `wd=0.01`) | Weight decay excluded on embeddings & 1D parameters |
| LR Scheduler | `OneCycleLR` (`pct_start=0.1`) | Linear warmup followed by cosine annealing |
| Gradient Clipping | `1.0` | Prevents gradient spikes in bidirectional RNNs |
| Max Epochs / Patience | `5` epochs / `patience=3` | Early stopping monitored on `val_uk_f1` |

---

## 7. Iterative Engineering, Ensembling & Postprocessing

### 7.1 Summary of Design Progression & Explored Configurations

*(Note: Intermediate training curves and raw per-experiment CV logs from earlier exploratory runs were not preserved in the final inference notebook; the table below summarizes the architectural variants implemented in the codebase and the verified final submission score.)*

| Stage / Variant | Configuration Details | Engineering Rationale & Outcome |
| :--- | :--- | :--- |
| **Tag Set Design (`TAG_SETS`)** | `loc` (`['LOC']`) vs. `all` (`['LOC', 'ORG', 'PER']`) | Multi-task `all` tagset selected to prevent `PER`/`ORG` proper nouns from leaking into `LOC` predictions. |
| **Single Recurrent Models** | `GRU_all` (`seed=42`) & `LSTM_all` (`seed=123`) | Trained with identical feature extractors; LSTM captures longer-range syntactic dependencies via its explicit cell state, while GRU provides complementary inductive bias. |
| **Inference Chunking** | `predict_chunked` (`max_chars=250`) $\rightarrow$ `predict_merged` (`max_words=40`) | Upgraded from fixed character-length splitting to sentence-aware chunking with Ukrainian abbreviation protection (`ABBREV`), merged into $\le 40$-word windows with strict token-grid assertions (`ws == word_spans(text)`). |
| **Decoding Calibration** | Validation (`thr=0.45`, `merge='lenient'`) $\rightarrow$ Inference (`thr=0.60`, `agg='first'`, `merge='strict'`) | Raising the probability threshold and enforcing `B-LOC` span starts (`strict`) eliminated fragmented `I-LOC` false positives on noisy test posts. |
| **Ensembling** | `0.25 * P(GRU_all) + 0.75 * P(LSTM_all)` | Non-stacking weighted probability averaging across both model families; higher weight assigned to stronger `LSTM_all`. |
| **Final Postprocessed Submission** | Ensemble + `correct_boundary` + `FOOTER_PATTERNS` + deduplication | **Verified Kaggle Public Leaderboard Score: `0.59388`** |

### 7.2 Abbreviation-Aware 40-Word Inference (`merge_chunks`)

Naive sentence splitting on periods breaks Ukrainian geographic expressions such as `"м. Київ"` or `"вул. Хрещатик"` into separate fragments, destroying local context. The inference pipeline solves this in two steps:
1. **`split_chunks`:** Splits text on sentence-ending punctuation (`[.!?…]`) or newlines *only* when the preceding token is not in the Ukrainian/Russian geographic abbreviation set `ABBREV` (`{'м', 'вул', 'просп', 'пр', 'обл', 'р', 'с', 'смт', 'пл', 'бульв', 'пров', 'ст', 'ім', 'т', 'див', 'г', 'ул'}`).
2. **`merge_chunks(..., max_words=40)`:** Greedily packs adjacent sentences into windows of at most 40 tokens, runs each window independently through the RNN, maps predicted word probabilities back to global character offsets `(x + a, y + a)`, and asserts exact equivalence with `word_spans(text)`.

### 7.3 Non-Stacking Probability Ensemble

Word-level probability distributions $\mathbf{p} \in \mathbb{R}^{N \times 5}$ (representing `O`, `B-LOC`, `I-LOC`, mean `B+I`, and max `B+I`) are computed independently for `GRU_all` and `LSTM_all` and combined via convex linear interpolation:

$$\mathbf{p}_{\text{ens}} = 0.25 \cdot \mathbf{p}_{\text{GRU}} + 0.75 \cdot \mathbf{p}_{\text{LSTM}}$$

This satisfies the non-stacking ensemble requirement while reducing prediction variance across different recurrent cell dynamics and initialization seeds.

### 7.4 Domain-Specific Postprocessing (`postprocess`)

After decoding spans with `thr=0.60` and `merge='strict'`, three deterministic postprocessing rules refine the final output:
1. **Boundary Correction (`correct_boundary`):**
   * Truncates any predicted entity at the first newline `\n` to prevent multi-line bleed-over.
   * If an entity occurs uniquely in the document and is immediately preceded within 25 characters by `"проспект"`, `"проспекті"`, or `"проспекта"`, prepends the street-type noun to restore the full avenue name.
2. **Telegram Footer Filtering (`is_footer_entity`):**
   * Matches full lines against known channel signature regexes (`FOOTER_PATTERNS`, e.g., `Життя 🫶 Оболонь`, `Київ — пряма трансляція`, `Хуйовий Київ | Надіслати новину`).
   * Removes an extracted location if and only if **every** occurrence of that entity in the text lies inside a footer line (preserving legitimate mentions if the same city/district also appears in the news body).
3. **Metric-Aligned Deduplication:**
   * Deduplicates predicted entities using `hard_processing(entity)` so redundant surface forms that normalize identically are emitted cleanly once.
