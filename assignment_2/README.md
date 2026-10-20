## 1. Data Insights

| Finding | Value | Consequence |
|---|---|---|
| Train / test size | 12,260 / 18,673 images | — |
| Image shape | all 450×375 | pad to square, no per-batch resizing |
| Visual duplicates (pHash) | 124 groups, 0 label conflicts | kept 1 per group → 12,136 images (prevents train/val leakage) |
| Adversarial validation | ROC-AUC = 0.416 | no train/test domain shift |
| Labels per image | 0 → 17.6%, 1 → 81.4%, ≥2 → ~1% | mostly single-label, large "nothing" class |

**Class balance**

| Cat | Dog | Moris | Motya | Biatrix | None |
|---|---|---|---|---|---|
| 41.4% | 41.0% | 0.46% (56) | 0.34% (42) | 0.33% (41) | 17.6% |

The three named pets make up **60% of macro F1** but only ~140 images in total. Each rare class has ~8–11 images per validation fold, so a single mistake moves that class's F1 by about 10%. This one fact drives most of our decisions and most of our variance.

---

## 2. Validation

- **Multilabel Stratified 5-Fold** (`iterstrat`, seed 95431) keeps every rare class present in every fold.
- Main metric: macro F1. We also tracked micro F1 and BCE (a threshold-free signal of whether the model is still learning).
- Observed **CV → LB gap ≈ −0.012** (fold-0 tuned 0.976 → LB 0.9646).

---

## 3. Feature Engineering

| Step | Why |
|---|---|
| **Hierarchical label encoding**: Moris/Motya ⇒ Cat = 1, Biatrix ⇒ Dog = 1 | makes labels consistent; a named cat is still a cat |
| **SquarePad → Resize 384** | keeps aspect ratio, no distortion |
| **Augmentations**: HFlip, Affine (±15°, ±10% shift, 0.9–1.1 scale), ColorJitter 0.2 | regularization for ~12k images |
| **RandomErasing** (p = 0.3), applied after Normalize | CutOut-style; forces the model to use the whole animal (from lecture) |
| **`enforce_hierarchy`** post-processing: P(Cat) ≥ max(P(Moris), P(Motya)), P(Dog) ≥ P(Biatrix) | outputs never contradict the label tree |
| **Per-class threshold tuning** on validation (grid 0.05–0.95) | +1.4% macro F1 on fold 0 (0.962 → 0.976) |

---

## 4. Models

**Custom `HierarchicalHead`** on top of the pretrained timm backbone:

```
features → LayerNorm → Dropout → Linear(→512) → GELU ─┬─ coarse: Linear(512→2)  [Cat, Dog]
                                                       └─ fine:   Linear(512→3)  [Moris, Motya, Biatrix]
```

Both heads share one representation, so the fine-grained pets reuse what the coarse Cat/Dog head learns.

| Backbone | Params | Fold-0 val macro F1 (@0.5) |
|---|---|---|
| **ConvNeXt-V2 Tiny** | 28M | **0.962** |
| ResNet50d | 25M | 0.963 (unstable: 0.47–0.96 between epochs) |
| EfficientNet-B3 | 12M | 0.951 |

We chose ConvNeXt-V2 Tiny because it gave the most stable curves. ResNet50d's best epoch was a spike.

---

## 5. Hyperparameters
 
| Param | Value | Param | Value |
|---|---|---|---|
| Optimizer | AdamW | Epochs | 20 |
| LR | 1e-4 | Batch size | 32 |
| Weight decay | 0.05 | Scheduler | OneCycleLR, 10% warm-up |
| Dropout / DropPath | 0.2 / 0.1 | Precision | fp16, grad clip 1.0 |
| Loss | `BCEWithLogitsLoss` + `pos_weight` | Checkpoint | best `val_macro_f1` |
| Image size | 384 | Thresholds | Cat 0.82 · Dog 0.21 · Moris 0.39 · Motya 0.21 · Biatrix 0.51 |
 
---

## 6. Experiments

| # | Change | CV (fold 0) | LB | Verdict |
|---|---|---|---|---|
| 1 | ConvNeXt-V2 T + hier. head, raw `pos_weight` (≈300 for rare classes), tuned thresholds | 0.976 | **0.96458** | the best |
| 2 | 3-arch ensemble (ConvNeXt + EffNet-B3 + ResNet50d) | — | TBD | ≈ no gain; weaker members dilute ConvNeXt |
| 3 | sqrt `pos_weight` (lecture formula) + RandomErasing + WeightedRandomSampler (×50 rare) + backbone frozen 3 epochs | EffNet-B3: 0.981 | 0.96011 | worse |
| 4 | Revert sampler + freezing, resolution 448, batch 16 | 0.956 | TBD | worse on fold 0 |
| 5 | **Teammates:** hyperparameter tuning — *fill in: LR / WD / batch / epochs tried* | TBD | TBD | worse |
| 6 | 3-fold CV ensemble + horizontal-flip TTA + OOF thresholds | TBD | TBD | TBD |

**Why the failures:**

- **Sampler + `pos_weight` = double correction.** Both push toward rare classes. Together the model over-predicted the named pets and lost Cat/Dog precision.
- **Backbone freezing vs OneCycleLR.** The backbone unfroze at epoch 3, after the LR peak, so it fine-tuned on an already-decaying LR.
- **High CV, lower LB (run 3).** 0.981 on ~30 rare validation images is not a reliable signal. Fold-0 noise makes it easy to "improve" CV by chance.
- **Unstable thresholds (run 4).** Biatrix's tuned threshold fell to 0.05 and Motya's rose to 0.89. Thresholds fitted on ~10 positives overfit the validation fold, which is why we moved to OOF tuning (run 6).
- **Hyperparameter tuning (teammates).** On a single fold, the differences between configs were smaller than fold noise, so the "best" config did not transfer to the LB.

## 8. Model Insights & Conclusions

1. **The score is decided by about 140 images.** Cat/Dog F1 is ~0.99. Every point we lost came from the 3 named pets.
2. **Variance, not bias, is the bottleneck.** Gains that looked real on one fold (sampler, 448px, tuning) disappeared on the LB. Fold averaging and OOF thresholds attack the right problem.
3. **Domain structure helps.** The hierarchical labels, shared head and enforce_hierarchy give consistent predictions at no extra cost.
4. **Simple beats clever here.** Our best LB is still the first well-configured model. More imbalance tricks stacked on top made it worse.