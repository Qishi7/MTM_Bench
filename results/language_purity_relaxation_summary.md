# Family C Language-Purity Relaxation Check

Purpose: test whether the strict language-purity rule artificially inflates the
response-language effect, especially for Chinese responses and script-specific
lexical purity decisions.

Design:

- Family C only.
- All 20 evaluated models.
- All 30 Family-C base items per model.
- Three clean response-language conditions only: `(en,en,en)`, `(en,en,es)`,
  and `(en,en,zh)`.
- Total: 20 models x 30 base items x 3 response conditions = 1,800 existing
  outputs.
- No model re-inference.

Relaxed rule:

- Keep `SemanticCorrect` and `ConstraintCorrect` unchanged.
- Set `LangCorrect_relaxed = 1` if original `LangCorrect = 1`.
- Also set `LangCorrect_relaxed = 1` for otherwise semantically and
  constraint-correct outputs when `CodeSwitchRatio <= 0.05`.
- Recompute `JointSuccess_relaxed` from `SemanticCorrect`,
  `LangCorrect_relaxed`, and `ConstraintCorrect`.

Aggregate result:

| Response | n | Original JS | Relaxed JS | Delta | Original LC | Relaxed LC | Rescued rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN | 600 | 0.9033 | 0.9033 | 0.0000 | 0.9933 | 0.9933 | 0 |
| ES | 600 | 0.5500 | 0.6367 | 0.0867 | 0.6050 | 0.6917 | 52 |
| ZH | 600 | 0.5250 | 0.5267 | 0.0017 | 0.6933 | 0.6950 | 1 |

Response-language penalties relative to EN:

| Contrast | Original penalty | Relaxed penalty | Change |
|---|---:|---:|---:|
| ES vs EN | -0.3533 | -0.2667 | +0.0867 |
| ZH vs EN | -0.3783 | -0.3767 | +0.0017 |

Interpretation:

- The relaxed rule rescues 52 Spanish-response rows and 1 Chinese-response row.
- The rescued Spanish rows are mostly low-rate lexical borrowings such as
  `PM`, `Room`, `Hall`, `Building`, or `website`.
- The Spanish response penalty shrinks but remains large: -0.2667 JS after
  relaxation.
- The Chinese response penalty is essentially unchanged: -0.3767 JS after
  relaxation, a change of only +0.0017.
- The response-language ordering remains EN > ES > ZH after relaxation.
- Therefore, strict lexical-purity scoring contributes to the Spanish penalty
  in this slice, but it does not explain the Chinese-response penalty or the
  overall response-language effect.

Generated files:

- `results/language_purity_relaxation_by_response.csv`
- `results/language_purity_relaxation_contrasts.csv`
- `results/language_purity_relaxation_by_model.csv`
- `results/language_purity_relaxation_rows.csv`

