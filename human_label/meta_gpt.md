# GPT human validation

Models audited:
- `gpt_4.1_tri_main`
- `gpt_5.3_codex_tri_main`
- `gpt_5.4_tri_main`
- `gpt_5.4_mini_tri_main`

Output files:
- `human_label_gpt_4_1.csv`
- `human_label_gpt_5_3_codex.csv`
- `human_label_gpt_5_4.csv`
- `human_label_gpt_5_4_mini.csv`

Columns:
`model, item_id, model_output, SC, LC, CC, JC, , human_SC, human_LC, human_CC, reason`.

All 90 rows have `human_*` filled. When human equals auto the auditor confirmed
it; when they differ the human cell is the audit verdict and `reason` (short
Chinese) explains the override. `CodeSwitchRatio` not included.

## Sampling regime

The same 90 `item_id`s — fixed seed, stratified 3 family × 3 response language ×
10 per cell — were audited on all four GPT models, so per-model results are
directly comparable. Total audited rows: 90 × 4 = 360.

| Model | Audited | Disagreements | Rate |
|---|---:|---:|---:|
| gpt_4.1_tri_main         | 90 | 18 | 20.0% |
| gpt_5.3_codex_tri_main   | 90 | 19 | 21.1% |
| gpt_5.4_tri_main         | 90 | 18 | 20.0% |
| gpt_5.4_mini_tri_main    | 90 | 14 | 15.6% |
| **Total**                | **360** | **69** | **19.2%** |

Per-component counts (a row may disagree on more than one component):

| Model | SC | LC | CC |
|---|---:|---:|---:|
| gpt_4.1         | 6 | 2 | 11 |
| gpt_5.3_codex   | 6 | 2 | 12 |
| gpt_5.4         | 6 | 2 | 11 |
| gpt_5.4_mini    | 6 | 2 |  7 |

## Patterns observed

1. **Family B `only_output` violation missed by judge** (~30 cases). Model adds
   a wrapper around the gold value, e.g. `0.8 is larger.`, `最终电话号码是 414
   555 0199。`, `El método final de entrega es **subirlo a Moodle**`. SC=1
   stands; CC corrected 1 → 0.
2. **Family B `only_output` false positive** (~10 cases). Output is the gold
   value with only a capital letter and period: `Dr. Rivera.`, `The revised
   chapter and the response memo.`. CC corrected 0 → 1.
3. **Family A SC paraphrase false negative** (6 cases — LMA24_C24 ×3, LMA25_C2
   ×2, LMA26_C17). Stance correct but worded differently from gold. SC 0 → 1.
4. **Family C SC paraphrase false negative** (13 cases — LMC08, LMC10, LMC15,
   LMC21, LMC22 ×3, LMC23_C19 ×4). Main update captured with different
   wording (e.g. `seminar` for `专题讨论会`, `延长至…` for `现为…`). SC 0 → 1.
5. **Family B SC paraphrase boundaries** (no override). 18 cases where the gold
   string is missing but the answer is equivalent (`Viernes a las 3 p. m.` vs
   `el viernes a las 3 p. m.`, `十三` vs `13`, `No, porque al final se canceló`
   vs `No. Se canceló`). Auto already had SC=1 — confirmed.
6. **LC edge cases** (2 items per model). LMA16_C15: quoted source token
   `"佩服"` in an English answer is allowed by rubric (LC 0 → 1). LMC25_C12:
   `Dr. Rivera` and `Maria` are proper names already present in the gold, so
   may stay in a Chinese answer (LC 0 → 1).

## Net effect on JointSuccess (audited sample)

| Model | JS 0→1 | JS 1→0 | Net | Δ pp |
|---|---:|---:|---:|---:|
| gpt_4.1         | 11 |  6 |  +5 | +5.6 |
| gpt_5.3_codex   |  9 | 10 |  -1 | -1.1 |
| gpt_5.4         |  9 |  7 |  +2 | +2.2 |
| gpt_5.4_mini    | 12 |  2 | +10 | +11.1 |
| **Total**       | **41** | **25** | **+16** | **+4.4** |

Net effect is a small rise in JointSuccess, driven by Family A/C SC paraphrase
false-negatives outweighing Family B `only_output` violations the judge let
through. Small between-model differences on this sample should not be
over-interpreted.

## Bottom line

Disagreement rates across the four GPT models range from 15.6% to 21.1%
(aggregate 19.2%), comparable to prior audits (Claude 15.6%, DeepSeek
random-control 16.7%). Disagreements are systematic, mostly Family B
`only_output` boundaries (both directions) and Family A/C SC paraphrase
false-negatives. As before, treat the automatic scores as aggregate diagnostic
indicators and report this audit alongside them.
