# DeepSeek human validation

Models I ran and audited:
- `deepseek_r1_lan_main`, `deepseek_r1_tri_main`
- `deepseek_v3_1_lan_main`, `deepseek_v3_1_tri_main`
- `deepseek_v3_2_lan_main`, `deepseek_v3_2_tri_main`

Output: `human_label_deepseek.csv` (132 rows). Columns:
`model, audit_type, source_file, item_id, model_output, SC, LC, CC, JC, , human_SC, human_LC, human_CC, reason`.

Two extras vs the minimax sheet:
- `audit_type` separates the two sampling regimes (`random_control` vs `boundary_targeted`).
- `source_file` is `lan_main` or `tri_main`. Some item_ids exist in both files, so without this column you can't always tell which one a row came from just from `item_id`.

Rows where `human_*` is blank mean the judge label was confirmed; rows where `human_*` is filled mean I'm overriding it.

## Two sampling regimes

I ran two separate audits, kept in the same CSV but tagged with `audit_type`:

### 1. `random_control` — 30 items (5 per file, uniform random)

This is the regime that's directly comparable to owen's and unitystdy's audits.

| File | Audited | Disagreements |
|---|---:|---:|
| deepseek_r1_lan_main | 5 | 0 |
| deepseek_r1_tri_main | 5 | 2 |
| deepseek_v3_1_lan_main | 5 | 1 |
| deepseek_v3_1_tri_main | 5 | 2 |
| deepseek_v3_2_lan_main | 5 | 0 |
| deepseek_v3_2_tri_main | 5 | 0 |
| **Total** | **30** | **5 (16.7%)** |

Net effect on JointSuccess from applying my 5 overrides: **+6.7 percentage points** (3 cases JS 0→1 from SC fixes, 1 case JS 1→0 from a CC false positive, 1 case stays JS=0 because LC still fails).

One of the 5 random-control disagreements is a **judge-too-lenient** case in the opposite direction (LMB19_C15, where the model added "are usually larger than Pekingese" past the "only the final answer" constraint and the judge still gave CC=1). So this is not a one-sided "model deserves more credit" story.

### 2. `boundary_targeted` — 102 items (stratified by score bucket)

For each `(file × category × bucket)` I drew up to 2 items from each of `SC=0`, `LC=0 & SC=1`, `CC=0 & SC=1`, and 1 from all-correct, to inspect both sides of the decision boundary. **This is not a random sample** and the disagreement rate here is not a corpus-wide error rate — it's a human-validation result on cases that were already likely to be fragile.

| File | Audited | Disagreements |
|---|---:|---:|
| deepseek_r1_lan_main | 12 | 5 |
| deepseek_r1_tri_main | 20 | 13 |
| deepseek_v3_1_lan_main | 16 | 8 |
| deepseek_v3_1_tri_main | 21 | 15 |
| deepseek_v3_2_lan_main | 12 | 7 |
| deepseek_v3_2_tri_main | 21 | 17 |
| **Total** | **102** | **65** |

The point of this set is not to estimate a rate but to characterize **what kinds of cases the current judge gets wrong**. See patterns below.

## Patterns observed (in the boundary set, and confirmed in the random set)

1. **CC/LC double-penalty on zh-target "yes/no + brief reason" items.** When a model starts with English `No`/`no` and gives a Chinese reason, the judge often marks both `LC=0` and `CC=0`. The format (yes/no prefix + reason) is satisfied; the English prefix is an LC violation, not also a CC violation. Affects ~18 boundary cases; 1 random-control case (`LMA23_C2`).
2. **SC false negatives on Family A reversal items with valid paraphrased reasoning.** Negative/critical stance correctly captured but with non-gold wording. ~15 boundary cases; 1 random-control case (`LMA18_C3`).
3. **SC false negatives on Family B equivalents.** `回复备忘录` vs `回应备忘录`, correct final value with a harmless unit, etc. ~10 boundary cases; 1 random-control case (`LMB17_C23`).
4. **SC false negatives on Family C summaries with correct extra detail.** ~12 boundary cases; 1 random-control case (`LMD01_C7`).
5. **LC over-penalty on proper names / name-like units** (`Dr. Rivera`, `Johnston Hall`, `Career Center`, even when the gold itself preserves them). ~6 boundary cases.
6. **At least one CC false positive in the other direction.** `LMA16_C19`: model gives only the stance (`否。`) without the required reason but still gets `CC=1`. `LMB19_C15` in the random control is the symmetric case (model adds explanation past "only the final answer" and still gets `CC=1`).

Some boundary cases fall under more than one pattern, so the per-pattern counts overlap and do not sum to 65.

## One false positive (SC=1 → 0)

`LMB23_C24` on `deepseek_v3_1_tri_main`: output is just `Can`, which is a truncated/incomplete answer for `Canadá`. The judge accepted it (SC=1); should be SC=0. This is the only judge-too-lenient SC case I found.

## Bottom line

- The judge agreement rate on a comparable random subsample is **25 / 30 = 83.3%** (16.7% disagreement), which falls in the same ballpark as the team's other audits.
- The disagreements are systematic rather than scattered: about 4/5 of the corrections raise the model's score (mostly SC paraphrase tolerance and CC/LC boundary handling) and ~1/5 lower it (one CC false positive, one SC false positive).
- The targeted boundary audit was useful for characterizing the pattern of judge errors, but should not be quoted as a judge error rate.
