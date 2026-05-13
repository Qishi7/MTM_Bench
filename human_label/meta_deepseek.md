# DeepSeek human validation

Models audited:
- `deepseek_r1_tri_main`
- `deepseek_v3_1_tri_main`
- `deepseek_v3_2_tri_main`

Output files:
- `human_label_deepseek_r1.csv`
- `human_label_deepseek_v3_1.csv`
- `human_label_deepseek_v3_2.csv`

Rows where `human_*` is blank mean the automatic label was confirmed; rows where `human_*` is filled mean the human audit overrides the automatic label.

## Sampling regime

Uniform random sample of 30 instances per model from `tri_main`, drawn with a deterministic seed. 3 models × 30 = 90 audited model-output instances total.

| Model | Audited | Disagreements | Rate |
|---|---:|---:|---:|
| deepseek-r1   | 30 |  1 |  3.3% |
| deepseek-v3_1 | 30 |  7 | 23.3% |
| deepseek-v3_2 | 30 |  7 | 23.3% |
| **Total**     | **90** | **15** | **16.7%** |

Overall, I audited 90 DeepSeek outputs and disagreed with the automatic judge on 15 of them, for a disagreement rate of 16.7%. Per-model rates are based on only 30 audited instances each, so the model-level differences should be read as noisy; the main estimate is the aggregate 15 / 90 = 16.7%. The case-level corrections and reasons are in the per-model CSVs.
