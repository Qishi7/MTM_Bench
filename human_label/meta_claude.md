# Claude human validation

Models audited:
- `claude_haiku_tri_main`
- `claude_sonnet_tri_main`
- `claude_opus_tri_main`

Output files:
- `human_label_claude_haiku.csv`
- `human_label_claude_sonnet.csv`
- `human_label_claude_opus.csv`

Columns:
`model, item_id, model_output, SC, LC, CC, JC, , human_SC, human_LC, human_CC, reason`.

Rows where `human_*` is blank mean the automatic label was confirmed.
Rows where `human_*` is filled mean the human audit overrides the automatic label.
`ContaminationRatio` was not included in the disagreement calculation.

## Sampling regime

Each Claude model was audited on a stratified 90-instance sample:

- 3 task families: A, B, C
- 3 response languages: English, Spanish, Chinese
- 10 instances per family--response-language cell

This gives 90 audited instances per model and 270 audited Claude instances in total.

| Model | Audited | Disagreements | Disagreement rate |
|---|---:|---:|---:|
| claude_haiku_tri_main | 90 | 14 | 15.6% |
| claude_sonnet_tri_main | 90 | 16 | 17.8% |
| claude_opus_tri_main | 90 | 12 | 13.3% |
| **Total** | **270** | **42** | **15.6%** |

## Patterns observed

1. **Family B only-output boundary cases.**
   Some final-state extraction outputs were semantically correct but included an explanatory shell, such as
   “The final submission method is Moodle” or “0.8 is larger than 0.65.”
   These should have `SC=1` but `CC=0` under the strict `only_output` constraint.

2. **Family B short-answer false negatives.**
   Some concise final answers were incorrectly marked as constraint failures, especially Chinese answers with light punctuation,
   such as “周四，3号厅.” These should remain `CC=1` when they only express the final value.

3. **Family C semantic false negatives.**
   Several update-realization outputs correctly expressed the core change, such as “from old value to new value,”
   but were marked `SC=0`. Under the rubric, old-to-new framing is not a contradiction when the final state is clear.

4. **Family A semantic false negatives.**
   A small number of semantic-reversal outputs correctly identified the real evaluative stance but were marked `SC=0`.
   These were much less frequent than the Family B and C boundary cases.

5. **SC, LC, and CC separation.**
   Some boundary cases show why the three component labels must remain separate:
   language mixing should affect `LC`, formatting should affect `CC`, and neither should automatically change `SC`.

## Net effect on JointSuccess

Applying the human overrides would slightly increase the Claude models' audited-sample JointSuccess overall,
because the Family C semantic false negatives outnumber the Family B only-output false positives in this sample.
The effect should be interpreted as an audit finding rather than a replacement for full-corpus rescoring.

## Bottom line

Across the three Claude models, binary-label disagreement rates range from 13.3% to 17.8%, with an aggregate rate of 15.6%.
The disagreements are systematic rather than scattered, mainly involving Family B strict output constraints and Family C update-realization semantics.
These results support using the automatic scores as aggregate diagnostic indicators, while reporting human audit results and avoiding over-interpretation of small model-ranking differences.
