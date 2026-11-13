# MTM-Bench

**A controlled benchmark for multilingual task execution.** MTM-Bench varies the language of the instruction, source content, and required response independently. It measures whether a model can complete the task *and* answer in the requested language.

| Benchmark | Scope |
| --- | ---: |
| Languages | English, Spanish, Chinese |
| Task families | 3 |
| Base items per family | 30 |
| Language conditions per item | 27 |
| Instances per model | 2,430 |
| Evaluated models | 20 |

The complete benchmark is [`data/tri_main.csv`](data/tri_main.csv). Each row includes the prompt, language triplet `(instruction, content, response)`, task metadata, and gold answer or rubric. See the [annotation guidelines](data/annotation_guidelines.md) for task and scoring details.

## Benchmark design

| Family | Task | Success criterion |
| --- | --- | --- |
| A | Semantic reversal | Recover the intended meaning despite misleading surface wording |
| B | Final-state extraction | Return the final valid value after revisions |
| C | Language purity | Produce the requested localized answer without unwanted language carryover |

The primary outcome is **Joint Success (JS)**: semantic correctness, response-language correctness, and constraint correctness must all hold. The scored files also report each component and `CodeSwitchRatio` (called contamination ratio, CTR, in the paper).

![Language-role effects across the benchmark](figures/axes_effects.png)

## Reproduce the results

The released scored outputs make the main analysis **offline and API-free**. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_all_analysis.py
python scripts/make_tri_figures.py \
  --long summary/tri_main_analysis_long.csv \
  --outdir figures
```

`run_all_analysis.py` first rebuilds [`summary/`](summary/) from the 20 original-judge scored files, then regenerates the main tables, bootstrap comparisons, regression table, second-judge sensitivity table, and audit summary in [`results/`](results/). The figure command writes PNG and PDF versions in [`figures/`](figures/). Bootstrap runs may take several minutes.

The [main results table](results/table1_main_results.csv), [language-axis marginals](results/table3_axis_marginals.csv), [model/family/token table](results/table4_model_family_tokens.csv), [regression](results/regression_table.csv), and [judge sensitivity](results/table8_judge_sensitivity.csv) are available directly. The [response-only versus full-mismatch bootstrap](results/bootstrap_resp_only_vs_full_mismatch.csv) and [axis bootstrap](results/bootstrap_axis_ranges.csv) provide the reported uncertainty analyses.

### Language-purity sensitivity check

The Family C targeted check uses **existing outputs only**: 20 models x 30 base items x 3 response-language conditions (`en,en,en`, `en,en,es`, `en,en,zh`) = 1,800 rows. It compares the original language rule with a relaxed rule that rescues an otherwise correct response when `CodeSwitchRatio <= 0.05`.

```bash
python scripts/language_purity_relaxation_check.py \
  --scored_dir "outputs_scored/haiku judge" \
  --out_prefix results/language_purity_relaxation \
  --threshold 0.05 \
  --model_set all
```

See the [sensitivity summary](results/language_purity_relaxation_summary.md) and [per-response results](results/language_purity_relaxation_by_response.csv). This is a threshold-based sensitivity analysis, not an independent human re-adjudication of script-specific contamination.

## Data and code

```text
data/                 Benchmark instances and annotation guidelines
outputs_scored/       20 model files for each of two semantic judges
summary/              Aggregate long and wide analysis data
results/              Reproducible tables and statistical analyses
figures/              Paper figures in PNG and PDF
human_validation/     Anonymized audit labels and summaries
scripts/              Scoring, analysis, and optional model runners
```

[`outputs_scored/haiku judge/`](outputs_scored/haiku%20judge/) holds the **original** Claude Haiku semantic-judge labels used in the main paper analyses. [`outputs_scored/gpt judge/`](outputs_scored/gpt%20judge/) holds GPT-4.1 second-judge labels for Families A and C; language and constraint labels are unchanged. Both directories contain the same model responses, so this is a scoring comparison rather than new inference. The human audit is kept separate from automatic aggregate scores.

To rerun generation or semantic judging, use the provider-specific `scripts/run_models_*.py` runners and `scripts/score_outputs.py`. Those optional stages require provider API access and the relevant SDKs (`openai`, `anthropic`, or `google-genai`); they can differ as model versions change. The included scored outputs are the frozen evidence for the reported analyses.

## License

Released under the [Apache License 2.0](LICENSE.txt). Model weights and provider-owned documentation are not redistributed.
