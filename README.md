# MTM Bench

# MTM Bench

MTM Bench is a controlled benchmark for evaluating language-conditioned task execution in multilingual LLMs. Each instance is defined by a language triplet:

`(L_instr, L_content, L_resp)`

where the instruction language, content language, and required response language may differ.

The benchmark varies English, Spanish, and Chinese across all triplet combinations and evaluates whether models can both solve the task and produce the answer in the requested language.

## Benchmark

The main benchmark file is:

```text
data/tri_main.csv
```

It contains 2,430 instances:

```text
3 task families × 30 base items × 27 language triplets
```

The three task families are:

| Family | Task                   | Goal                                                            |
| ------ | ---------------------- | --------------------------------------------------------------- |
| A      | Semantic reversal      | Recover the true stance rather than the surface wording         |
| B      | Final-state extraction | Extract the final valid value after updates or revisions        |
| C      | Language purity        | Produce a localized update without English structural carryover |

## Metrics

Model outputs are scored with five metrics:

| Metric              | Meaning                                                   |
| ------------------- | --------------------------------------------------------- |
| `SemanticCorrect`   | Whether the output captures the required meaning          |
| `LangCorrect`       | Whether the output follows the required response language |
| `ConstraintCorrect` | Whether explicit output constraints are satisfied         |
| `CodeSwitchRatio`   | Degree of non-target-language contamination               |
| `JointSuccess`      | Product of semantic, language, and constraint correctness |

## Running models

Example:

```powershell
py .\scripts\run_models.py `
  --items .\data\tri_main.csv `
  --model MODEL_NAME `
  --out .\outputs\MODEL_tri_main.csv `
  --no-dry-run `
  --max-tokens 256 `
  --temperature 0
```

## Scoring outputs

Example:

```powershell
py .\scripts\score_outputs.py `
  --items .\data\tri_main.csv `
  --results .\outputs\MODEL_tri_main.csv `
  --out .\outputs_scored\MODEL_tri_main_scored.csv `
  --anthropic-api-key $env:ANTHROPIC_API_KEY `
  --cache-file llm_judge_cache.json
```

Families A and C use an LLM-assisted semantic judge. Family B is scored with deterministic final-state rules. Language and constraint metrics are rule-based.

## Analysis

To summarize scored outputs:

```powershell
py .\scripts\analyze_results.py `
  --scored-dir .\outputs_scored `
  --pattern "*tri_main_scored.csv"
```

## Current results

The full scored outputs are stored in:

```text
outputs_scored/
```

The current main analysis is based on 20 models evaluated on the full 2,430-instance benchmark.

Main finding: response language is the strongest source of degradation, while different task families fail through different channels.
