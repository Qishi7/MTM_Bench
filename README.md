# Multilingual Task Mismatch

A pilot benchmark for evaluating **language mismatch across instruction, content, and response** in multilingual LLMs.

## Core idea
Each example is defined by a triplet:

\[
(L_{instr}, L_{content}, L_{resp})
\]

- `L_instr`: language of the instruction
- `L_content`: language of the content being processed
- `L_resp`: required response language

The pilot focuses on whether a model:
1. completes the task correctly,
2. answers in the required language,
3. avoids obvious code-switching,
4. satisfies both task and language constraints jointly.

## Main metrics
- **TaskCorrect**: whether the task is completed correctly
- **LangCorrect**: whether the output follows the target response language
- **CodeSwitch**: degree of non-target-language mixing
- **JointSuccess**: 1 only if both task and language are correct

## Pilot scope
Suggested starting point:
- Languages: English, Chinese
- Task families: reading comprehension, extraction, constrained summarization
- Regimes:
  - `(L,L,L)` same-language baseline
  - `(A,A,B)` response mismatch
  - `(A,B,A)` content mismatch
- Models: 3 to 4

## Repository structure
```text
multilingual-task-mismatch/
├── README.md
├── data/
│   ├── pilot_items.csv
│   └── annotation_guidelines.md
├── outputs/
│   └── pilot_results.csv
├── scripts/
│   ├── run_models.py
│   ├── score_outputs.py
│   └── summarize_results.py
└── notes/
    └── experiment_log.md
```

## Data files
### `data/pilot_items.csv`
Benchmark items for the pilot.

Key fields:
- `item_id`
- `task_family`
- `difficulty`
- `regime`
- `instruction_language`
- `content_language`
- `response_language`
- `instruction_text`
- `content_text`
- `question_or_prompt`
- `gold_answer_or_rubric`
- `notes`

### `outputs/pilot_results.csv`
Stores raw model outputs and evaluation labels.

Key fields:
- `model`
- `item_id`
- `output_text`
- `TaskCorrect`
- `LangCorrect`
- `CodeSwitch`
- `JointSuccess`
- `evaluator_notes`

## Quick start
### 1. Populate pilot items
Fill in `data/pilot_items.csv` with your examples.

### 2. Run a model
Example:
```bash
python scripts/run_models.py \
  --items data/pilot_items.csv \
  --model demo-model \
  --out outputs/pilot_results.csv
```

By default, the script runs in a **dry-run mode** that creates placeholder outputs. Replace the stub in `scripts/run_models.py` with your actual API call.

### 3. Score outputs
You can manually label `TaskCorrect`, `LangCorrect`, and `CodeSwitch` in the CSV, then compute `JointSuccess` and summary statistics:
```bash
python scripts/score_outputs.py --infile outputs/pilot_results.csv
python scripts/summarize_results.py --infile outputs/pilot_results.csv
```

## Recommended workflow
1. Write 30 to 60 pilot items.
2. Run 3 models.
3. Manually score outputs.
4. Remove items that are too easy, too hard, or poorly specified.
5. Expand to more languages or add `(A,B,C)` full mismatch after calibration.

## Notes
This repo is designed for fast pilot iteration, not for final public benchmark release.
