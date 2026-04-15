# LLL-Bench: Language-Conditioned Task Execution Benchmark

A benchmark for testing how well multilingual LLMs handle language mismatch across:

```text
(L_instr, L_content, L_resp)
```

In this benchmark, `L_content` is fixed to English. We vary instruction language and response language.

## What collaborators need to do

Each person only needs to run their assigned model.

Your workflow is:

1. Run `smoke_test.csv` first to verify your API key, script, and output format
2. Run `lan_main.csv` for the full experiment
3. Send your output CSV back

The final scoring review and analysis will be done centrally.

---

## Files you need

```text
data/smoke_test.csv           # 12-item smoke test
outputs/smoke_test_result.csv # reference output format example
data/lan_main.csv             # full benchmark (168 instances per model)
```

---

## Benchmark structure

### Task families

- **A: Semantic reversal**
  - tests whether subtle stance reversal is preserved across languages
  - main metric: `SemanticCorrect`

- **B: Final-state extraction**
  - tests whether the model extracts the final valid value, not just any mentioned value
  - main metrics: `SemanticCorrect`, `ConstraintCorrect`

- **D: Language-purity control**
  - tests whether the output stays in the required language and avoids code-switching
  - main metrics: `LangCorrect`, `CodeSwitchRatio`

There are **24 base items total**:
- 8 items in A
- 8 items in B
- 8 items in D

### Regimes

`L_content` is fixed to English.

| Regime | Triplet | Description |
|---|---|---|
| Same-language baseline | `(en, en, en)` | No mismatch |
| Response mismatch | `(en, en, zh)` / `(en, en, es)` | Only response language differs |
| Instruction mismatch | `(zh, en, en)` / `(es, en, en)` | Only instruction language differs |
| Full mismatch | `(zh, en, es)` / `(es, en, zh)` | Instruction and response both differ from content |

So the full benchmark contains:

```text
7 triplets × 24 items = 168 instances per model
```

---

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `SemanticCorrect` | 0/1 | Core meaning is correct |
| `LangCorrect` | 0/1 | Output is in the required response language |
| `ConstraintCorrect` | 0/1 | Required output format is satisfied |
| `CodeSwitchRatio` | float | Non-target-language contamination ratio |
| `JointSuccess` | 0/1 | All three binary metrics are 1 |

---

## Before you run the full benchmark

Please **run the smoke test first**.

Why:
- it checks that your API key works
- it checks that the script is reading the CSV correctly
- it checks that your output format matches the expected format
- it helps avoid wasting tokens on the full run

---

## Installation

Install only what you need for your assigned model.

Examples:

| Model family | Install command |
|---|---|
| Claude | `pip install anthropic` |
| Gemini | `pip install -U google-genai` |
| OpenAI | `pip install openai` |
| Qwen | `pip install dashscope` |
| Together / Llama | `pip install openai` |
| Mistral | `pip install mistralai` |

---

## Set your API key

### Mac / Linux
```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export TOGETHER_API_KEY="..."
export MISTRAL_API_KEY="..."
```

### Windows PowerShell
```powershell
$env:ANTHROPIC_API_KEY = "..."
$env:OPENAI_API_KEY    = "..."
$env:GEMINI_API_KEY    = "..."
$env:DASHSCOPE_API_KEY = "..."
$env:TOGETHER_API_KEY  = "..."
$env:MISTRAL_API_KEY   = "..."
```

---

## Step 1: Smoke test

Use the 12-item smoke test before the full benchmark.

### Example: Claude
```bash
python scripts/run_models.py \
  --items data/smoke_test.csv \
  --model claude-haiku-4-5-20251001 \
  --out outputs/smoke_test_result.csv \
  --no-dry-run
```

### Example: Gemini
```bash
python scripts/run_models_gemini.py \
  --items data/smoke_test.csv \
  --model gemini-2.5-flash \
  --out outputs/smoke_test_result.csv \
  --no-dry-run
```

Check that:
- outputs are non-empty
- there are no obvious `[ERROR]` rows
- the CSV format looks correct

---

## Step 2: Full benchmark

Once the smoke test looks good, run the full benchmark.

### Example
```bash
python scripts/run_models.py \
  --items data/lan_main.csv \
  --model claude-haiku-4-5-20251001 \
  --out outputs/claude_haiku_lan_main.csv \
  --no-dry-run
```

---

## If your model is not supported yet

If there is no existing `run_models_*.py` script for your assigned model, adapt one of the existing runners or add a new runner.

Requirements:
- input must read the same CSV format
- output must follow the same CSV structure as the existing result files
- do not change the benchmark data format

In other words, you only need to make your model runner compatible with the current pipeline.

---

## Token / cost advice

To avoid wasting tokens:

1. Run `smoke_test.csv` first
2. If needed, test only a few rows before running all 168 instances
3. Keep `temperature = 0`
4. Use moderate output limits such as `--max-tokens 256`
5. Per test, only 155 token used for each task, so in total you will use 155x256 tokens

Roughly:
- `smoke_test.csv` is small and cheap
- `lan_main.csv` has **168 instances per model**, so token usage can add up quickly

---

## Output file naming

Please save outputs with clear names, for example:

```text
outputs/claude_haiku_lan_main.csv
outputs/claude_sonnet_lan_main.csv
outputs/claude_opus_lan_main.csv
outputs/gemini_flash_lan_main.csv
```

For smoke test outputs:

```text
outputs/smoke_test_result.csv
```

---

## What to send back

After you finish running your assigned model, send back:

- the raw result CSV for that model

Do **not** send API keys.

Scoring review and final cross-model analysis will be done centrally.

---

## Minimal checklist

- [ ] Install the package for your model
- [ ] Set your API key
- [ ] Run `smoke_test.csv`
- [ ] Confirm the output file looks normal
- [ ] Run `lan_main.csv`
- [ ] Send the result CSV back
