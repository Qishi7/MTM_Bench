# LLL-Bench: Language-Conditioned Task Execution Benchmark

A benchmark for evaluating **language mismatch across instruction, content, and response** in multilingual LLMs.

## Core idea

Each example is defined by a language triplet:

```
(L_instr, L_content, L_resp)
```

- `L_instr` — language of the task instruction
- `L_content` — language of the content being processed (fixed to **English** in this benchmark)
- `L_resp` — required response language

We evaluate whether a model:
1. completes the task correctly (**SemanticCorrect**)
2. answers in the required language (**LangCorrect**)
3. satisfies explicit format/output constraints (**ConstraintCorrect**)
4. avoids code-switching (**CodeSwitchRatio**)
5. satisfies all of the above jointly (**JointSuccess**)

---

## Metrics

| Metric | Type | Description |
|---|---|---|
| `SemanticCorrect` | 0/1 | Core factual content matches gold answer |
| `LangCorrect` | 0/1 | Output is predominantly in `L_resp` |
| `ConstraintCorrect` | 0/1 | Format/output constraints satisfied |
| `CodeSwitchRatio` | float [0,1] | Proportion of non-target-language tokens |
| `JointSuccess` | 0/1 | All three binary metrics = 1 |

---

## Benchmark scope

**Languages:** English (en), Chinese (zh), Spanish (es)

**Task families:**

| Family | Label | What it tests |
|---|---|---|
| Semantic reversal | A | Whether cross-lingual processing preserves subtle polarity (irony, understatement, reversed stance) |
| Final-state extraction | B | Whether the model extracts the *latest* valid value (deadline, contact, room) not just any mentioned value |
| Language-purity control | D | Whether the output respects strict language-purity rules (proper nouns may stay, everything else must be in `L_resp`) |

8 items per family → **24 base items**

**Mismatch regimes** (L_content fixed to `en`):

| Regime | Triplet | Description |
|---|---|---|
| Same-language baseline | `(en, en, en)` | No mismatch |
| Response mismatch | `(en, en, zh)` / `(en, en, es)` | Only response language differs |
| Instruction mismatch | `(zh, en, en)` / `(es, en, en)` | Only instruction language differs |
| Full mismatch | `(zh, en, es)` / `(es, en, zh)` | Instruction and response both differ from content |

7 distinct triplets × 24 items = **168 instances per model**

---

## Repository structure

```
lll-bench/
├── README.md
├── data/
│   ├── items.csv                  # 24 base benchmark items
│   └── annotation_guidelines.md  # item authoring and scoring guidelines
├── outputs/                       # one CSV per model run (auto-created)
│   └── {model}_scored.csv
├── scripts/
│   ├── run_models.py              # Anthropic (Claude)
│   ├── run_models_gemini.py       # Google Gemini
│   ├── run_models_openai.py       # OpenAI (GPT-4o, o1, ...)
│   ├── run_models_qwen.py         # Alibaba Qwen (DashScope)
│   ├── run_models_together.py     # Llama via Together AI
│   ├── run_models_mistral.py      # Mistral
│   └── score_outputs.py           # automated scoring pipeline
└── notes/
    └── experiment_log.md
```

---

## Data format

### `data/items.csv`

Each row is one base item. The same item is reused across all 7 triplets by varying the instruction text and gold answer at run time.

| Field | Description |
|---|---|
| `item_id` | Unique ID, e.g. `LM001` |
| `task_family` | `A`, `B`, or `D` |
| `difficulty` | `easy`, `medium`, `hard` |
| `instruction_language` | `en`, `zh`, or `es` (of the instruction variant) |
| `content_language` | always `en` |
| `response_language` | `en`, `zh`, or `es` |
| `instruction_text` | Full instruction in `instruction_language` |
| `content_text` | English source text |
| `question` | Question or prompt appended after content |
| `gold_answer_or_rubric` | Reference answer in `response_language` |
| `semantic_keywords` | Comma-separated core keywords for scoring (optional) |
| `notes` | Authoring notes |

### `outputs/{model}_scored.csv`

| Field | Description |
|---|---|
| `model` | Model identifier string |
| `item_id` | Matches `items.csv` |
| `output_text` | Raw model output |
| `input_tokens` | Token count (from API) |
| `output_tokens` | Token count (from API) |
| `total_tokens` | Token count (from API) |
| `SemanticCorrect` | 0 or 1 |
| `LangCorrect` | 0 or 1 |
| `ConstraintCorrect` | 0 or 1 |
| `CodeSwitchRatio` | float |
| `JointSuccess` | 0 or 1 |
| `evaluator_notes` | Auto or manual notes |

---

## Quick start (for everyone)

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd lll-bench
```

Install only the package for the model you are running:

| Model family | Install command |
|---|---|
| Claude | `pip install anthropic` |
| Gemini | `pip install -U google-genai` |
| OpenAI / GPT-4o | `pip install openai` |
| Qwen | `pip install dashscope` |
| Llama (Together) | `pip install openai` |
| Mistral | `pip install mistralai` |

### Step 2 — Set your API key

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export TOGETHER_API_KEY="..."
export MISTRAL_API_KEY="..."
```

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
# etc.
```

### Step 3 — Dry run (verify setup, no API calls)

```bash
python scripts/run_models.py \
  --items data/items.csv \
  --model claude-haiku-4-5-20251001 \
  --out outputs/haiku_raw.csv
```

Output rows will contain `[DRY RUN]` placeholders. No API key needed.

### Step 4 — Real run

Add `--no-dry-run`:

```bash
# Claude
python scripts/run_models.py \
  --items data/items.csv \
  --model claude-haiku-4-5-20251001 \
  --out outputs/haiku_raw.csv \
  --no-dry-run

# Gemini
python scripts/run_models_gemini.py \
  --items data/items.csv \
  --model gemini-2.5-flash \
  --out outputs/gemini_raw.csv \
  --no-dry-run

# GPT-4o
python scripts/run_models_openai.py \
  --items data/items.csv \
  --model gpt-4o \
  --out outputs/gpt4o_raw.csv \
  --no-dry-run

# Qwen
python scripts/run_models_qwen.py \
  --items data/items.csv \
  --model qwen-max \
  --out outputs/qwen_raw.csv \
  --no-dry-run

# Llama 3 via Together
python scripts/run_models_together.py \
  --items data/items.csv \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --out outputs/llama_raw.csv \
  --no-dry-run

# Mistral
python scripts/run_models_mistral.py \
  --items data/items.csv \
  --model mistral-large-latest \
  --out outputs/mistral_raw.csv \
  --no-dry-run
```

Each script prints per-item progress and a token summary on completion.

### Step 5 — Score outputs

```bash
python scripts/score_outputs.py \
  --items data/items.csv \
  --results outputs/haiku_raw.csv \
  --out outputs/haiku_scored.csv \
  --anthropic-api-key $ANTHROPIC_API_KEY
```

The scorer uses an LLM judge for `SemanticCorrect` (requires Anthropic key). Results are cached in `llm_judge_cache.json` — the same output is never sent to the API twice. Without a key, the scorer falls back to keyword overlap.

### Step 6 — Send back your output file

Share the file `outputs/{model}_scored.csv` with the project lead. **Do not share your API key.**

---

## For collaborators: what you need to do

This section is a checklist. You are responsible for exactly one model (or one model family).

### Your responsibilities

- [ ] Run the dry run to confirm your setup works
- [ ] Run the real experiment using your assigned model and script
- [ ] Verify the output file looks reasonable (no `[ERROR]` rows, output text is non-empty)
- [ ] Run `score_outputs.py` on your results
- [ ] Send the scored CSV to the project lead

### What you do NOT need to do

- You do not need to modify `items.csv`
- You do not need to write any new code
- You do not need to run scoring for other models
- You do not need to share your API key with anyone

### Common issues

**`[ERROR] AuthenticationError`** — your API key is not set correctly. Re-run the `export` / `$env:` command in the same terminal window.

**`[ERROR] RateLimitError`** — increase `--sleep-seconds` (default is 1.0). Try `--sleep-seconds 3`.

**Empty `output_text` rows** — some models return empty strings for very constrained prompts. This is expected for a small number of items; `SemanticCorrect` will be 0 automatically.

**Script exits immediately** — check that `data/items.csv` exists and has a header row.

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--max-tokens` | 256 | Max output tokens per call |
| `--temperature` | 0.0 | Sampling temperature |
| `--sleep-seconds` | 1.0 | Delay between API calls |

---

## Notes

- `L_content` is fixed to English across all items. The mismatch axes are `L_instr` and `L_resp`.
- Items are authored natively in each language, not machine-translated.
- The LLM judge cache (`llm_judge_cache.json`) is safe to commit — it contains only hashes and 0/1 verdicts, no raw model outputs.
- If you rerun scoring on the same file, cached results are reused automatically.
