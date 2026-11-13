import os
import glob
import math
import pandas as pd
import numpy as np

AUDIT_DIR = '/mnt/data/audit_zip/human_label'
SUPP_DIR = '/mnt/data/mtm_supp/MTM_Ano'
OUT = '/mnt/data/targeted_audit_cases.csv'
SUMMARY_OUT = '/mnt/data/targeted_audit_summary.csv'

MODEL_MAP = {
    'claude-haiku-4-5': 'claude-haiku-4-5',
    'claude-opus-4-1': 'claude-opus-4-1',
    'claude-sonnet-4-6': 'claude-sonnet-4-6',
    'deepseek-r1': 'deepseek/deepseek-r1',
    'deepseek-v3_1': 'deepseek/deepseek-chat-v3.1',
    'deepseek-v3_2': 'deepseek/deepseek-v3.2',
    'gpt-4.1': 'gpt-4.1',
    'gpt-5.3-codex': 'gpt-5.3-codex',
    'gpt-5.4': 'gpt-5.4',
    'gpt-5.4-mini': 'gpt-5.4-mini',
    'minimax-m2.5': 'minimax-m2.5',
    'minimax-m2.7': 'minimax-m2.7',
    'qwen3-max': 'qwen3-max',
    'qwen3.5-plus': 'qwen3.5-plus',
    'qwen3.5-flash': 'qwen3.5-flash',
    'gemini-2.5-flash-lite': 'gemini-2.5-flash-lite',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'llama3.1:8b': 'llama3.1:8b',
    'accounts/fireworks/models/llama-v3p3-70b-instruct': 'accounts/fireworks/models/llama-v3p3-70b-instruct',
    'llama3:latest': 'llama3:latest',
}

FILE_MODEL_MAP = {
    'human_label_gemini_2.5_flash_lite.csv': 'gemini-2.5-flash-lite',
    'human_label_gemini_2.5_flash_no_think.csv': 'gemini-2.5-flash',
    'human_label_llama3.1_8b.csv': 'llama3.1:8b',
    'human_label_llama3.3_70b_fireworks.csv': 'accounts/fireworks/models/llama-v3p3-70b-instruct',
    'human_label_llama3.csv': 'llama3:latest',
}

# Load scored outputs to recover metadata and output text where audit sheets omit them.
scored_rows = {}
for f in glob.glob(os.path.join(SUPP_DIR, 'outputs_scored', '*_tri_main_scored.csv')):
    if os.path.basename(f).startswith('token_usage'):
        continue
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        scored_rows[(str(r['model']), str(r['item_id']))] = r.to_dict()


def first_existing(row, names, default=np.nan):
    for name in names:
        if name in row.index:
            val = row[name]
            if not (pd.isna(val) if not isinstance(val, str) else val == ''):
                return val
    return default


def to_int01(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {'true', 'yes', 'y'}:
            return 1
        if s in {'false', 'no', 'n'}:
            return 0
        if s == '':
            return np.nan
        try:
            return int(float(s))
        except Exception:
            return np.nan
    try:
        return int(float(x))
    except Exception:
        return np.nan


def infer_family(item_id):
    if not isinstance(item_id, str) or len(item_id) < 3:
        return ''
    fam = item_id[:3]
    if fam == 'LMA':
        return 'A'
    if fam == 'LMB':
        return 'B'
    if fam in {'LMC', 'LMD'}:
        return 'C'
    return ''


def override_type(auto_sc, auto_lc, auto_cc, human_sc, human_lc, human_cc):
    changed = []
    if auto_sc != human_sc:
        changed.append('semantic')
    if auto_lc != human_lc:
        changed.append('language')
    if auto_cc != human_cc:
        changed.append('constraint')
    if not changed:
        return 'none'
    if len(changed) == 1:
        return changed[0]
    return 'multiple:' + '+'.join(changed)


def make_record(row, source_file, explicit_model=None):
    item_id = str(first_existing(row, ['item_id', 'Item_ID'])).strip()
    raw_model = explicit_model if explicit_model is not None else first_existing(row, ['model', 'Model'], '')
    raw_model = str(raw_model).strip()
    model = MODEL_MAP.get(raw_model, raw_model)
    if not model:
        model = FILE_MODEL_MAP.get(os.path.basename(source_file), '')

    meta = scored_rows.get((model, item_id), {})

    # Auto labels. Prefer the audit sheet columns, and fall back to the full
    # scored-output file when the audit sheet leaves confirmed rows blank.
    auto_sc = to_int01(first_existing(row, ['SC', 'Current_SC'], meta.get('SemanticCorrect', np.nan)))
    auto_lc = to_int01(first_existing(row, ['LC', 'Current_LC'], meta.get('LangCorrect', np.nan)))
    auto_cc = to_int01(first_existing(row, ['CC', 'Current_CC'], meta.get('ConstraintCorrect', np.nan)))
    auto_js = to_int01(first_existing(row, ['JC', 'JC ', 'Current_JS'], meta.get('JointSuccess', np.nan)))
    if pd.isna(auto_js) and not any(pd.isna(x) for x in [auto_sc, auto_lc, auto_cc]):
        auto_js = int(auto_sc * auto_lc * auto_cc)

    # Human labels. Blank human cells mean the automatic label was confirmed.
    hsc_raw = first_existing(row, ['human_SC', 'human_SemanticCorrect', 'SemanticCorrect', 'Human_SC'])
    hlc_raw = first_existing(row, ['human_LC', 'human_LangCorrect', 'LangCorrect', 'Human_LC'])
    hcc_raw = first_existing(row, ['human_CC', 'human_ConstraintCorrect', 'ConstraintCorrect', 'Human_CC'])
    human_sc = to_int01(hsc_raw)
    human_lc = to_int01(hlc_raw)
    human_cc = to_int01(hcc_raw)
    if pd.isna(human_sc):
        human_sc = auto_sc
    if pd.isna(human_lc):
        human_lc = auto_lc
    if pd.isna(human_cc):
        human_cc = auto_cc
    human_js = int(human_sc * human_lc * human_cc) if not any(pd.isna(x) for x in [human_sc, human_lc, human_cc]) else np.nan

    task_family = meta.get('task_family', first_existing(row, ['Task_Family'], ''))
    response_language = meta.get('response_language', first_existing(row, ['Response_Language'], ''))
    output_text = first_existing(row, ['model_output', 'Model_Output'], meta.get('output_text', ''))
    reason = first_existing(row, ['reason', 'notes', 'notes.1', 'adjudication_note', 'Evaluator_Notes'], '')

    return {
        'audit_subset': 'targeted',
        'model': model,
        'instance_id': item_id,
        'core_id': meta.get('core_id', item_id.split('_')[0] if '_' in item_id else ''),
        'task_family': task_family,
        'response_language': response_language,
        'auto_sc': int(auto_sc) if not pd.isna(auto_sc) else '',
        'auto_lc': int(auto_lc) if not pd.isna(auto_lc) else '',
        'auto_cc': int(auto_cc) if not pd.isna(auto_cc) else '',
        'human_sc': int(human_sc) if not pd.isna(human_sc) else '',
        'human_lc': int(human_lc) if not pd.isna(human_lc) else '',
        'human_cc': int(human_cc) if not pd.isna(human_cc) else '',
        'auto_js': int(auto_js) if not pd.isna(auto_js) else '',
        'human_js': int(human_js) if not pd.isna(human_js) else '',
        'override_type': override_type(auto_sc, auto_lc, auto_cc, human_sc, human_lc, human_cc) if not any(pd.isna(x) for x in [auto_sc, auto_lc, auto_cc, human_sc, human_lc, human_cc]) else '',
        'source_file': os.path.basename(source_file),
        'output_text': output_text,
        'reason_category_or_note': reason,
    }

records = []
# CSV audit files. Exclude hm_* legacy workbooks and unflagged check.
for f in sorted(glob.glob(os.path.join(AUDIT_DIR, 'human_label_*.csv'))):
    name = os.path.basename(f)
    if name == 'unflagged_sanity_check.csv':
        continue
    df = pd.read_csv(f, encoding='utf-8-sig')
    explicit_model = FILE_MODEL_MAP.get(name)
    for _, row in df.iterrows():
        records.append(make_record(row, f, explicit_model=explicit_model))

# Qwen workbook.
qwen_file = os.path.join(AUDIT_DIR, 'human_label_qwen.xlsm')
if os.path.exists(qwen_file):
    qdf = pd.read_excel(qwen_file, sheet_name='Audit_Records', engine='openpyxl')
    for _, row in qdf.iterrows():
        records.append(make_record(row, qwen_file))

out = pd.DataFrame(records)
# Drop blank separator rows carried over from spreadsheet exports.
out = out[out['instance_id'].notna() & (out['instance_id'].astype(str).str.lower() != 'nan') & (out['instance_id'].astype(str).str.strip() != '')]
# Drop exact duplicate rows if any.
out = out.drop_duplicates(subset=['model', 'instance_id', 'source_file']).reset_index(drop=True)
# Sort for readability.
out = out.sort_values(['model', 'instance_id', 'source_file']).reset_index(drop=True)

# Save.
out.to_csv(OUT, index=False, encoding='utf-8')

# Summary.
summary_rows = []
for subset_name, g in [('targeted', out)]:
    n = len(g)
    comp_override = ((g['auto_sc'] != g['human_sc']) | (g['auto_lc'] != g['human_lc']) | (g['auto_cc'] != g['human_cc'])).sum()
    js_override = (g['auto_js'] != g['human_js']).sum()
    summary_rows.append({
        'audit_subset': subset_name,
        'n': n,
        'component_overrides': int(comp_override),
        'component_override_rate': comp_override / n if n else np.nan,
        'js_overrides': int(js_override),
        'js_override_rate': js_override / n if n else np.nan,
        'sc_agreement': (g['auto_sc'] == g['human_sc']).mean(),
        'lc_agreement': (g['auto_lc'] == g['human_lc']).mean(),
        'cc_agreement': (g['auto_cc'] == g['human_cc']).mean(),
        'overall_component_agreement': ((g['auto_sc'] == g['human_sc']) & (g['auto_lc'] == g['human_lc']) & (g['auto_cc'] == g['human_cc'])).mean(),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False, encoding='utf-8')
print(f'Wrote {OUT}: {out.shape}')
print(f'Wrote {SUMMARY_OUT}: {summary.shape}')
print(summary.to_string(index=False))
print('\nRows by source file:')
print(out['source_file'].value_counts().sort_index().to_string())
print('\nRows by model:')
print(out['model'].value_counts().sort_index().to_string())
