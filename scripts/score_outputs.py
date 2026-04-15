import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
import urllib.request
import urllib.error
from typing import Dict, List, Set, Tuple


SPANISH_FUNCTION_WORDS = {
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "no", "se", "un", "una",
    "por", "para", "con", "como", "su", "sus", "al", "lo", "es", "fue", "era", "pero",
    "si", "sí", "porque", "aunque", "más", "menos", "esto", "esta", "estas", "estos",
    "le", "les", "ya", "muy", "solo", "sólo", "realmente", "autor", "pasaje",
}

ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "not", "no", "that", "this", "these",
    "those", "it", "its", "as", "by", "from", "at", "than", "rather", "because",
    "what", "who", "when", "where", "while", "although", "then", "still", "finally",
}

EN_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

PROPER_NAME_ALLOWLIST = {
    "databridge", "campusconnect", "campuslink", "eventportal", "eventgrid",
    "openpath", "mediahub", "greenlab", "rivernet", "linkdesk",
    "rivera", "chen", "professor", "room", "hall", "building",
    "q", "a",
}

EN_CONTAMINATION_TOKENS = {
    "am", "pm", "room", "building", "hall", "deadline", "report", "proposal",
    "budget", "form", "teacher", "summary",
} | EN_MONTHS

# Explanatory prefixes banned in only_output tasks
_EXPLANATORY_PREFIXES = (
    "the answer is",
    "la respuesta es",
    "最终答案是",
    "the final deadline is",
    "the final room number is",
    "the final room is",
    "the final files are",
    "the final person is",
    "the person to contact is",
    "最终需要提交的是",
    "最终截止时间是",
    "最终联系人是",
    "the deadline is",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-score outputs with SemanticCorrect(0/1), LangCorrect(0/1), "
                    "ConstraintCorrect(0/1), CodeSwitchRatio, JointSuccess(0/1)."
    )
    parser.add_argument("--items",   required=True, help="Path to benchmark items CSV")
    parser.add_argument("--results", required=True, help="Path to model outputs CSV")
    parser.add_argument("--out",     required=True, help="Path to scored output CSV")
    parser.add_argument(
        "--anthropic-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key for LLM-based semantic scoring (falls back to keyword method if absent)",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for semantic judgement (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--cache-file",
        default="llm_judge_cache.json",
        help="Path to JSON cache for LLM judge results (default: llm_judge_cache.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def index_items(items: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["item_id"].strip(): row for row in items if row.get("item_id", "").strip()}


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM judge cache
# ---------------------------------------------------------------------------

def _cache_key(output: str, gold: str) -> str:
    raw = json.dumps({"output": output, "gold": gold}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_cache(path: str) -> Dict[str, int]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(path: str, cache: Dict[str, int]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:'[A-Za-zÀ-ÿ0-9]+)?", text)


def normalized_tokens(text: str) -> List[str]:
    return tokenize_words(normalize_text(text))


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    chunks = re.split(r"[.!?。！？]+", stripped)
    return len([c for c in chunks if c.strip()])


def starts_with_yes_no(text: str, target_lang: str) -> bool:
    stripped = normalize_text(text.strip())
    if target_lang == "en":
        return stripped.startswith("yes") or stripped.startswith("no")
    if target_lang == "es":
        return stripped.startswith("sí") or stripped.startswith("si") or stripped.startswith("no")
    return False


def looks_like_list_or_heading(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("#"):
        return True
    if re.search(r"(?m)^\s*[-*•]\s+", stripped):
        return True
    if re.search(r"(?m)^\s*\d+\.\s+", stripped):
        return True
    return False


def english_like_token(token: str) -> bool:
    tl = normalize_text(token)
    if tl in ENGLISH_FUNCTION_WORDS or tl in EN_CONTAMINATION_TOKENS:
        return True
    if re.fullmatch(r"(am|pm)", tl):
        return True
    return False


def has_explanatory_prefix(text: str) -> bool:
    norm = normalize_text(text.strip())
    return any(norm.startswith(p) for p in _EXPLANATORY_PREFIXES)


# ---------------------------------------------------------------------------
# LangCorrect
# ---------------------------------------------------------------------------

def lang_correct_and_codeswitch_ratio(
    text: str,
    target_lang: str,
    constraint_correct_val: int = -1,
) -> Tuple[int, float]:
    """
    Change vs previous version:
    - es: ultra-short answers (≤ 3 word-tokens) get a ConstraintCorrect-based
      exemption.  If ConstraintCorrect=1 and there is no CJK and no English
      contamination, accept as language-correct even without Spanish signal words.
      Covers legitimate extraction answers like "Plan C", "327", "No.".
    """
    stripped = text.strip()
    if not stripped:
        return 0, 0.0

    tokens = tokenize_words(stripped)
    total_tokens = max(len(tokens), 1)

    # ── Chinese ──────────────────────────────────────────────────────────────
    if target_lang == "zh":
        if not contains_cjk(stripped):
            return 0, 1.0
        english_like = sum(
            1 for tok in tokens
            if re.fullmatch(r"[a-z0-9]+", normalize_text(tok))
            and normalize_text(tok) not in PROPER_NAME_ALLOWLIST
        )
        ratio = english_like / total_tokens
        lang_ok = 1 if ratio <= 0.20 else 0
        return lang_ok, round(ratio, 4)

    # ── Spanish ──────────────────────────────────────────────────────────────
    if target_lang == "es":
        if contains_cjk(stripped):
            return 0, 1.0

        spanish_hits = 0
        english_contam = 0
        for tok in tokens:
            tl = normalize_text(tok)
            if tl in SPANISH_FUNCTION_WORDS or re.search(r"[áéíóúñ]", tok.lower()):
                spanish_hits += 1
            if english_like_token(tok) and tl not in PROPER_NAME_ALLOWLIST:
                english_contam += 1

        spanish_ratio = spanish_hits / total_tokens
        cs_ratio = round(min(1.0, english_contam / total_tokens), 4)

        # Ultra-short extraction exemption (≤ 3 word-tokens)
        if total_tokens <= 3:
            if constraint_correct_val == 1 and english_contam == 0:
                return 1, cs_ratio
            lang_ok = 1 if spanish_hits >= 1 else 0
            return lang_ok, cs_ratio

        if total_tokens <= 4:
            lang_ok = 1 if spanish_hits >= 1 else 0
        else:
            lang_ok = 1 if spanish_ratio >= 0.15 else 0

        return lang_ok, cs_ratio

    # ── English ──────────────────────────────────────────────────────────────
    if target_lang == "en":
        if contains_cjk(stripped):
            return 0, 1.0

        english_hits = 0
        spanish_like = 0
        for tok in tokens:
            tl = normalize_text(tok)
            if tl in ENGLISH_FUNCTION_WORDS or tl in EN_MONTHS:
                english_hits += 1
            if tl in SPANISH_FUNCTION_WORDS or re.search(r"[áéíóúñ]", tok.lower()):
                spanish_like += 1

        cs_ratio = round(spanish_like / total_tokens, 4)

        if total_tokens <= 4:
            lang_ok = 1 if english_hits >= 1 else 0
        else:
            lang_ok = 1 if (english_hits / total_tokens) >= 0.15 else 0

        return lang_ok, cs_ratio

    return 0, 0.0


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def infer_constraint_type(item: Dict[str, str]) -> str:
    instruction = (item.get("instruction_text") or "").lower()
    question    = (item.get("question") or "").lower()
    full = instruction + " " + question

    if "yes or no plus one brief reason" in full or "先写sí或no" in full or "先写yes或no" in full:
        return "yes_no_brief"
    if (
        "only the final deadline" in full
        or "只用西班牙语回答最终截止时间" in full
        or "only the final room number" in full
        or "only the final files" in full
        or "最终需要提交的文件" in full
        or "only the person" in full
        or "最终应联系谁" in full
        or "do not add explanation" in full
        or "不要补充说明" in full
        or "不要输出流程" in full
    ):
        return "only_output"
    if "one sentence" in full or "一句话" in full or "only one sentence" in full:
        return "one_sentence"
    if "do not use the word" in full or "不得使用" in full or "不要使用" in full:
        return "banned_word"
    if "must begin with" in full or "必须以" in full:
        return "starts_with"
    return "generic"


def extract_max_words(item: Dict[str, str]) -> int:
    full = ((item.get("instruction_text") or "") + " " + (item.get("question") or "")).lower()
    for patt in [r"at most (\d+) words", r"no more than (\d+) words", r"不超过(\d+)个词"]:
        m = re.search(patt, full)
        if m:
            return int(m.group(1))
    return 0


def extract_banned_words(item: Dict[str, str]) -> List[str]:
    full = ((item.get("instruction_text") or "") + " " + (item.get("question") or "")).lower()
    banned: List[str] = []
    banned.extend(re.findall(r"do not use the word\s+([A-Za-zÀ-ÿ]+)", full))
    banned.extend(re.findall(r"不要使用[\"'\u201c\u201d\u2018\u2019]?([A-Za-z\u00c0-\u00ff\u4e00-\u9fff]+)[\"'\u201c\u201d\u2018\u2019]?", full))
    banned.extend(re.findall(r"不得使用[\"'\u201c\u201d\u2018\u2019]?([A-Za-z\u00c0-\u00ff\u4e00-\u9fff]+)[\"'\u201c\u201d\u2018\u2019]?", full))
    return [normalize_text(b) for b in banned]


def extract_required_prefix(item: Dict[str, str]) -> str:
    instr = item.get("instruction_text") or ""
    m = re.search(r"must begin with\s+['\"]([^'\"]+)['\"]", instr, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"必须以\s*[\u201c\u201d\u2018\u2019]([^\u201c\u201d\u2018\u2019]+)[\u201c\u201d\u2018\u2019]开头", instr)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# ConstraintCorrect
# ---------------------------------------------------------------------------

def constraint_correct(text: str, item: Dict[str, str]) -> int:
    """
    Changes vs previous version:
    1. only_output: block explanatory prefixes ("the answer is", "la respuesta es", etc.)
    2. yes_no_brief: relaxed — removed hard token-count cap.
       Constraint is: starts with Yes/No + at most 2 sentences. Length is secondary.
    """
    stripped = text.strip()
    if not stripped or stripped.startswith("[ERROR]"):
        return 0

    ctype = infer_constraint_type(item)

    if looks_like_list_or_heading(stripped):
        return 0

    # Word-count cap
    max_words = extract_max_words(item)
    if max_words > 0:
        if len(tokenize_words(stripped)) > max_words + 2:
            return 0

    # Banned words
    banned_words = extract_banned_words(item)
    lowered = normalize_text(stripped)
    if banned_words and any(bw in lowered for bw in banned_words):
        return 0

    # Required prefix
    required_prefix = extract_required_prefix(item)
    if required_prefix and not stripped.startswith(required_prefix):
        return 0

    # Type-specific

    if ctype == "yes_no_brief":
        target_lang = (item.get("response_language") or "").strip().lower()
        if not starts_with_yes_no(stripped, target_lang):
            return 0
        # At most 2 sentences; hard token cap removed
        if count_sentences(stripped) > 2:
            return 0
        return 1

    if ctype == "only_output":
        # Block explanatory prefixes
        if has_explanatory_prefix(stripped):
            return 0
        if count_sentences(stripped) > 2:
            return 0
        if len(stripped.splitlines()) > 2:
            return 0
        if ":" in stripped and len(tokenize_words(stripped)) > 10:
            return 0
        if len(tokenize_words(stripped)) > 20:
            return 0
        return 1

    if ctype == "one_sentence":
        if count_sentences(stripped) not in {1, 2}:
            return 0
        return 1

    if ctype == "starts_with":
        return 1

    return 1


# ---------------------------------------------------------------------------
# Keyword / stem helpers
# ---------------------------------------------------------------------------

def keyword_stem(token: str) -> str:
    t = normalize_text(token)
    for suf in [
        "mente", "ciones", "cion", "siones", "sion", "idades", "idad",
        "amientos", "amiento", "adoras", "adores", "adora", "ador",
        "ing", "edly", "ed", "ly", "es", "s",
    ]:
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[:-len(suf)]
    return t


def derive_keywords(item: Dict[str, str]) -> Set[str]:
    if item.get("semantic_keywords", "").strip():
        raw = item["semantic_keywords"]
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return {keyword_stem(p) for p in parts}

    gold = item.get("gold_answer_or_rubric", "")
    toks = normalized_tokens(gold)
    out: Set[str] = set()
    for tok in toks:
        if len(tok) <= 2:
            continue
        if tok in ENGLISH_FUNCTION_WORDS or tok in SPANISH_FUNCTION_WORDS:
            continue
        out.add(keyword_stem(tok))
    return out


# ---------------------------------------------------------------------------
# Numeric / date anchor extraction
# ---------------------------------------------------------------------------

def extract_numeric_anchors(text: str) -> Set[str]:
    t = normalize_text(text)
    hits: Set[str] = set()
    patterns = [
        r"\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{2,4})?\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"\s+\d{1,2}(?:\s*,?\s*\d{4})?\b",
        r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",
        r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b",
        r"\b\d{1,2}\s*(?:am|pm)\b",
        r"\b\d{2,}\b",
    ]
    for p in patterns:
        for m in re.finditer(p, t):
            hits.add(m.group().strip())
    return hits


# ---------------------------------------------------------------------------
# yes/no mismatch check
# ---------------------------------------------------------------------------

def sentence_initial_yes_no_mismatch(output: str, gold: str, response_lang: str) -> bool:
    out_low  = normalize_text(output).strip()
    gold_low = normalize_text(gold).strip()
    if response_lang == "es":
        out_pref  = "si" if out_low.startswith("si")  else "no" if out_low.startswith("no")  else ""
        gold_pref = "si" if gold_low.startswith("si") else "no" if gold_low.startswith("no") else ""
    elif response_lang == "en":
        out_pref  = "yes" if out_low.startswith("yes")  else "no" if out_low.startswith("no")  else ""
        gold_pref = "yes" if gold_low.startswith("yes") else "no" if gold_low.startswith("no") else ""
    else:
        return False
    return bool(out_pref and gold_pref and out_pref != gold_pref)


# ---------------------------------------------------------------------------
# LLM semantic judge (with cache)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a strict factual evaluator for a multilingual benchmark. "
    "Your only job is to decide whether a model output conveys the same core "
    "factual content as the gold answer / rubric. "
    "Ignore differences in language, phrasing, politeness, or extra context "
    "as long as the key fact(s) are present and correct. "
    "Key facts include both positive information AND explicit negations "
    "(e.g. 'not canceled', 'no fue cancelada'). "
    "If the output omits a key negation present in the gold answer, return 0. "
    "Reply with ONLY the digit 1 (correct) or 0 (incorrect). No other text."
)


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 16,
        "system": _LLM_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body["content"][0]["text"].strip()


def semantic_correct_llm(
    output: str,
    gold: str,
    api_key: str,
    model: str,
    cache: Dict[str, int],
    cache_path: str,
) -> int:
    """LLM judge with local JSON cache. Same (output, gold) pair never calls API twice."""
    key = _cache_key(output, gold)

    if key in cache:
        return cache[key]

    prompt = (
        f"Gold answer / rubric:\n{gold}\n\n"
        f"Model output:\n{output}\n\n"
        "Does the model output convey the same core factual content as the gold answer? "
        "Reply with only 0 or 1."
    )
    try:
        reply = _call_anthropic(prompt, api_key, model)
        digit = re.search(r"[01]", reply)
        result = int(digit.group()) if digit else 0
    except Exception as exc:
        print(f"[LLM judge] API call failed: {exc}", file=sys.stderr)
        result = 0

    cache[key] = result
    save_cache(cache_path, cache)
    return result


# ---------------------------------------------------------------------------
# SemanticCorrect
# ---------------------------------------------------------------------------

def semantic_correct_binary(
    output: str,
    item: Dict[str, str],
    api_key: str = "",
    llm_model: str = "claude-haiku-4-5-20251001",
    cache: Dict[str, int] = None,
    cache_path: str = "llm_judge_cache.json",
) -> int:
    if cache is None:
        cache = {}

    stripped = output.strip()
    if not stripped or stripped.startswith("[ERROR]"):
        return 0

    lowered = normalize_text(stripped)
    if "incomplete" in lowered and "content" in lowered and "question" in lowered:
        return 0

    gold          = item.get("gold_answer_or_rubric", "")
    response_lang = (item.get("response_language") or "").strip().lower()

    if sentence_initial_yes_no_mismatch(stripped, gold, response_lang):
        return 0

    gold_nums = extract_numeric_anchors(gold)
    if gold_nums:
        out_nums = extract_numeric_anchors(stripped)
        if not (gold_nums & out_nums):
            return 0

    if api_key:
        return semantic_correct_llm(stripped, gold, api_key, llm_model, cache, cache_path)

    keys = derive_keywords(item)
    if not keys:
        return 0

    out_keys = {keyword_stem(t) for t in normalized_tokens(stripped) if len(t) > 2}
    overlap  = len(keys & out_keys)

    if len(keys) <= 3:
        return 1 if overlap >= 1 else 0

    needed = max(1, int(round(len(keys) * 0.30)))
    return 1 if overlap >= needed else 0


# ---------------------------------------------------------------------------
# Joint success
# ---------------------------------------------------------------------------

def compute_joint_success(semantic: int, lang: int, constraint: int) -> int:
    return 1 if semantic == 1 and lang == 1 and constraint == 1 else 0


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def merge_and_score(
    items_by_id: Dict[str, Dict[str, str]],
    results: List[Dict[str, str]],
    api_key: str = "",
    llm_model: str = "claude-haiku-4-5-20251001",
    cache: Dict[str, int] = None,
    cache_path: str = "llm_judge_cache.json",
) -> List[Dict[str, str]]:
    if cache is None:
        cache = {}

    scored_rows = []
    for row in results:
        item_id = (row.get("item_id") or "").strip()
        item    = items_by_id.get(item_id)

        row_out = dict(row)

        if item is None:
            row_out["SemanticCorrect"]   = ""
            row_out["LangCorrect"]       = ""
            row_out["ConstraintCorrect"] = ""
            row_out["CodeSwitchRatio"]   = ""
            row_out["JointSuccess"]      = ""
            notes = (row.get("evaluator_notes") or "").strip()
            row_out["evaluator_notes"] = (notes + " | item_id not found").strip(" |")
            scored_rows.append(row_out)
            continue

        output_text = row.get("output_text", "")
        target_lang = (item.get("response_language") or "").strip().lower()

        # ConstraintCorrect first — result feeds into LangCorrect ES exemption
        constraint_ok = constraint_correct(output_text, item)

        sem = semantic_correct_binary(
            output_text, item, api_key, llm_model, cache, cache_path
        )
        lang_ok, cs_ratio = lang_correct_and_codeswitch_ratio(
            output_text, target_lang, constraint_correct_val=constraint_ok
        )
        joint = compute_joint_success(sem, lang_ok, constraint_ok)

        row_out["SemanticCorrect"]   = str(sem)
        row_out["LangCorrect"]       = str(lang_ok)
        row_out["ConstraintCorrect"] = str(constraint_ok)
        row_out["CodeSwitchRatio"]   = str(cs_ratio)
        row_out["JointSuccess"]      = str(joint)
        row_out["evaluator_notes"]   = row.get("evaluator_notes", "")

        scored_rows.append(row_out)

    return scored_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, str]]):
    ensure_parent_dir(path)
    fieldnames = [
        "model", "item_id", "output_text",
        "input_tokens", "output_tokens", "total_tokens",
        "SemanticCorrect", "LangCorrect", "ConstraintCorrect",
        "CodeSwitchRatio", "JointSuccess", "evaluator_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    items   = read_csv(args.items)
    results = read_csv(args.results)

    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)
    if not results:
        print(f"No results found in {args.results}")
        sys.exit(1)

    cache = load_cache(args.cache_file)
    cache_hits_before = len(cache)

    if args.anthropic_api_key:
        print(f"LLM semantic scoring enabled (model: {args.llm_model})")
        print(f"Cache: {args.cache_file} ({cache_hits_before} existing entries)")
    else:
        print("No ANTHROPIC_API_KEY — using keyword-overlap fallback for SemanticCorrect.")

    items_by_id = index_items(items)
    scored_rows = merge_and_score(
        items_by_id, results,
        api_key=args.anthropic_api_key,
        llm_model=args.llm_model,
        cache=cache,
        cache_path=args.cache_file,
    )
    write_csv(args.out, scored_rows)

    new_calls = len(cache) - cache_hits_before
    print(f"Wrote scored outputs to {args.out}")
    if args.anthropic_api_key:
        print(f"LLM judge: {new_calls} new API calls, {cache_hits_before} cache hits")


if __name__ == "__main__":
    main()
