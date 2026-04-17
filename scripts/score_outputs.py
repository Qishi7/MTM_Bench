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
    # System / platform names — always allowed in any language
    "databridge", "campusconnect", "campuslink", "eventportal", "eventgrid",
    "openpath", "mediahub", "greenlab", "rivernet", "linkdesk",
    "cityarchive", "passgrid", "northlab", "meetdesk",
    # Person names
    "rivera", "chen",
    # Time suffixes: internationally understood (e.g. "4 PM" in a Chinese sentence)
    "am", "pm",
}

# ---------------------------------------------------------------------------
# Tokens that ARE English contamination when appearing in es/zh output.
# Principle 3: venue words (room/hall/building), month names, and common
# English structural words must be localized — they are NOT proper names.
# ---------------------------------------------------------------------------
EN_STRUCTURAL_TOKENS = {
    # Venue descriptors — must be localized in es/zh
    "room", "hall", "building",
    # Academic / admin English words
    "deadline", "report", "proposal", "budget", "form", "summary", "teacher",
    "professor",
    # English months — must be localized
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


# Legacy column aliases produced by older versions of this script or external tools.
_COLUMN_ALIASES: Dict[str, str] = {
    "TaskCorrect":   "SemanticCorrect",
    "CodeSwitch":    "CodeSwitchRatio",
}


def normalize_result_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Rename any legacy column names to canonical names in-place."""
    if not rows:
        return rows
    # Detect which aliases are present in this file's header
    aliases_present = {old: new for old, new in _COLUMN_ALIASES.items() if old in rows[0]}
    if not aliases_present:
        return rows
    normalized = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            new_row[aliases_present.get(k, k)] = v
        normalized.append(new_row)
    return normalized


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


def strip_markdown_formatting(text: str) -> str:
    """Remove leading markdown bold/italic markers (**, *, __, _) that models sometimes prepend."""
    return re.sub(r"^[\*_]{1,3}", "", text).strip()


def starts_with_yes_no(text: str, target_lang: str) -> bool:
    # Strip markdown formatting before checking (e.g. "**No.**" -> "No.")
    cleaned = normalize_text(strip_markdown_formatting(text.strip()))
    if target_lang == "en":
        return cleaned.startswith("yes") or cleaned.startswith("no")
    if target_lang == "es":
        return cleaned.startswith("si") or cleaned.startswith("no")  # covers sí/si
    if target_lang == "zh":
        # Accept common Chinese yes/no starters
        return text.strip().startswith(("是", "否", "不", "对", "错", "没"))
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
    """Return True if token is an English word that should be localized in es/zh output."""
    tl = normalize_text(token)
    if tl in ENGLISH_FUNCTION_WORDS or tl in EN_STRUCTURAL_TOKENS:
        return True
    return False


def has_explanatory_prefix(text: str) -> bool:
    norm = normalize_text(text.strip())
    return any(norm.startswith(p) for p in _EXPLANATORY_PREFIXES)


# ---------------------------------------------------------------------------
# LangCorrect  (Principle 3 — strict language purity)
#
# Rules applied to ALL non-English target languages:
#   ALLOWED  : proper names in PROPER_NAME_ALLOWLIST, pure digits, single
#              letters used as labels (A/B/D), "am"/"pm" as time suffixes.
#   FORBIDDEN: English function words, English month names, venue words
#              (room/hall/building), any other English structural tokens.
#              Any forbidden token → contamination count rises.
#   DECISION : if contamination_count >= 1 → LC = 0  (zero tolerance).
#              CodeSwitchRatio = contamination_tokens / total_alpha_tokens.
#
# For English target language:
#   Must contain English function words; CJK or Spanish signal words → LC = 0.
# ---------------------------------------------------------------------------

def _is_allowlisted(tok: str) -> bool:
    """Token is a proper name or time suffix that may appear in any language."""
    tl = normalize_text(tok)
    if tl in PROPER_NAME_ALLOWLIST:
        return True
    # Pure digit strings (room numbers, years, etc.) are language-neutral
    if re.fullmatch(r"\d+", tok):
        return True
    # Single letters used as building/section labels (A, B, C, D …)
    if re.fullmatch(r"[A-Za-z]", tok):
        return True
    return False


def _count_en_contamination(tokens: List[str], target_lang: str = "") -> int:
    """
    Count tokens that are forbidden English words in a non-English output.
    Allowlisted proper names, digits, and single-letter labels are excluded.
    For Spanish output, tokens that are also valid Spanish words are excluded
    (e.g. 'no', 'a', 'con', 'en', 'solo' are both English and Spanish).
    """
    count = 0
    for tok in tokens:
        if _is_allowlisted(tok):
            continue
        tl = normalize_text(tok)
        # Skip tokens that are legitimate Spanish words (even if also English)
        if target_lang == "es" and (
            tl in SPANISH_FUNCTION_WORDS
            or re.search(r"[áéíóúñ]", tok.lower())
        ):
            continue
        # English function words or structural tokens that must be localized
        if tl in ENGLISH_FUNCTION_WORDS or tl in EN_STRUCTURAL_TOKENS:
            count += 1
    return count


def lang_correct_and_codeswitch_ratio(
    text: str,
    target_lang: str,
    constraint_correct_val: int = -1,
) -> Tuple[int, float]:
    """
    Strict language purity scoring (Principle 3).

    For zh / es targets:
      - Any English function word or structural token (room, hall, building,
        month names, etc.) that is NOT allowlisted counts as contamination.
      - Even ONE contamination token → LangCorrect = 0.
      - CodeSwitchRatio = contamination_tokens / alpha_tokens.

    For en target:
      - Must contain English function words.
      - CJK characters → LC = 0.
      - Spanish signal words count toward CodeSwitchRatio.
    """
    stripped = strip_markdown_formatting(text.strip())
    if not stripped:
        return 0, 0.0

    tokens = tokenize_words(stripped)
    total_alpha = max(len(tokens), 1)

    # Pure numeric answers (e.g. "327", "410", "3:30") are language-neutral.
    # If the entire output contains no alphabetic characters (only digits,
    # colons, spaces, punctuation), accept as LC=1 regardless of target language.
    if not re.search(r"[A-Za-zÀ-ÿ\u4e00-\u9fff]", stripped):
        return 1, 0.0

    # Pure proper-name answers (e.g. "Rivera", "Professor Rivera", "DataBridge")
    # where every alphabetic token is allowlisted are language-neutral —
    # they carry no language-specific shell beyond the name itself.
    if tokens and all(_is_allowlisted(tok) for tok in tokens):
        return 1, 0.0

    # ── Chinese ──────────────────────────────────────────────────────────────
    if target_lang == "zh":
        # Must contain CJK characters
        if not contains_cjk(stripped):
            return 0, 1.0

        # Check for forbidden English tokens
        en_contam = _count_en_contamination(tokens, target_lang="zh")
        cs_ratio = round(min(1.0, en_contam / total_alpha), 4)

        # Strict: any contamination → fail
        if en_contam >= 1:
            return 0, cs_ratio

        # Also fail if a large block of Latin text appears that isn't a proper name
        # (catches cases like bilingual output where model writes both zh and en)
        latin_words = [
            tok for tok in tokens
            if re.fullmatch(r"[A-Za-z]{2,}", tok) and not _is_allowlisted(tok)
        ]
        if latin_words:
            cs_ratio = round(min(1.0, len(latin_words) / total_alpha), 4)
            return 0, cs_ratio

        return 1, cs_ratio

    # ── Spanish ──────────────────────────────────────────────────────────────
    if target_lang == "es":
        # CJK in a Spanish answer → wrong language entirely
        if contains_cjk(stripped):
            return 0, 1.0

        # Check for forbidden English tokens (excluding words valid in Spanish too)
        en_contam = _count_en_contamination(tokens, target_lang="es")
        cs_ratio = round(min(1.0, en_contam / total_alpha), 4)

        # Strict: any contamination → fail
        if en_contam >= 1:
            return 0, cs_ratio

        # Also check that the answer actually looks Spanish (has Spanish signal)
        spanish_hits = sum(
            1 for tok in tokens
            if normalize_text(tok) in SPANISH_FUNCTION_WORDS
            or re.search(r"[áéíóúñ]", tok.lower())
        )

        # Ultra-short extraction answers (≤ 3 tokens): if no contamination
        # and constraint is satisfied, accept even without explicit Spanish signal.
        # Covers answers like "327", "No.", "Sala 327".
        if total_alpha <= 3:
            if en_contam == 0 and constraint_correct_val == 1:
                return 1, cs_ratio
            return (1 if spanish_hits >= 1 else 0), cs_ratio

        # Normal-length: require at least some Spanish signal
        lang_ok = 1 if spanish_hits >= 1 or total_alpha <= 4 else 0
        return lang_ok, cs_ratio

    # ── English ──────────────────────────────────────────────────────────────
    if target_lang == "en":
        # CJK → wrong language
        if contains_cjk(stripped):
            return 0, 1.0

        english_hits = sum(
            1 for tok in tokens
            if normalize_text(tok) in ENGLISH_FUNCTION_WORDS
            or normalize_text(tok) in EN_MONTHS
        )
        spanish_like = sum(
            1 for tok in tokens
            if normalize_text(tok) in SPANISH_FUNCTION_WORDS
            or re.search(r"[áéíóúñ]", tok.lower())
        )
        cs_ratio = round(spanish_like / total_alpha, 4)

        # Short extraction answers (≤ 5 tokens) with no Spanish/CJK contamination:
        # accept as LC=1 even without explicit English function words.
        # Covers "Professor Rivera", "Room 327", "Friday at 4 PM", etc.
        if total_alpha <= 5 and spanish_like == 0:
            return 1, cs_ratio

        if total_alpha <= 4:
            lang_ok = 1 if english_hits >= 1 else 0
        else:
            lang_ok = 1 if (english_hits / total_alpha) >= 0.15 else 0

        return lang_ok, cs_ratio

    return 0, 0.0


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def infer_constraint_type(item: Dict[str, str]) -> str:
    instruction = (item.get("instruction_text") or "").lower()
    question    = (item.get("question") or "").lower()
    full = instruction + " " + question

    # yes_no_brief: English, Spanish, and Chinese variants
    if (
        "yes or no plus one brief reason" in full
        or "先写sí或no" in full
        or "先写yes或no" in full
        or "先写sí或no" in full          # normalized form
        or "sí o no y una razón breve" in full
        or "sí o no más una razón breve" in full
        or "yes or no and one brief reason" in full
    ):
        return "yes_no_brief"

    # only_output: short direct-value extraction, no explanation
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

    # one_sentence: English, Chinese, and Spanish variants
    if (
        "one sentence" in full
        or "一句话" in full
        or "only one sentence" in full
        or "una sola oración" in full
        or "una oración" in full
        or "en una sola oración" in full
        or "en una oración" in full
    ):
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
    3. Strip leading markdown bold/italic (** or *) before all checks.
    """
    # Strip markdown bold/italic that models sometimes prepend (e.g. "**No.**")
    stripped = strip_markdown_formatting(text.strip())
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
    # Strip markdown bold/italic before comparing (e.g. "**No.**" -> "no.")
    out_low  = normalize_text(strip_markdown_formatting(output.strip()))
    gold_low = normalize_text(strip_markdown_formatting(gold.strip()))
    if response_lang == "es":
        out_pref  = "si" if out_low.startswith("si")  else "no" if out_low.startswith("no")  else ""
        gold_pref = "si" if gold_low.startswith("si") else "no" if gold_low.startswith("no") else ""
    elif response_lang == "en":
        out_pref  = "yes" if out_low.startswith("yes")  else "no" if out_low.startswith("no")  else ""
        gold_pref = "yes" if gold_low.startswith("yes") else "no" if gold_low.startswith("no") else ""
    elif response_lang == "zh":
        _ZH_YES = ("是", "对", "正确")
        _ZH_NO  = ("否", "不", "错", "没")
        out_raw  = output.strip()
        gold_raw = gold.strip()
        out_pref  = "yes" if any(out_raw.startswith(w) for w in _ZH_YES) \
                    else "no" if any(out_raw.startswith(w) for w in _ZH_NO) else ""
        gold_pref = "yes" if any(gold_raw.startswith(w) for w in _ZH_YES) \
                    else "no" if any(gold_raw.startswith(w) for w in _ZH_NO) else ""
    else:
        return False
    return bool(out_pref and gold_pref and out_pref != gold_pref)


# ---------------------------------------------------------------------------
# LLM semantic judge (with cache)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a factual evaluator for a multilingual benchmark. "
    "Decide whether a model output correctly captures the core required content. "
    "Mark 1 if the output gets the main point right without contradiction. "
    "Mark 0 only if the output is factually wrong, contradicts the gold answer, "
    "or misses a critical element that changes the meaning.\n\n"

    "RULE 1 — Semantic reversal tasks: Many items involve irony or criticism disguised as praise. "
    "The gold answer captures the REAL stance. "
    "The model output must match the real stance — a positive output when gold is negative gets 0. "
    "Check: (1) yes/no polarity correct? (2) real target of judgment correct?\n\n"

    "RULE 2 — Final-state extraction tasks: Only the FINAL valid value counts. "
    "If gold is 'Friday at 4 PM' and output says 'Monday', return 0. "
    "Accept equivalent cross-language expressions "
    "('星期五下午4点' = 'Friday at 4 PM', '里维拉教授' = 'Professor Rivera'). "
    "Extra context around the correct value is fine.\n\n"

    "RULE 3 — Summary tasks: Output does NOT need to match every detail in the gold. "
    "Mark 1 if the core update is captured correctly. "
    "Mark 0 only if the output is factually wrong or directly contradicts the gold. "
    "Missing minor details (ticket platform, registration code, extra context) is acceptable. "
    "IMPORTANT: if gold says 'not canceled' but output does not mention cancellation status at all, "
    "that is acceptable — only mark 0 if output implies it WAS canceled.\n\n"

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
    task_family: str,
    api_key: str,
    model: str,
    cache: Dict[str, int],
    cache_path: str,
) -> int:
    """LLM judge with local JSON cache. Same (output, gold, task_family) never calls API twice."""
    key = _cache_key(output + "|" + task_family, gold)

    if key in cache:
        return cache[key]

    if task_family == "reading_comprehension":
        task_hint = (
            "SEMANTIC REVERSAL task. Apply RULE 1. "
            "The gold answer is the author's REAL stance, not the surface reading. "
            "Polarity must match exactly."
        )
    elif task_family == "extraction":
        task_hint = (
            "FINAL-STATE EXTRACTION task. Apply RULE 2. "
            "Only the final valid value matters. "
            "Accept cross-language equivalents. Extra context around the value is fine."
        )
    else:
        # summary_mixed_language
        task_hint = (
            "SUMMARY task. Apply RULE 3. "
            "Mark 1 if the core update is captured correctly. "
            "Missing minor details is fine. "
            "Mark 0 only if output is factually wrong or contradicts the gold. "
            "Omitting 'not canceled' is acceptable unless output implies it WAS canceled."
        )

    prompt = (
        f"Task hint: {task_hint}\n\n"
        f"Gold:\n{gold}\n\n"
        f"Output:\n{output}\n\n"
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

    stripped = strip_markdown_formatting(output.strip())
    if not stripped or stripped.startswith("[ERROR]"):
        return 0

    lowered = normalize_text(stripped)
    if "incomplete" in lowered and "content" in lowered and "question" in lowered:
        return 0

    gold          = item.get("gold_answer_or_rubric", "")
    response_lang = (item.get("response_language") or "").strip().lower()
    task_family   = (item.get("task_family") or "").strip().lower()

    if sentence_initial_yes_no_mismatch(stripped, gold, response_lang):
        return 0

    gold_nums = extract_numeric_anchors(gold)
    if gold_nums:
        out_nums = extract_numeric_anchors(stripped)
        if not (gold_nums & out_nums):
            return 0

    if api_key:
        return semantic_correct_llm(
            stripped, gold, task_family, api_key, llm_model, cache, cache_path
        )

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

        # Pass through item-level metadata for downstream analysis
        for meta_col in ("core_id", "category", "condition_id", "regime",
                         "task_family", "difficulty",
                         "instruction_language", "content_language", "response_language"):
            if meta_col not in row_out and item.get(meta_col):
                row_out[meta_col] = item[meta_col]

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
        # Identity
        "model", "item_id", "core_id", "category", "condition_id",
        # Triplet metadata
        "regime", "task_family", "difficulty",
        "instruction_language", "content_language", "response_language",
        # Model output
        "output_text", "input_tokens", "output_tokens", "total_tokens",
        # Auto scores
        "SemanticCorrect", "LangCorrect", "ConstraintCorrect",
        "CodeSwitchRatio", "JointSuccess",
        # Human annotation (leave blank for manual fill-in)
        "HumanCorrect",
        "evaluator_notes",
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
    results = normalize_result_columns(read_csv(args.results))

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