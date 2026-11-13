
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import socket
import urllib.error
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

SPANISH_FUNCTION_WORDS = {
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "no", "se", "un", "una",
    "por", "para", "con", "como", "su", "sus", "al", "lo", "es", "fue", "era", "pero",
    "si", "sí", "porque", "aunque", "más", "menos", "esto", "esta", "estas", "estos",
    "le", "les", "ya", "muy", "solo", "sólo", "solamente", "únicamente", "unicamente",
    "realmente", "autor", "pasaje", "ahora", "antes", "después", "despues", "todavía",
    "todavia", "sigue", "final", "finalmente", "reunión", "reunion", "correo",
    "electrónico", "electronico", "subir", "enviar", "plataforma", "portal", "hora",
    "viernes", "lunes", "martes", "miércoles", "miercoles", "jueves", "sábado",
    "sabado", "domingo", "sala", "salón", "salon", "aula", "edificio", "oficina",
    "biblioteca", "laboratorio",
}

SPANISH_SIGNAL_WORDS_FOR_EN = {
    "porque", "aunque", "más", "menos", "correo", "electrónico", "electronico",
    "reunión", "reunion", "viernes", "lunes", "martes", "miércoles", "miercoles",
    "jueves", "sábado", "sabado", "domingo", "plataforma", "portal", "subir",
    "enviar", "después", "despues", "todavía", "todavia", "sala", "salón", "salon",
    "aula", "edificio",
}

ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "not", "no", "that", "this", "these",
    "those", "it", "its", "as", "by", "from", "at", "than", "rather", "because",
    "what", "who", "when", "where", "while", "although", "then", "still", "finally",
    "now", "please", "through", "via", "upload", "email", "meeting", "moved",
    "deadline", "contact", "final", "answer", "response", "room", "hall", "building",
    "office", "library", "workshop", "seminar", "lecture", "lab", "session",
}

EN_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

ES_MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre",
}

ALLOWED_PROPER_NAMES = {
    "rivera", "chen", "johnston",
}

ALLOWED_PLATFORM_NAMES = {
    "databridge", "campusconnect", "campuslink", "eventportal", "eventgrid",
    "openpath", "mediahub", "greenlab", "rivernet", "linkdesk",
    "cityarchive", "passgrid", "northlab", "meetdesk",
    "canvas", "zoom", "moodle", "teams", "google", "meet",
}

EN_STRUCTURAL_TOKENS = {
    "room", "hall", "building", "deadline", "report", "proposal", "budget",
    "form", "summary", "teacher", "professor", "meeting", "submission",
    "submissions", "contact", "email", "mail", "upload", "portal", "system",
    "office", "session", "check", "review", "workshop", "seminar", "lecture",
    "lab", "online", "link", "ticket", "support", "registration", "closed",
    "open", "moved", "relocated", "still", "canceled", "cancelled",
    "friday", "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "am", "pm",
} | EN_MONTHS


ENGLISH_SHELL_TOKENS = {
    "student", "students", "center", "centre", "career", "department",
    "website", "site", "web", "lobby", "atrium", "desk", "floor",
    "classroom", "phone", "number", "version", "method", "location",
    "entrance", "campus", "offices",
}

ENGLISH_SHELL_BIGRAMS = {
    ("west", "lobby"),
    ("east", "lobby"),
    ("north", "lobby"),
    ("south", "lobby"),
    ("student", "center"),
    ("student", "centre"),
    ("career", "center"),
    ("career", "centre"),
    ("department", "website"),
    ("student", "portal"),
    ("help", "desk"),
    ("front", "desk"),
}

_EXPLANATORY_PREFIXES = (
    "the answer is",
    "the correct answer is",
    "the final answer is",
    "answer:",
    "final answer:",
    "la respuesta es",
    "respuesta:",
    "respuesta final:",
    "la respuesta final es",
    "la fecha limite final es",
    "la fecha límite final es",
    "la sala final es",
    "最终答案是",
    "答案是",
    "答案：",
    "最终截止时间是",
    "最终联系人是",
    "最终房间是",
    "最终房间号是",
    "最终需要提交的是",
    "the final deadline is",
    "the final room number is",
    "the final room is",
    "the final files are",
    "the final file is",
    "the final person is",
    "the person to contact is",
    "the deadline is",
)

MILD_MIXING_TOKENS = {"am", "pm"}

ONLY_OUTPUT_VERB_PATTERNS = [
    r"\bcontact\b", r"\bemail\b", r"\bmail\b", r"\bupload\b", r"\bsubmit\b",
    r"\bregister\b", r"\bcall\b", r"\bvisit\b", r"\bgo\b", r"\busing\b",
    r"\buse\b", r"\bthrough\b", r"\bvia\b", r"\bneed to\b", r"\bmust\b",
    r"\bshould\b", r"\bfirst\b", r"\bthen\b", r"\bafter\b",
    r"\bcontactar\b", r"\bcontacta\b", r"\benviar\b", r"\benvía\b", r"\benvia\b",
    r"\bsubir\b", r"\bsube\b", r"\bregistrar\b", r"\bregistra\b",
    r"\bmediante\b", r"\ba traves de\b", r"\ba través de\b", r"\bprimero\b", r"\bluego\b",
    r"联系", r"上传", r"提交", r"通过", r"使用", r"先", r"然后", r"之后", r"去", r"到",
]

ONLY_OUTPUT_CONNECTOR_PATTERNS = [
    r"\band\b", r"\bor\b", r"\bthen\b", r"\bafter\b", r"\bbefore\b",
    r"\by\b", r"\bo\b", r"\bluego\b", r"\bdespues\b", r"\bdespués\b",
    r"然后", r"以及", r"并且", r"或者", r"之后",
]

SHORT_ES_MARKERS = {
    "sala", "salon", "salón", "edificio", "piso", "aula", "oficina", "biblioteca",
    "laboratorio", "portal", "correo", "reunion", "reunión", "profesor", "profesora",
}

SHORT_EN_MARKERS = {
    "room", "building", "floor", "office", "library", "lab",
    "portal", "email", "meeting",
}

ES_NUMBER_WORD_VALUES = {
    "cero": 0,
    "uno": 1, "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16, "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21, "veintiun": 21, "veintiún": 21, "veintiuna": 21,
    "veintidos": 22, "veintidós": 22,
    "veintitres": 23, "veintitrés": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26, "veintiséis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
    "cien": 100,
}

ES_TENS_WORD_VALUES = {
    k: v for k, v in ES_NUMBER_WORD_VALUES.items() if v in {30, 40, 50, 60, 70, 80, 90}
}

ES_UNIT_WORD_VALUES = {
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}

ES_NUMBER_WORDS = set(ES_NUMBER_WORD_VALUES.keys())


EN_NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

EN_TENS_WORD_VALUES = {
    k: v for k, v in EN_NUMBER_WORD_VALUES.items() if v in {20, 30, 40, 50, 60, 70, 80, 90}
}

EN_UNIT_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

EN_NUMBER_WORDS = set(EN_NUMBER_WORD_VALUES.keys())

ES_TITLE_WORDS = {"profesor", "profesora"}

NAME_ALIAS_PATTERNS = {
    "rivera": [r"\brivera\b", r"里维拉", r"里韦拉"],
    "chen": [r"\bchen\b", r"陈"],
    "johnston": [r"\bjohnston\b", r"约翰斯顿"],
}

QUANTITY_HEAD_PATTERNS = {
    "seat": [r"seat", r"seats", r"asiento", r"asientos", r"座位"],
    "file": [r"file", r"files", r"archivo", r"archivos", r"文件", r"档案"],
    "report": [r"report", r"reports", r"informe", r"informes", r"报告"],
    "chair": [r"chair", r"chairs", r"silla", r"sillas", r"椅子"],
    "scissor": [r"scissor", r"scissors", r"tijera", r"tijeras", r"剪刀"],
    "seat_empty": [r"asiento", r"asientos", r"seat", r"seats", r"座位"],
}

SPANISH_CONTENT_WORDS = {
    "hay", "archivo", "archivos", "tijera", "tijeras", "entrada", "salida",
    "oeste", "este", "norte", "sur", "asiento", "asientos", "vacio",
    "vacios", "vacío", "vacíos", "plaza", "plazas", "puerta", "puertas",
    "silla", "sillas", "cupo", "cupos", "disponible", "disponibles",
}

def _looks_spanish_content_word(tok: str) -> bool:
    tl = normalize_text(tok)
    if tl in SPANISH_CONTENT_WORDS or tl in SPANISH_FUNCTION_WORDS or tl in ES_MONTHS or tl in ES_NUMBER_WORDS or tl in ES_TITLE_WORDS:
        return True
    if re.search(r"[áéíóúñ]", tok.lower()):
        return True
    common_suffixes = (
        "cion", "ciones", "sion", "siones", "dad", "dades", "mente",
        "ario", "aria", "arios", "arias", "ado", "ada", "ados", "adas",
        "ido", "ida", "idos", "idas", "ante", "antes", "ente", "entes",
        "oso", "osa", "osos", "osas", "ico", "ica", "icos", "icas",
        "ero", "era", "eros", "eras", "aje", "ajes", "eza", "ezas",
        "ivo", "iva", "ivos", "ivas", "logia", "logias", "ista", "istas",
    )
    if tl.endswith(common_suffixes):
        return True
    if len(tl) >= 5 and tl.endswith(("os", "as", "es", "o", "a", "e")) and re.fullmatch(r"[a-z]+", tl):
        return True
    return False



def _extract_canonical_names(text: str) -> Set[str]:
    hits: Set[str] = set()
    for canon, patterns in NAME_ALIAS_PATTERNS.items():
        for patt in patterns:
            if re.search(patt, text, flags=re.IGNORECASE):
                hits.add(canon)
                break
    return hits


def _token_number_values(tokens: List[str]) -> Set[str]:
    values: Set[str] = set()
    values |= _extract_spanish_number_values(tokens)
    values |= _extract_english_number_values(tokens)
    values |= _extract_chinese_number_values(tokens)
    for tok in tokens:
        ntok = normalize_text(tok)
        if re.fullmatch(r"\d+", ntok):
            values.add(str(int(ntok)))
    return values


def _canonical_head_from_token(tok: str) -> str | None:
    tl = normalize_text(tok)
    for canon, pats in QUANTITY_HEAD_PATTERNS.items():
        for patt in pats:
            if re.fullmatch(patt, tl, flags=re.IGNORECASE) or re.search(rf"\b{patt}\b", tl, flags=re.IGNORECASE):
                return canon
    return None


def _extract_quantity_phrases(text: str) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    raw = strip_markdown_formatting(text.strip())
    if not raw:
        return pairs

    tokens = tokenize_words(raw)
    norm_tokens = [normalize_text(tok) for tok in tokens]

    # English/Spanish token-based patterns: <num> <head> and optional article/classifier in between.
    for i, tok in enumerate(norm_tokens):
        num_vals: Set[str] = set()
        if re.fullmatch(r"\d+", tok):
            num_vals.add(str(int(tok)))
        if tok in ES_NUMBER_WORDS:
            num_vals.add(str(ES_NUMBER_WORD_VALUES[tok]))
        if tok in EN_NUMBER_WORDS:
            num_vals.add(str(EN_NUMBER_WORD_VALUES[tok]))

        if i + 1 < len(norm_tokens) and tok in ES_TENS_WORD_VALUES and norm_tokens[i + 1] == 'y' and i + 2 < len(norm_tokens):
            nxt = norm_tokens[i + 2]
            if nxt in ES_UNIT_WORD_VALUES:
                num_vals.add(str(ES_TENS_WORD_VALUES[tok] + ES_UNIT_WORD_VALUES[nxt]))
        if i + 1 < len(norm_tokens) and tok in EN_TENS_WORD_VALUES:
            nxt = norm_tokens[i + 1]
            if nxt in EN_UNIT_WORD_VALUES:
                num_vals.add(str(EN_TENS_WORD_VALUES[tok] + EN_UNIT_WORD_VALUES[nxt]))

        if not num_vals:
            continue

        candidates = []
        if i + 1 < len(norm_tokens):
            candidates.append(norm_tokens[i + 1])
        if i + 2 < len(norm_tokens) and norm_tokens[i + 1] in {'de', 'del', 'la', 'las', 'el', 'los', 'of', 'the', 'a', 'an', 'empty', 'vacios', 'vacías', 'vacios', 'vacantes', 'vacias'}:
            candidates.append(norm_tokens[i + 2])
        if i + 2 < len(norm_tokens) and norm_tokens[i + 1] in {'empty', 'vacio', 'vacio', 'vacios', 'vacía', 'vacias'}:
            candidates.append(norm_tokens[i + 2])

        for cand in candidates:
            head = _canonical_head_from_token(cand)
            if head:
                for nv in num_vals:
                    pairs.add((head, nv))

    # Chinese patterns: 七个座位, 5个文件, 两份报告, 七个座位空着
    zh_num_pat = r"(\d+|零|〇|一|二|两|三|四|五|六|七|八|九|十|百|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四|二十五|二十六|二十七|二十八|二十九|三十)"
    zh_head_map = {
        '座位': 'seat',
        '文件': 'file',
        '报告': 'report',
        '椅子': 'chair',
        '剪刀': 'scissor',
    }
    for head_zh, canon in zh_head_map.items():
        patt = rf"{zh_num_pat}(?:个|份|把|张|位|只)?{head_zh}"
        for m in re.finditer(patt, raw):
            num_text = m.group(1)
            vals = _token_number_values([num_text])
            for nv in vals:
                pairs.add((canon, nv))

    return pairs


MONTH_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

WEEKDAY_PATTERNS = {
    "mon": [r"\bmonday\b", r"\blunes\b", r"星期一", r"周一", r"礼拜一"],
    "tue": [r"\btuesday\b", r"\bmartes\b", r"星期二", r"周二", r"礼拜二"],
    "wed": [r"\bwednesday\b", r"\bmiércoles\b", r"\bmiercoles\b", r"星期三", r"周三", r"礼拜三"],
    "thu": [r"\bthursday\b", r"\bjueves\b", r"星期四", r"周四", r"礼拜四"],
    "fri": [r"\bfriday\b", r"\bviernes\b", r"星期五", r"周五", r"礼拜五"],
    "sat": [r"\bsaturday\b", r"\bsábado\b", r"\bsabado\b", r"星期六", r"周六", r"礼拜六"],
    "sun": [r"\bsunday\b", r"\bdomingo\b", r"星期日", r"星期天", r"周日", r"周天", r"礼拜日", r"礼拜天"],
}

AM_MARKERS = {"am", "a.m.", "a.m", "morning", "de la manana", "de la mañana", "上午", "早上", "清晨"}
PM_MARKERS = {"pm", "p.m.", "p.m", "afternoon", "evening", "de la tarde", "de la noche", "下午", "晚上", "傍晚", "中午"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Auto-score outputs with SemanticCorrect(0/1), LangCorrect(0/1), "
            "ConstraintCorrect(0/1), CodeSwitchRatio, JointSuccess(0/1)."
        )
    )
    parser.add_argument("--items", required=True, help="Path to benchmark items CSV")
    parser.add_argument("--results", required=True, help="Path to model outputs CSV")
    parser.add_argument("--out", required=True, help="Path to scored output CSV")
    parser.add_argument(
        "--anthropic-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key for LLM-based semantic scoring",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for semantic judgement",
    )
    parser.add_argument(
        "--cache-file",
        default="llm_judge_cache.json",
        help="Path to JSON cache for LLM judge results",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_COLUMN_ALIASES: Dict[str, str] = {
    "TaskCorrect": "SemanticCorrect",
    "CodeSwitch": "CodeSwitchRatio",
}


def normalize_result_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows
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

def _cache_key(output: str, gold: str, family: str) -> str:
    raw = json.dumps(
        {"output": output, "gold": gold, "family": family},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def normalize_ampm_surface(text: str) -> str:
    text = re.sub(r"\b([ap])\s*\.\s*m\s*\.\b", r"\1m", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([ap])\s*\.\s*m\b", r"\1m", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([ap])\s*m\b", r"\1m", text, flags=re.IGNORECASE)
    return text


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-zÀ-ÿ0-9]+(?:'[A-Za-zÀ-ÿ0-9]+)?", text)


def normalized_tokens(text: str) -> List[str]:
    return tokenize_words(normalize_text(text))


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    safe = normalize_ampm_surface(stripped)
    chunks = re.split(r"[!?。！？]+|(?<!\b[aApP]m)\.(?!\d)", safe)
    return len([c for c in chunks if c.strip()])


def strip_markdown_formatting(text: str) -> str:
    return re.sub(r"^[\*_]{1,3}", "", text).strip()


def starts_with_yes_no(text: str, target_lang: str) -> bool:
    cleaned = normalize_text(strip_markdown_formatting(text.strip()))
    if target_lang == "en":
        return cleaned.startswith("yes") or cleaned.startswith("no")
    if target_lang == "es":
        return cleaned.startswith("si") or cleaned.startswith("no")
    if target_lang == "zh":
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


def has_explanatory_prefix(text: str) -> bool:
    norm = normalize_text(text.strip())
    return any(norm.startswith(p) for p in _EXPLANATORY_PREFIXES)


# ---------------------------------------------------------------------------
# Allowlist helpers
# ---------------------------------------------------------------------------

def _is_pure_digit_token(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+", tok))


def _is_single_letter_label(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]", tok))


def _is_allowed_proper_name(tok: str) -> bool:
    return normalize_text(tok) in ALLOWED_PROPER_NAMES


def _is_allowed_platform_name(tok: str) -> bool:
    return normalize_text(tok) in ALLOWED_PLATFORM_NAMES


def _is_allowlisted_token(tok: str) -> bool:
    return (
        _is_pure_digit_token(tok)
        or _is_single_letter_label(tok)
        or _is_allowed_proper_name(tok)
        or _is_allowed_platform_name(tok)
    )


def _contains_forbidden_english_shell(tokens: List[str]) -> bool:
    for tok in tokens:
        if normalize_text(tok) in EN_STRUCTURAL_TOKENS:
            return True
    return False


def _count_es_english_shells(tokens: List[str]) -> int:
    norm_tokens: List[str] = []
    count = 0

    for tok in tokens:
        if contains_cjk(tok):
            continue
        if _is_allowlisted_token(tok):
            continue

        tl = normalize_text(tok)
        norm_tokens.append(tl)

        if _looks_spanish_content_word(tok):
            continue

        if tl in ENGLISH_FUNCTION_WORDS or tl in EN_STRUCTURAL_TOKENS or tl in ENGLISH_SHELL_TOKENS:
            count += 1

    for a, b in zip(norm_tokens, norm_tokens[1:]):
        if (a, b) in ENGLISH_SHELL_BIGRAMS:
            count += 1

    return count


def _count_en_contamination(tokens: List[str], target_lang: str = "") -> int:
    count = 0

    if target_lang == "es":
        return _count_es_english_shells(tokens)

    for tok in tokens:
        if contains_cjk(tok):
            continue
        if _is_allowlisted_token(tok):
            continue

        tl = normalize_text(tok)

        if target_lang == "zh":
            if tl in ENGLISH_FUNCTION_WORDS or tl in EN_STRUCTURAL_TOKENS or tl in EN_MONTHS:
                count += 1
            elif re.fullmatch(r"[a-z]{2,}", tl):
                count += 1
            continue

        if tl in ENGLISH_FUNCTION_WORDS or tl in EN_STRUCTURAL_TOKENS or tl in EN_MONTHS:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Short extraction override helpers
# ---------------------------------------------------------------------------

def count_short_units(text: str) -> int:
    if not text.strip():
        return 0
    return len(tokenize_words(text))


def count_codeswitch_units(text: str) -> int:
    if not text.strip():
        return 0
    return len(tokenize_words(text))


def is_short_output(text: str, max_units: int = 8) -> bool:
    return count_short_units(text) <= max_units


def canonical_family(task_family: str, category: str = "") -> str:
    tf = (task_family or "").strip().lower()
    cat = (category or "").strip().lower()

    if tf in {"a", "semantic_reversal", "reading_comprehension"} or cat == "a":
        return "a"
    if tf in {"b", "extraction", "closed_form", "closed_form_extraction", "final_state_extraction"} or cat == "b":
        return "b"
    if tf in {"c", "summary_mixed_language", "language_purity", "update_summary"} or cat == "c":
        return "c"
    return "c"


def short_text_lang_check(text: str, target_lang: str) -> int:
    stripped = strip_markdown_formatting(text.strip())
    if not stripped:
        return 0

    tokens = normalized_tokens(stripped)

    if target_lang == "es":
        has_es_marker = any(tok in SHORT_ES_MARKERS for tok in tokens)
        has_en_marker = any(tok in SHORT_EN_MARKERS for tok in tokens)
        has_spanish_char = bool(re.search(r"[áéíóúñ]", stripped.lower()))

        if (has_es_marker or has_spanish_char) and not has_en_marker:
            return 1
        return 0

    if target_lang == "zh":
        if not contains_cjk(stripped):
            return 0
        bad_latin = [
            tok for tok in tokens
            if re.fullmatch(r"[A-Za-z]{2,}", tok) and not _is_allowlisted_token(tok)
        ]
        return 1 if not bad_latin else 0

    if target_lang == "en":
        has_en_marker = any(tok in SHORT_EN_MARKERS for tok in tokens)
        has_es_marker = any(tok in SHORT_ES_MARKERS for tok in tokens)
        if has_en_marker and not has_es_marker:
            return 1
        return 0

    return 0


def maybe_apply_short_extraction_override(
    item: Dict[str, str],
    output_text: str,
    semantic_ok: int,
    lang_ok: int,
    cs_ratio: float,
) -> Tuple[int, str]:
    if lang_ok == 1:
        return lang_ok, ""

    if semantic_ok != 1:
        return lang_ok, ""

    if round(cs_ratio, 4) != 0.0:
        return lang_ok, ""

    task_family = canonical_family(item.get("task_family", ""), item.get("category", ""))
    if task_family != "b":
        return lang_ok, ""

    if not is_short_output(output_text, max_units=8):
        return lang_ok, ""

    target_lang = (item.get("response_language") or "").strip().lower()
    rescued = short_text_lang_check(output_text, target_lang)

    if rescued == 1:
        return 1, "short extraction override"

    return lang_ok, ""


# ---------------------------------------------------------------------------
# LangCorrect and CodeSwitchRatio
# ---------------------------------------------------------------------------

def lang_correct_and_codeswitch_ratio(
    text: str,
    target_lang: str,
    constraint_correct_val: int = -1,
) -> Tuple[int, float]:
    stripped = strip_markdown_formatting(text.strip())
    if not stripped:
        return 0, 0.0

    tokens = tokenize_words(stripped)
    total_units = max(count_codeswitch_units(stripped), 1)

    if not re.search(r"[A-Za-zÀ-ÿ\u4e00-\u9fff]", stripped):
        return 1, 0.0

    if tokens and all(_is_allowlisted_token(tok) for tok in tokens) and not _contains_forbidden_english_shell(tokens):
        return 1, 0.0

    if target_lang == "zh":
        if not contains_cjk(stripped):
            return 0, 1.0

        en_contam = _count_en_contamination(tokens, target_lang="zh")
        cs_ratio = round(min(1.0, en_contam / total_units), 4)

        if en_contam >= 1:
            return 0, cs_ratio

        latin_words = [
            tok for tok in tokens
            if re.fullmatch(r"[A-Za-z]{2,}", tok) and not _is_allowlisted_token(tok)
        ]
        if latin_words:
            cs_ratio = round(min(1.0, len(latin_words) / total_units), 4)
            return 0, cs_ratio

        return 1, cs_ratio

    if target_lang == "es":
        if contains_cjk(stripped):
            return 0, 1.0

        en_contam = _count_en_contamination(tokens, target_lang="es")
        cs_ratio = round(min(1.0, en_contam / total_units), 4)

        if en_contam >= 1:
            return 0, cs_ratio

        spanish_hits = sum(1 for tok in tokens if _looks_spanish_content_word(tok))
        plain_latin_tokens = [
            tok for tok in tokens
            if re.fullmatch(r"[A-Za-zÀ-ÿ]{2,}", tok)
            and not _is_allowlisted_token(tok)
        ]
        digit_tokens = [tok for tok in tokens if re.fullmatch(r"\d+", tok)]

        if total_units <= 3:
            if spanish_hits >= 1:
                return 1, cs_ratio
            if digit_tokens and len(plain_latin_tokens) >= 1 and en_contam == 0:
                return 1, cs_ratio
            return 0, cs_ratio

        if spanish_hits >= 1:
            return 1, cs_ratio

        if len(plain_latin_tokens) >= 2 and en_contam == 0:
            return 1, cs_ratio

        return 0, cs_ratio

    if target_lang == "en":
        if contains_cjk(stripped):
            return 0, 1.0

        english_hits = sum(
            1 for tok in tokens
            if normalize_text(tok) in ENGLISH_FUNCTION_WORDS
            or normalize_text(tok) in EN_MONTHS
        )
        spanish_like = sum(
            1 for tok in tokens
            if normalize_text(tok) in SPANISH_SIGNAL_WORDS_FOR_EN
            or normalize_text(tok) in ES_MONTHS
            or re.search(r"[áéíóúñ]", tok.lower())
        )
        cs_ratio = round(spanish_like / total_units, 4)

        if total_units <= 5 and spanish_like == 0:
            return 1, cs_ratio

        if total_units <= 4:
            lang_ok = 1 if english_hits >= 1 else 0
        else:
            lang_ok = 1 if (english_hits / max(len(tokens), 1)) >= 0.15 else 0

        return lang_ok, cs_ratio

    return 0, 0.0


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def infer_constraint_type(item: Dict[str, str]) -> str:
    explicit = (item.get("constraint_type") or "").strip().lower()
    if explicit:
        return explicit

    instruction = normalize_text(item.get("instruction_text") or "")
    question = normalize_text(item.get("question") or "")
    full = instruction + " " + question

    if (
        "yes or no plus one brief reason" in full
        or "yes or no and one brief reason" in full
        or "si o no y una razon breve" in full
        or "si o no mas una razon breve" in full
        or "si o no y una breve razon" in full
        or "回答是或否" in full
        or "先写yes或no" in full
        or "先写si或no" in full
        or "先写是或否" in full
    ):
        return "yes_no_brief"

    if (
        "only output" in full
        or "output only" in full
        or "do not add explanation" in full
        or "do not include explanation" in full
        or "only the answer" in full
        or "output only one" in full
        or "no explanation" in full
        or "no preamble" in full
        or "solo responde" in full
        or "solo escrib" in full
        or "solo da" in full
        or "solo la respuesta" in full
        or "solo la respuesta final" in full
        or "solamente escribe" in full
        or "solamente la respuesta" in full
        or "unicamente la respuesta" in full
        or "sin explicacion" in full
        or "sin ningun comentario" in full
        or "sin comentario" in full
        or "sin agregar" in full
        or "sin anadir" in full
        or "sin aadir" in full
        or "no anadas explicacion" in full
        or "no agregues explicacion" in full
        or "不要补充说明" in full
        or "不要输出流程" in full
        or "不要添加说明" in full
        or "不要解释" in full
        or "无需解释" in full
        or "只输出" in full
        or "仅输出" in full
        or "只写" in full
        or "仅写" in full
        or "直接输出" in full
        or "直接回答" in full
    ):
        return "only_output"

    if (
        "one sentence" in full
        or "only one sentence" in full
        or "in one sentence" in full
        or "一句话" in full
        or "只用一句话" in full
        or "用一句话" in full
        or "单句" in full
        or "una sola oracion" in full
        or "en una sola oracion" in full
        or "en una oracion" in full
        or "una sola frase" in full
        or "en una frase" in full
        or "una unica oracion" in full
        or "una unica frase" in full
    ):
        return "one_sentence"

    if (
        "do not use the word" in full
        or "不得使用" in full
        or "不要使用" in full
        or "no uses la palabra" in full
    ):
        return "banned_word"

    if (
        "must begin with" in full
        or "必须以" in full
        or "debe empezar con" in full
        or "debe comenzar con" in full
    ):
        return "starts_with"

    family = canonical_family(item.get("task_family", ""), item.get("category", ""))
    if family == "b":
        return "only_output"

    return "generic"


def extract_max_words(item: Dict[str, str]) -> int:
    full = normalize_text((item.get("instruction_text") or "") + " " + (item.get("question") or ""))
    for patt in [r"at most (\d+) words", r"no more than (\d+) words", r"不超过(\d+)个词"]:
        m = re.search(patt, full)
        if m:
            return int(m.group(1))
    return 0


def extract_banned_words(item: Dict[str, str]) -> List[str]:
    full = normalize_text((item.get("instruction_text") or "") + " " + (item.get("question") or ""))
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


def violates_only_output_style(stripped: str, item: Dict[str, str]) -> bool:
    family = canonical_family(item.get("task_family", ""), item.get("category", ""))

    if has_explanatory_prefix(stripped):
        return True

    if looks_like_list_or_heading(stripped):
        return True

    if count_sentences(stripped) > 1:
        return True

    if len([ln for ln in stripped.splitlines() if ln.strip()]) > 1:
        return True

    if any(ch in stripped for ch in [";", "；"]):
        return True

    if family == "b":
        if len(tokenize_words(stripped)) > 12:
            return True

        if re.search(r"[，,、]", stripped):
            return True

        low = normalize_text(stripped)
        for patt in ONLY_OUTPUT_VERB_PATTERNS + ONLY_OUTPUT_CONNECTOR_PATTERNS:
            if re.search(patt, low):
                return True
    else:
        if len(tokenize_words(stripped)) > 12:
            return True

    return False


# ---------------------------------------------------------------------------
# ConstraintCorrect
# ---------------------------------------------------------------------------

def constraint_correct(text: str, item: Dict[str, str]) -> int:
    stripped = strip_markdown_formatting(text.strip())
    if not stripped or stripped.startswith("[ERROR]"):
        return 0

    ctype = infer_constraint_type(item)

    if looks_like_list_or_heading(stripped):
        return 0

    max_words = extract_max_words(item)
    if max_words > 0 and len(tokenize_words(stripped)) > max_words + 2:
        return 0

    banned_words = extract_banned_words(item)
    lowered = normalize_text(stripped)
    if banned_words and any(bw in lowered for bw in banned_words):
        return 0

    required_prefix = extract_required_prefix(item)
    if required_prefix and not stripped.startswith(required_prefix):
        return 0

    if ctype == "yes_no_brief":
        target_lang = (item.get("response_language") or "").strip().lower()
        if not starts_with_yes_no(stripped, target_lang):
            return 0
        if count_sentences(stripped) > 2:
            return 0
        return 1

    if ctype == "only_output":
        return 0 if violates_only_output_style(stripped, item) else 1

    if ctype == "one_sentence":
        return 1 if count_sentences(stripped) == 1 else 0

    if ctype == "starts_with":
        return 1

    return 1


# ---------------------------------------------------------------------------
# Keyword helpers
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


def _extract_spanish_number_values(tokens: List[str]) -> Set[str]:
    values: Set[str] = set()
    norm_tokens = [normalize_text(tok) for tok in tokens if not contains_cjk(tok)]

    for i, tok in enumerate(norm_tokens):
        if tok in ES_NUMBER_WORD_VALUES:
            values.add(str(ES_NUMBER_WORD_VALUES[tok]))

        if i + 2 < len(norm_tokens):
            a, b, c = norm_tokens[i], norm_tokens[i + 1], norm_tokens[i + 2]
            if a in ES_TENS_WORD_VALUES and b == "y" and c in ES_UNIT_WORD_VALUES:
                values.add(str(ES_TENS_WORD_VALUES[a] + ES_UNIT_WORD_VALUES[c]))

    return values




def _extract_english_number_values(tokens: List[str]) -> Set[str]:
    values: Set[str] = set()
    norm_tokens = [normalize_text(tok) for tok in tokens if not contains_cjk(tok)]

    for i, tok in enumerate(norm_tokens):
        if tok in EN_NUMBER_WORD_VALUES:
            values.add(str(EN_NUMBER_WORD_VALUES[tok]))

        if i + 1 < len(norm_tokens):
            a, b = norm_tokens[i], norm_tokens[i + 1]
            if a in EN_TENS_WORD_VALUES and b in EN_UNIT_WORD_VALUES:
                values.add(str(EN_TENS_WORD_VALUES[a] + EN_UNIT_WORD_VALUES[b]))

    return values


_ZH_DIGIT_VALUES = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

def _parse_simple_zh_number(token: str) -> int | None:
    if not token or any(ch not in set("零〇一二两三四五六七八九十百") for ch in token):
        return None

    if token in _ZH_DIGIT_VALUES:
        return _ZH_DIGIT_VALUES[token]

    if token == "十":
        return 10

    if "百" in token:
        parts = token.split("百", 1)
        left = parts[0]
        right = parts[1]
        hundreds = _ZH_DIGIT_VALUES.get(left, 1) if left else 1
        remainder = _parse_simple_zh_number(right) if right else 0
        if remainder is None:
            return None
        return hundreds * 100 + remainder

    if "十" in token:
        parts = token.split("十", 1)
        left = parts[0]
        right = parts[1]
        tens = _ZH_DIGIT_VALUES.get(left, 1) if left else 1
        ones = _ZH_DIGIT_VALUES.get(right, 0) if right else 0
        if right and right not in _ZH_DIGIT_VALUES:
            return None
        return tens * 10 + ones

    return None


def _extract_chinese_number_values(tokens: List[str]) -> Set[str]:
    values: Set[str] = set()
    for tok in tokens:
        if contains_cjk(tok):
            val = _parse_simple_zh_number(tok)
            if val is not None:
                values.add(str(val))
                continue
            m = re.match(r"^[零〇一二两三四五六七八九十百]+", tok)
            if m:
                prefix = m.group(0)
                val = _parse_simple_zh_number(prefix)
                if val is not None:
                    values.add(str(val))
    return values
# ---------------------------------------------------------------------------
# Semantic anchors for family B
# ---------------------------------------------------------------------------

@dataclass
class SemanticAnchors:
    weekdays: Set[str] = field(default_factory=set)
    times: Set[str] = field(default_factory=set)
    dates: Set[str] = field(default_factory=set)
    numbers: Set[str] = field(default_factory=set)
    names: Set[str] = field(default_factory=set)
    quantity_phrases: Set[Tuple[str, str]] = field(default_factory=set)


def _canonical_time(hour: int, minute: int, meridiem: str = "") -> str:
    h = int(hour)
    m = int(minute)
    mer = normalize_text(meridiem).replace(" ", "")

    if mer in {"pm", "afternoon", "evening", "delatarde", "delanoche", "下午", "晚上", "傍晚"}:
        if h < 12:
            h += 12
    elif mer in {"am", "morning", "delamanana", "de lamanana", "上午", "早上", "清晨"}:
        if h == 12:
            h = 0
    elif mer in {"中午"}:
        if 1 <= h <= 11:
            h += 12

    return f"{h:02d}:{m:02d}"


def _extract_weekdays(text: str) -> Set[str]:
    t = normalize_text(text)
    hits = set()
    for canon, patterns in WEEKDAY_PATTERNS.items():
        for patt in patterns:
            if re.search(patt, t, flags=re.IGNORECASE):
                hits.add(canon)
                break
    return hits


def _extract_dates(text: str) -> Set[str]:
    t = normalize_text(text)
    hits: Set[str] = set()

    for m in re.finditer(r"\b(\d{1,4})[-/](\d{1,2})(?:[-/](\d{1,4}))?\b", t):
        a, b, c = m.group(1), m.group(2), m.group(3)
        if c:
            if len(a) == 4:
                hits.add(f"{int(a):04d}-{int(b):02d}-{int(c):02d}")
            else:
                hits.add(f"{int(a):02d}-{int(b):02d}-{int(c):04d}" if len(c) == 4 else f"{int(a):02d}-{int(b):02d}-{int(c):02d}")
        else:
            hits.add(f"{int(a):02d}-{int(b):02d}")

    month_union = "|".join(sorted(MONTH_TO_NUM.keys(), key=len, reverse=True))
    patt1 = rf"\b({month_union})\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?\b"
    patt2 = rf"\b(\d{{1,2}})\s+({month_union})(?:\s*,?\s*(\d{{4}}))?\b"

    for m in re.finditer(patt1, t):
        month = MONTH_TO_NUM[m.group(1)]
        day = int(m.group(2))
        year = m.group(3)
        hits.add(f"{int(year):04d}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}")

    for m in re.finditer(patt2, t):
        day = int(m.group(1))
        month = MONTH_TO_NUM[m.group(2)]
        year = m.group(3)
        hits.add(f"{int(year):04d}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}")

    for m in re.finditer(r"(\d{1,2})月(\d{1,2})日", text):
        month = int(m.group(1))
        day = int(m.group(2))
        hits.add(f"{month:02d}-{day:02d}")

    return hits


def _extract_times(text: str) -> Set[str]:
    raw = text
    t = normalize_text(normalize_ampm_surface(text))
    hits: Set[str] = set()

    # Chinese time expressions
    for m in re.finditer(r"(上午|下午|中午|晚上|早上)?\s*(\d{1,2})(?::(\d{2}))?\s*点(?:(\d{1,2})分|半)?", raw):
        marker = m.group(1) or ""
        hour = int(m.group(2))
        minute = 0
        if m.group(3):
            minute = int(m.group(3))
        elif m.group(4):
            minute = int(re.sub(r"分", "", m.group(4)))
        elif "半" in m.group(0):
            minute = 30
        hits.add(_canonical_time(hour, minute, marker))

    # Spanish natural phrases
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(?:de la|de)\s*(manana|mañana|tarde|noche)\b", t):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        part = m.group(3)
        marker = "de la manana" if part in {"manana", "mañana"} else ("de la tarde" if part == "tarde" else "de la noche")
        hits.add(_canonical_time(hour, minute, marker))

    # a las 4 / at 4 + optional am/pm
    for m in re.finditer(r"\b(?:a las|at)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        marker = m.group(3) or ""
        hits.add(_canonical_time(hour, minute, marker))

    # bare 4 pm / 4:00 pm
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        marker = m.group(3)
        hits.add(_canonical_time(hour, minute, marker))

    # 24-hour time
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", t):
        hour = int(m.group(1))
        minute = int(m.group(2))
        hits.add(_canonical_time(hour, minute, ""))

    return hits


def _text_without_time_date_spans(text: str) -> str:
    safe = normalize_text(normalize_ampm_surface(text))
    safe = re.sub(r"(上午|下午|中午|晚上|早上)?\s*\d{1,2}(?::\d{2})?\s*点(?:(\d{1,2})分|半)?", " ", safe)
    safe = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", " ", safe)
    safe = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:de la|de)\s*(manana|mañana|tarde|noche)\b", " ", safe)
    safe = re.sub(r"\b(?:a las|at)\s+\d{1,2}(?::\d{2})?\s*(am|pm)?\b", " ", safe)
    safe = re.sub(r"\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b", " ", safe)
    month_union = "|".join(sorted(MONTH_TO_NUM.keys(), key=len, reverse=True))
    safe = re.sub(rf"\b({month_union})\s+\d{{1,2}}(?:\s*,?\s*\d{{4}})?\b", " ", safe)
    safe = re.sub(rf"\b\d{{1,2}}\s+({month_union})(?:\s*,?\s*\d{{4}})?\b", " ", safe)
    safe = re.sub(r"\d{1,2}月\d{1,2}日", " ", safe)
    return safe


def extract_numeric_anchors(text: str) -> Set[str]:
    anchors = extract_semantic_anchors(text)
    out = set()
    out |= anchors.times
    out |= anchors.dates
    out |= anchors.numbers
    return out


def extract_semantic_anchors(text: str) -> SemanticAnchors:
    stripped = strip_markdown_formatting(text.strip())
    anchors = SemanticAnchors()
    if not stripped:
        return anchors

    anchors.weekdays = _extract_weekdays(stripped)
    anchors.times = _extract_times(stripped)
    anchors.dates = _extract_dates(stripped)
    anchors.names = _extract_canonical_names(stripped)
    anchors.quantity_phrases = _extract_quantity_phrases(stripped)

    reduced = _text_without_time_date_spans(stripped)
    reduced_tokens = tokenize_words(reduced)
    anchors.numbers |= _extract_spanish_number_values(reduced_tokens)
    anchors.numbers |= _extract_english_number_values(reduced_tokens)
    anchors.numbers |= _extract_chinese_number_values(reduced_tokens)

    for tok in reduced_tokens:
        ntok = normalize_text(tok)
        if contains_cjk(tok):
            continue
        if _is_allowed_proper_name(tok):
            anchors.names.add(ntok)
            continue
        if re.fullmatch(r"\d+", ntok):
            anchors.numbers.add(str(int(ntok)))

    return anchors


def _safe_sentence_split(text: str) -> List[str]:
    safe = normalize_ampm_surface(text)
    safe = re.sub(r"([。！？!?])", r"\1\n", safe)
    safe = re.sub(r"(?<!\b[aApP]m)\.(?!\d)", ".\n", safe)
    return [p.strip() for p in safe.splitlines() if p.strip()]


def extract_final_answer_span(text: str) -> str:
    stripped = strip_markdown_formatting(text.strip())
    if not stripped:
        return ""

    safe = normalize_ampm_surface(stripped)
    lines = [ln.strip() for ln in safe.splitlines() if ln.strip()]
    tail = lines[-1] if lines else safe

    sentence_pieces = _safe_sentence_split(tail)
    tail = sentence_pieces[-1] if sentence_pieces else tail

    subpieces = re.split(r"\bbut\b|\bhowever\b|\bpero\b|而是|但是|不过", tail, flags=re.IGNORECASE)
    subpieces = [p.strip() for p in subpieces if p.strip()]
    tail = subpieces[-1] if subpieces else tail

    return tail.strip()


def significant_token_set(text: str) -> Set[str]:
    toks = normalized_tokens(text)
    out = set()
    for tok in toks:
        if len(tok) <= 1:
            continue
        if tok in ENGLISH_FUNCTION_WORDS or tok in SPANISH_FUNCTION_WORDS:
            continue
        out.add(keyword_stem(tok))
    return out


def family_b_semantic_correct(output: str, item: Dict[str, str]) -> int:
    stripped = strip_markdown_formatting(output.strip())
    if not stripped or stripped.startswith("[ERROR]"):
        return 0

    gold = item.get("gold_answer_or_rubric", "")
    final_span = extract_final_answer_span(stripped)
    if not final_span:
        return 0

    gold_norm = normalize_text(normalize_ampm_surface(gold)).strip()
    final_norm = normalize_text(normalize_ampm_surface(final_span)).strip()

    gold_a = extract_semantic_anchors(gold)
    out_a = extract_semantic_anchors(final_span)

    if gold_a.weekdays and out_a.weekdays and not (gold_a.weekdays & out_a.weekdays):
        return 0
    if gold_a.times and out_a.times and not (gold_a.times & out_a.times):
        return 0
    if gold_a.dates and out_a.dates and not (gold_a.dates & out_a.dates):
        return 0
    if gold_a.names and out_a.names and not (gold_a.names & out_a.names):
        return 0
    if gold_a.quantity_phrases and out_a.quantity_phrases and not (gold_a.quantity_phrases & out_a.quantity_phrases):
        # if both extracted quantity phrases but disagree, fail early
        return 0

    # For standalone numbers, require overlap when no stronger anchor already carries the value.
    if gold_a.numbers and not (gold_a.times or gold_a.dates or gold_a.quantity_phrases):
        if out_a.numbers and not (gold_a.numbers & out_a.numbers):
            return 0

    gold_keys = significant_token_set(gold)
    final_keys = significant_token_set(final_span)

    if gold_norm == final_norm:
        return 1
    if gold_norm and (gold_norm in final_norm or final_norm in gold_norm):
        return 1

    # Strong anchor success.
    if gold_a.weekdays and (gold_a.weekdays & out_a.weekdays):
        if (not gold_a.times or gold_a.times & out_a.times) and (not gold_a.dates or gold_a.dates & out_a.dates):
            return 1
    if gold_a.times and (gold_a.times & out_a.times):
        if (not gold_a.weekdays or gold_a.weekdays & out_a.weekdays):
            return 1
    if gold_a.dates and (gold_a.dates & out_a.dates):
        return 1
    if gold_a.names and (gold_a.names & out_a.names):
        return 1
    if gold_a.quantity_phrases and (gold_a.quantity_phrases & out_a.quantity_phrases):
        return 1
    if gold_a.numbers and (gold_a.numbers & out_a.numbers):
        return 1

    if gold_keys:
        if len(gold_keys) <= 2:
            return 1 if (gold_keys & final_keys) else 0
        needed = max(1, int(round(len(gold_keys) * 0.50)))
        return 1 if len(gold_keys & final_keys) >= needed else 0

    return 0


# ---------------------------------------------------------------------------
# yes/no mismatch check
# ---------------------------------------------------------------------------

def sentence_initial_yes_no_mismatch(output: str, gold: str, response_lang: str) -> bool:
    out_low = normalize_text(strip_markdown_formatting(output.strip()))
    gold_low = normalize_text(strip_markdown_formatting(gold.strip()))

    if response_lang == "es":
        out_pref = "si" if out_low.startswith("si") else "no" if out_low.startswith("no") else ""
        gold_pref = "si" if gold_low.startswith("si") else "no" if gold_low.startswith("no") else ""
    elif response_lang == "en":
        out_pref = "yes" if out_low.startswith("yes") else "no" if out_low.startswith("no") else ""
        gold_pref = "yes" if gold_low.startswith("yes") else "no" if gold_low.startswith("no") else ""
    elif response_lang == "zh":
        zh_yes = ("是", "对", "正确")
        zh_no = ("否", "不", "错", "没")
        out_raw = output.strip()
        gold_raw = gold.strip()
        out_pref = "yes" if any(out_raw.startswith(w) for w in zh_yes) else "no" if any(out_raw.startswith(w) for w in zh_no) else ""
        gold_pref = "yes" if any(gold_raw.startswith(w) for w in zh_yes) else "no" if any(gold_raw.startswith(w) for w in zh_no) else ""
    else:
        return False

    return bool(out_pref and gold_pref and out_pref != gold_pref)


# ---------------------------------------------------------------------------
# LLM semantic judge
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a factual evaluator for a multilingual benchmark. "
    "Judge ONLY semantic correctness. Ignore language purity, code-switching, "
    "and formatting constraints.\n\n"
    "Family A: semantic reversal or irony. The gold answer captures the author's "
    "real stance, not the surface wording. Do not reward literal praise if the "
    "passage is actually critical. The output is correct only if polarity is "
    "correct and the target of criticism is correct.\n\n"
    "Family C: update summary. Mark 1 if the core update is captured accurately "
    "and there is no contradiction. Minor paraphrase, mixed-language wording, "
    "or non-target-language shells do NOT make it semantically wrong. Only score "
    "semantic content.\n\n"
    "Reply with ONLY the digit 1 or 0."
)


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 16,
            "system": _LLM_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    last_exc = None
    for attempt in range(3):
        try:
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            return body["content"][0]["text"].strip()

        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(3)
            else:
                raise last_exc


def semantic_correct_llm(
    output: str,
    gold: str,
    task_family: str,
    category: str,
    api_key: str,
    model: str,
    cache: Dict[str, int],
    cache_path: str,
) -> int:
    family = canonical_family(task_family, category)
    key = _cache_key(output, gold, family)

    if key in cache:
        return cache[key]

    if family == "a":
        task_hint = (
            "Family A: semantic reversal or irony. Decide whether the output matches "
            "the writer's REAL stance. Do not reward literal praise if the passage "
            "is actually critical. The output is correct only if polarity is correct "
            "and the target of criticism is correct."
        )
    else:
        task_hint = (
            "Family C: summary of update. Judge only whether the main update is accurate "
            "and non-contradictory. Ignore any language mixing or formatting issues."
        )

    # The semantic judge receives only the task hint, gold answer/rubric, and
    # model output. Model identity is intentionally not included in the prompt.
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

    gold = item.get("gold_answer_or_rubric", "")
    response_lang = (item.get("response_language") or "").strip().lower()
    task_family = (item.get("task_family") or "").strip().lower()
    category = (item.get("category") or "").strip().lower()
    family = canonical_family(task_family, category)

    if sentence_initial_yes_no_mismatch(stripped, gold, response_lang):
        return 0

    if family == "b":
        return family_b_semantic_correct(stripped, item)

    gold_nums = extract_numeric_anchors(gold)
    if gold_nums:
        out_nums = extract_numeric_anchors(stripped)
        if not (gold_nums & out_nums):
            return 0

    if api_key and family in {"a", "c"}:
        return semantic_correct_llm(
            stripped, gold, task_family, category, api_key, llm_model, cache, cache_path
        )

    keys = derive_keywords(item)
    if not keys:
        return 0

    out_keys = {keyword_stem(t) for t in normalized_tokens(stripped) if len(t) > 1}
    overlap = len(keys & out_keys)

    if len(keys) <= 3:
        return 1 if overlap >= 1 else 0

    needed = max(1, int(round(len(keys) * 0.30)))
    return 1 if overlap >= needed else 0


# ---------------------------------------------------------------------------
# JointSuccess
# ---------------------------------------------------------------------------

def compute_joint_success(semantic: int, lang: int, constraint: int) -> int:
    return 1 if semantic == 1 and lang == 1 and constraint == 1 else 0


# ---------------------------------------------------------------------------
# Review flag
# ---------------------------------------------------------------------------

def build_review_flag(
    item: Dict[str, str],
    output_text: str,
    semantic: int,
    lang: int,
    constraint: int,
    cs_ratio: float,
) -> str:
    flags = []
    category = (item.get("category") or "").strip().lower()

    if semantic == 1 and lang == 0:
        flags.append("cc_lw")

    if semantic == 0 and lang == 1:
        flags.append("lc_cw")

    if semantic == 1 and lang == 1 and constraint == 0:
        flags.append("cv")

    if 0 < cs_ratio <= 0.20:
        flags.append("light_codeswitch")

    if category == "a" and constraint == 1 and semantic == 0:
        flags.append("a_family_review")

    if category == "b" and constraint == 1 and semantic == 0:
        flags.append("b_family_review")

    if category == "c" and (semantic == 0 or lang == 0 or constraint == 0):
        if any(tok in normalize_text(output_text) for tok in ALLOWED_PLATFORM_NAMES):
            flags.append("c_platform_name")

    return "|".join(flags)


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
        item = items_by_id.get(item_id)

        row_out = dict(row)

        if item is None:
            row_out["SemanticCorrect"] = ""
            row_out["LangCorrect"] = ""
            row_out["ConstraintCorrect"] = ""
            row_out["CodeSwitchRatio"] = ""
            row_out["JointSuccess"] = ""
            row_out["review_flag"] = "item_not_found"
            notes = (row.get("evaluator_notes") or "").strip()
            row_out["evaluator_notes"] = (notes + " | item_id not found").strip(" |")
            scored_rows.append(row_out)
            continue

        output_text = row.get("output_text", "")
        target_lang = (item.get("response_language") or "").strip().lower()

        for meta_col in (
            "core_id", "category", "condition_id", "regime", "task_family",
            "difficulty", "instruction_language", "content_language",
            "response_language", "constraint_type"
        ):
            if meta_col not in row_out and item.get(meta_col):
                row_out[meta_col] = item[meta_col]

        constraint_ok = constraint_correct(output_text, item)

        sem = semantic_correct_binary(
            output_text, item, api_key, llm_model, cache, cache_path
        )
        lang_ok, cs_ratio = lang_correct_and_codeswitch_ratio(
            output_text, target_lang, constraint_correct_val=constraint_ok
        )

        lang_ok, override_note = maybe_apply_short_extraction_override(
            item=item,
            output_text=output_text,
            semantic_ok=sem,
            lang_ok=lang_ok,
            cs_ratio=cs_ratio,
        )

        joint = compute_joint_success(sem, lang_ok, constraint_ok)
        review_flag = build_review_flag(item, output_text, sem, lang_ok, constraint_ok, cs_ratio)

        if override_note:
            review_flag = (review_flag + "|short_override").strip("|")

        old_notes = (row.get("evaluator_notes") or "").strip()
        if override_note:
            row_out["evaluator_notes"] = (old_notes + " | " + override_note).strip(" |")
        else:
            row_out["evaluator_notes"] = old_notes

        row_out["SemanticCorrect"] = str(sem)
        row_out["LangCorrect"] = str(lang_ok)
        row_out["ConstraintCorrect"] = str(constraint_ok)
        row_out["CodeSwitchRatio"] = str(cs_ratio)
        row_out["JointSuccess"] = str(joint)
        row_out["review_flag"] = review_flag

        scored_rows.append(row_out)

    return scored_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: List[Dict[str, str]]):
    ensure_parent_dir(path)
    fieldnames = [
        "model", "item_id", "core_id", "category", "condition_id",
        "regime", "task_family", "difficulty",
        "instruction_language", "content_language", "response_language",
        "constraint_type",
        "output_text", "input_tokens", "output_tokens", "total_tokens",
        "SemanticCorrect", "LangCorrect", "ConstraintCorrect",
        "CodeSwitchRatio", "JointSuccess",
        "review_flag",
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

    items = read_csv(args.items)
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
        print(f"LLM semantic scoring enabled for family A/C only (model: {args.llm_model})")
        print(f"Cache: {args.cache_file} ({cache_hits_before} existing entries)")
    else:
        print("No ANTHROPIC_API_KEY — using rule-based fallback for all families.")

    items_by_id = index_items(items)
    scored_rows = merge_and_score(
        items_by_id,
        results,
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
