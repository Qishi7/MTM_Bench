import argparse
import csv
import os
import sys
import time
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            os.environ[key] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark items through a local Llama model via Ollama."
            "(OpenAI-compatible API) and write outputs to CSV."
        )
    )
    parser.add_argument("--items", required=True, help="Path to input items CSV")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
        help="Ollama model name, e.g. llama3.1:8b, llama3.2:3b, llama3.1:70b",
    )
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("OLLAMA_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ),
        help=(
            "OpenAI-compatible API base URL. Defaults to OLLAMA_BASE_URL, "
            "then OPENAI_BASE_URL, then http://localhost:11434/v1."
        ),
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually call the API instead of writing placeholder outputs",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens for model output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between API calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries on transient errors (429/5xx/timeout/connection)",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=4.0,
        help="Base for exponential backoff (waits base*2^attempt)",
    )
    return parser.parse_args()


def read_items(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def build_prompt(item: Dict[str, str]) -> str:
    instruction = (item.get("instruction_text") or "").strip()
    content = (item.get("content_text") or "").strip()
    question = (item.get("question") or "").strip()

    prompt = f"{instruction}\n\nContent:\n{content}\n\nQuestion:\n{question}"
    return prompt.strip()


def dry_run_output(item: Dict[str, str], model_name: str) -> Dict[str, object]:
    item_id = item.get("item_id", "UNKNOWN")
    response_language = item.get("response_language", "")
    return {
        "output_text": f"[DRY RUN] Model={model_name}; item={item_id}; response_language={response_language}",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def build_client(base_url: str):
    api_key = (
        os.environ.get("OLLAMA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "ollama"
    )

    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
    }

    return OpenAI(**client_kwargs)


def normalize_output_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    if content is None:
        return ""

    return str(content).strip()


TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_KEYWORDS = (
    "unavailable",
    "overloaded",
    "deadline",
    "timeout",
    "temporarily",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "retry",
    "connection",
    "reset by peer",
)


def is_transient_error(err: Exception) -> bool:
    code = getattr(err, "status_code", None) or getattr(err, "code", None)
    if isinstance(code, int) and code in TRANSIENT_STATUS_CODES:
        return True
    msg = str(err).lower()
    if any(k in msg for k in TRANSIENT_KEYWORDS):
        return True
    cls_name = type(err).__name__.lower()
    if any(k in cls_name for k in ("ratelimit", "timeout", "connection", "apierror")):
        return True
    return False


def _one_openai_call(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    choice = completion.choices[0] if completion.choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    output_text = normalize_output_text(getattr(message, "content", ""))

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0

    return {
        "output_text": output_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_openai_compatible(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 5,
    retry_base_seconds: float = 4.0,
) -> Dict[str, object]:
    attempt = 0
    while True:
        try:
            return _one_openai_call(
                client=client,
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            if attempt >= max_retries or not is_transient_error(e):
                raise
            wait = retry_base_seconds * (2 ** attempt)
            print(
                f"  ! transient error (attempt {attempt + 1}/{max_retries}): "
                f"{type(e).__name__}: {str(e)[:120]} -> sleeping {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
            attempt += 1


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


OUTPUT_FIELDNAMES = [
    "model",
    "item_id",
    "output_text",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "TaskCorrect",
    "LangCorrect",
    "CodeSwitch",
    "JointSuccess",
    "evaluator_notes",
]


def write_outputs(out_path: str, rows: List[Dict[str, object]]):
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_done_ids(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    done = set()
    try:
        with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = (row.get("item_id") or "").strip()
                if iid:
                    done.add(iid)
    except Exception:
        pass
    return done


def open_appender(out_path: str):
    """Open the output CSV in append mode; write header if the file is new/empty."""
    ensure_parent_dir(out_path)
    new_file = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    f = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress_bar(idx: int, total: int, width: int = 24) -> str:
    frac = idx / total if total else 0.0
    filled = int(round(width * frac))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    load_dotenv()
    args = parse_args()

    items = read_items(args.items)
    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)

    if not args.model:
        print("Error: --model is required, or set OLLAMA_MODEL in .env/environment.")
        sys.exit(1)

    client = None
    if args.no_dry_run:
        if OpenAI is None:
            print("Error: openai package is not installed.")
            print("Install it with: pip install openai")
            sys.exit(1)

        client = build_client(args.base_url)

    total_items = len(items)

    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0

    done_ids = load_done_ids(args.out)
    if done_ids:
        print(f"Resuming: found {len(done_ids)} already-completed items in {args.out}, skipping them.")

    out_file, out_writer = open_appender(args.out)

    run_start = time.monotonic()
    print(
        f"Running {total_items} items on model '{args.model}' "
        f"(dry_run={not args.no_dry_run})"
    )

    processed = 0
    try:
        for idx, item in enumerate(items, start=1):
            item_id = item.get("item_id", f"item_{idx}")
            if item_id in done_ids:
                continue
            prompt = build_prompt(item)

            item_start = time.monotonic()
            try:
                if args.no_dry_run:
                    result = call_openai_compatible(
                        client=client,
                        model_name=args.model,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        retry_base_seconds=args.retry_base_seconds,
                    )
                else:
                    result = dry_run_output(item, args.model)
            except Exception as e:
                result = {
                    "output_text": f"[ERROR] {type(e).__name__}: {e}",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                }
            item_elapsed = time.monotonic() - item_start

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            total_tokens = result["total_tokens"]

            if isinstance(input_tokens, int):
                sum_input_tokens += input_tokens
            if isinstance(output_tokens, int):
                sum_output_tokens += output_tokens
            if isinstance(total_tokens, int):
                sum_total_tokens += total_tokens

            out_writer.writerow(
                {
                    "model": args.model,
                    "item_id": item_id,
                    "output_text": result["output_text"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "TaskCorrect": "",
                    "LangCorrect": "",
                    "CodeSwitch": "",
                    "JointSuccess": "",
                    "evaluator_notes": "",
                }
            )
            out_file.flush()
            try:
                os.fsync(out_file.fileno())
            except OSError:
                pass
            processed += 1

            elapsed = time.monotonic() - run_start
            avg = elapsed / processed if processed else 0.0
            remaining_items = total_items - idx
            remaining = avg * remaining_items
            pct = 100.0 * idx / total_items
            bar = render_progress_bar(idx, total_items)

            print(
                f"{bar} {idx:>4}/{total_items} "
                f"({pct:5.1f}%) "
                f"elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(remaining)} "
                f"avg={avg:5.2f}s "
                f"last={item_elapsed:5.2f}s "
                f"| {item_id} in={input_tokens} out={output_tokens}",
                flush=True,
            )

            if args.no_dry_run and idx < total_items:
                time.sleep(args.sleep_seconds)
    finally:
        out_file.close()

    total_elapsed = time.monotonic() - run_start
    print(f"\nAppended {processed} new outputs for model '{args.model}' to {args.out}")
    print(f"(Skipped {len(done_ids)} already-done items; file now contains "
          f"{len(done_ids) + processed}/{total_items}.)")
    print(f"Total input tokens (this run):  {sum_input_tokens}")
    print(f"Total output tokens (this run): {sum_output_tokens}")
    print(f"Total tokens (this run):        {sum_total_tokens}")
    print(f"Total time (this run):          {format_duration(total_elapsed)} "
          f"({total_elapsed:.1f}s, avg {total_elapsed / max(processed, 1):.2f}s/item)")


if __name__ == "__main__":
    main()
