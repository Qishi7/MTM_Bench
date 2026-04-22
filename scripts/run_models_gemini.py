import argparse
import csv
import os
import sys
import time
from typing import Dict, List

try:
    from google import genai
except ImportError:
    genai = None


for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark items through a Gemini model and write outputs to CSV."
    )
    parser.add_argument("--items", required=True, help="Path to input items CSV")
    parser.add_argument("--model", required=True, help="Model name, e.g. gemini-2.5-flash")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually call the Gemini API instead of writing placeholder outputs",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=4,
        help="Sleep between API calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries on transient errors (429/500/503/UNAVAILABLE/etc.)",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=4.0,
        help="Base sleep for exponential backoff on retry (doubles each attempt)",
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

    prompt = (
        f"{instruction}\n\n"
        f"Content:\n{content}\n\n"
        f"Question:\n{question}"
    )
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


TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_KEYWORDS = (
    "unavailable",
    "overloaded",
    "deadline",
    "timeout",
    "temporarily",
    "rate limit",
    "resource_exhausted",
    "retry",
)


def is_transient_error(err: Exception) -> bool:
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if isinstance(code, int) and code in TRANSIENT_STATUS_CODES:
        return True
    msg = str(err).lower()
    if any(k in msg for k in TRANSIENT_KEYWORDS):
        return True
    return False


def _one_call(client, model_name: str, prompt: str) -> Dict[str, object]:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    output_text = getattr(response, "text", "") or ""

    usage = getattr(response, "usage_metadata", None)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0

    return {
        "output_text": output_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_gemini(
    client,
    model_name: str,
    prompt: str,
    max_retries: int = 5,
    retry_base_seconds: float = 4.0,
) -> Dict[str, object]:
    attempt = 0
    while True:
        try:
            return _one_call(client, model_name, prompt)
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


def main():
    args = parse_args()

    items = read_items(args.items)
    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)

    client = None
    if args.no_dry_run:
        if genai is None:
            print("Error: google-genai package is not installed.")
            print("Install it with: pip install -U google-genai")
            sys.exit(1)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY is not set.")
            print('In PowerShell, run: $env:GEMINI_API_KEY="your_key_here"')
            sys.exit(1)

        client = genai.Client(api_key=api_key)

    total_items = len(items)

    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0

    done_ids = load_done_ids(args.out)
    if done_ids:
        print(f"Resuming: found {len(done_ids)} already-completed items in {args.out}, skipping them.")

    out_file, out_writer = open_appender(args.out)

    processed = 0
    try:
        for idx, item in enumerate(items, start=1):
            item_id = item.get("item_id", f"item_{idx}")
            if item_id in done_ids:
                continue
            prompt = build_prompt(item)

            try:
                if args.no_dry_run:
                    result = call_gemini(
                        client=client,
                        model_name=args.model,
                        prompt=prompt,
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

            print(
                f"[{idx}/{total_items}] {item_id} done "
                f"(in={input_tokens}, out={output_tokens}, total={total_tokens})"
            )

            if args.no_dry_run and idx < total_items:
                time.sleep(args.sleep_seconds)
    finally:
        out_file.close()

    print(f"\nAppended {processed} new outputs for model '{args.model}' to {args.out}")
    print(f"(Skipped {len(done_ids)} already-done items; file now contains "
          f"{len(done_ids) + processed}/{total_items}.)")
    print(f"Total input tokens (this run):  {sum_input_tokens}")
    print(f"Total output tokens (this run): {sum_output_tokens}")
    print(f"Total tokens (this run):        {sum_total_tokens}")


if __name__ == "__main__":
    main()