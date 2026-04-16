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


def call_openai_compatible(
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


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_outputs(out_path: str, rows: List[Dict[str, object]]):
    ensure_parent_dir(out_path)
    fieldnames = [
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
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    output_rows = []
    total_items = len(items)

    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0

    run_start = time.monotonic()
    print(
        f"Running {total_items} items on model '{args.model}' "
        f"(dry_run={not args.no_dry_run})"
    )

    for idx, item in enumerate(items, start=1):
        item_id = item.get("item_id", f"item_{idx}")
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

        output_rows.append(
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

        elapsed = time.monotonic() - run_start
        avg = elapsed / idx
        remaining = avg * (total_items - idx)
        pct = 100.0 * idx / total_items
        bar = render_progress_bar(idx, total_items)

        print(
            f"{bar} {idx:>3}/{total_items} "
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

    write_outputs(args.out, output_rows)

    total_elapsed = time.monotonic() - run_start
    print(f"\nWrote {len(output_rows)} outputs for model '{args.model}' to {args.out}")
    print(f"Total input tokens:  {sum_input_tokens}")
    print(f"Total output tokens: {sum_output_tokens}")
    print(f"Total tokens:        {sum_total_tokens}")
    print(f"Total time:          {format_duration(total_elapsed)} "
          f"({total_elapsed:.1f}s, avg {total_elapsed / max(total_items, 1):.2f}s/item)")


if __name__ == "__main__":
    main()
