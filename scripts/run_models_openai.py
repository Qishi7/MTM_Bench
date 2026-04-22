import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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
            "Run benchmark items through an OpenAI GPT model (Responses API) "
            "and write outputs to CSV."
        )
    )
    parser.add_argument("--items", required=True, help="Path to input items CSV")
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model name, e.g. gpt-5.4, gpt-5.4-mini, gpt-5.3-codex",
    )
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV and skip already completed items",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually call the OpenAI API instead of writing placeholder outputs",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help=(
            "Max output tokens. For GPT-5 reasoning models, reasoning tokens "
            "count against this budget, so default is higher than for chat models."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (ignored by GPT-5 reasoning models)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for GPT-5 series models (supported values vary by model)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
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


def build_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def is_gpt5_series(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith("gpt-5") or "codex" in name or name.startswith("o1") or name.startswith("o3") or name.startswith("o4")


def call_openai(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> Dict[str, object]:
    kwargs = {
        "model": model_name,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }

    if is_gpt5_series(model_name):
        kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        kwargs["temperature"] = temperature

    response = client.responses.create(**kwargs)

    output_text = (response.output_text or "").strip()

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
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


def load_existing_outputs(out_path: str) -> Tuple[List[Dict[str, object]], Set[str]]:
    if not os.path.exists(out_path):
        return [], set()

    rows: List[Dict[str, object]] = []
    completed_ids: Set[str] = set()

    with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            output_text = (row.get("output_text") or "").strip()
            if output_text.startswith("[ERROR]"):
                continue

            rows.append(row)
            if item_id and output_text:
                completed_ids.add(item_id)

    return rows, completed_ids


def safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main():
    load_dotenv()
    args = parse_args()

    items = read_items(args.items)
    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)

    client = None
    if args.no_dry_run:
        if OpenAI is None:
            print("Error: openai package is not installed.")
            print("Install it with: pip install openai")
            sys.exit(1)

        client = build_client()

    output_rows: List[Dict[str, object]] = []
    completed_item_ids: Set[str] = set()
    if args.resume:
        output_rows, completed_item_ids = load_existing_outputs(args.out)

    total_items = len(items)

    sum_input_tokens = sum(safe_int(row.get("input_tokens")) for row in output_rows)
    sum_output_tokens = sum(safe_int(row.get("output_tokens")) for row in output_rows)
    sum_total_tokens = sum(safe_int(row.get("total_tokens")) for row in output_rows)

    pending_items = [
        item
        for item in items
        if (item.get("item_id", "") or "").strip() not in completed_item_ids
    ]
    total_pending = len(pending_items)

    if args.resume and completed_item_ids:
        print(
            f"Resuming from {args.out}: "
            f"{len(completed_item_ids)} completed, {total_pending} remaining."
        )
    else:
        print(f"Starting fresh run: {total_pending} item(s) to process.")

    if total_pending == 0:
        print(f"No pending items. Output already complete at {args.out}")
        return

    for idx, item in enumerate(pending_items, start=1):
        item_id = item.get("item_id", f"item_{idx}")
        prompt = build_prompt(item)

        try:
            if args.no_dry_run:
                result = call_openai(
                    client=client,
                    model_name=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
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
        write_outputs(args.out, output_rows)

        print(
            f"[{idx}/{total_pending}] {item_id} done "
            f"(in={input_tokens}, out={output_tokens}, total={total_tokens})"
        )

        if args.no_dry_run and idx < total_pending:
            time.sleep(args.sleep_seconds)

    print(f"\nWrote {len(output_rows)} outputs for model '{args.model}' to {args.out}")
    print(f"Total input tokens:  {sum_input_tokens}")
    print(f"Total output tokens: {sum_output_tokens}")
    print(f"Total tokens:        {sum_total_tokens}")


if __name__ == "__main__":
    main()
