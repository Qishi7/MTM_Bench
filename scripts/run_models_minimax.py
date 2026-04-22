import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Set

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

MINIMAX_BASE_URL = "https://api.minimax.io/v1"

FIELDNAMES = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark items through a MiniMax model and write outputs to CSV."
    )
    parser.add_argument("--items", required=True, help="Path to input items CSV")
    parser.add_argument("--model", required=True, help="Model name, e.g. MiniMax-M2.7")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually call the MiniMax API instead of writing placeholder outputs",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for model output (includes reasoning tokens)",
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


def load_done_ids(out_path: str) -> Set[str]:
    """Return the set of item_ids already written to the output file."""
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row.get("item_id", "")
            if item_id:
                done.add(item_id)
    return done


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


def call_minimax(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"reasoning_split": True},
    )

    choice = response.choices[0]
    message = choice.message

    # With reasoning_split=True, MiniMax puts chain-of-thought in
    # `reasoning_details` and the final answer in `content`.
    output_text = (message.content or "").strip()

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

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


def open_output_writer(out_path: str, resuming: bool):
    """Open the output CSV for appending (resume) or fresh writing."""
    ensure_parent_dir(out_path)
    mode = "a" if resuming else "w"
    f = open(out_path, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not resuming:
        writer.writeheader()
    return f, writer


def main():
    args = parse_args()

    items = read_items(args.items)
    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)

    client = None
    if args.no_dry_run:
        if OpenAI is None:
            print("Error: openai package is not installed.")
            print("Install it with: uv add openai")
            sys.exit(1)

        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            print("Error: MINIMAX_API_KEY is not set.")
            print('In PowerShell, run: $env:MINIMAX_API_KEY="your_key_here"')
            sys.exit(1)

        client = OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)

    done_ids = load_done_ids(args.out)
    resuming = len(done_ids) > 0
    if resuming:
        print(f"Resuming: {len(done_ids)} item(s) already done, skipping them.")

    total_items = len(items)
    new_count = 0
    skipped_count = 0

    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0

    out_file, writer = open_output_writer(args.out, resuming)
    try:
        for idx, item in enumerate(items, start=1):
            item_id = item.get("item_id", f"item_{idx}")

            if item_id in done_ids:
                skipped_count += 1
                print(f"[{idx}/{total_items}] {item_id} skipped (already done)")
                continue

            prompt = build_prompt(item)

            try:
                if args.no_dry_run:
                    result = call_minimax(
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

            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            total_tokens = result["total_tokens"]

            if isinstance(input_tokens, int):
                sum_input_tokens += input_tokens
            if isinstance(output_tokens, int):
                sum_output_tokens += output_tokens
            if isinstance(total_tokens, int):
                sum_total_tokens += total_tokens

            row = {
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
            writer.writerow(row)
            out_file.flush()
            new_count += 1

            print(
                f"[{idx}/{total_items}] {item_id} done "
                f"(in={input_tokens}, out={output_tokens}, total={total_tokens})"
            )

            if args.no_dry_run and idx < total_items:
                time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Progress saved to {args.out} ({new_count} new item(s) written)."
        )
        sys.exit(1)
    finally:
        out_file.close()

    print(f"\nWrote {new_count} new output(s) for model '{args.model}' to {args.out}")
    if skipped_count:
        print(f"Skipped {skipped_count} already-done item(s).")
    print(f"Total input tokens (this run):  {sum_input_tokens}")
    print(f"Total output tokens (this run): {sum_output_tokens}")
    print(f"Total tokens (this run):        {sum_total_tokens}")


if __name__ == "__main__":
    main()
