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

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


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

    output_rows = []
    total_items = len(items)

    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0

    for idx, item in enumerate(items, start=1):
        item_id = item.get("item_id", f"item_{idx}")
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

        print(
            f"[{idx}/{total_items}] {item_id} done "
            f"(in={input_tokens}, out={output_tokens}, total={total_tokens})"
        )

        if args.no_dry_run and idx < total_items:
            time.sleep(args.sleep_seconds)

    write_outputs(args.out, output_rows)

    print(f"\nWrote {len(output_rows)} outputs for model '{args.model}' to {args.out}")
    print(f"Total input tokens:  {sum_input_tokens}")
    print(f"Total output tokens: {sum_output_tokens}")
    print(f"Total tokens:        {sum_total_tokens}")


if __name__ == "__main__":
    main()
