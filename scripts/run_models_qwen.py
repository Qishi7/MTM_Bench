import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark items through a Qwen model and write outputs to CSV."
    )
    parser.add_argument("--items", required=True, help="Path to input items CSV")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. qwen3.5-plus, qwen-plus, or qwen3.5-flash",
    )
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("QWEN_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or DEFAULT_QWEN_BASE_URL
        ),
        help="Qwen OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually call the Qwen API instead of writing placeholder outputs",
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
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for more reproducible generations",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode for hybrid-thinking models",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking mode for hybrid-thinking models",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Optional token budget for thinking mode; only used with --enable-thinking",
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


def build_extra_body(
    enable_thinking: bool,
    disable_thinking: bool,
    thinking_budget: Optional[int],
) -> Optional[Dict[str, object]]:
    extra_body: Dict[str, object] = {}

    if enable_thinking:
        extra_body["enable_thinking"] = True
        if thinking_budget is not None:
            extra_body["thinking_budget"] = thinking_budget
    elif disable_thinking:
        extra_body["enable_thinking"] = False

    return extra_body or None


def extract_output_text(completion) -> str:
    if not completion.choices:
        return ""

    message = completion.choices[0].message
    content = getattr(message, "content", "") or ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts).strip()

    return str(content).strip()


def call_qwen(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    enable_thinking: bool,
    disable_thinking: bool,
    thinking_budget: Optional[int],
) -> Dict[str, object]:
    request_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if seed is not None:
        request_kwargs["seed"] = seed

    extra_body = build_extra_body(
        enable_thinking=enable_thinking,
        disable_thinking=disable_thinking,
        thinking_budget=thinking_budget,
    )
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    completion = client.chat.completions.create(**request_kwargs)

    output_text = extract_output_text(completion)

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else (input_tokens + output_tokens)

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

    if args.thinking_budget is not None and not args.enable_thinking:
        print("Error: --thinking-budget requires --enable-thinking.")
        sys.exit(1)

    items = read_items(args.items)
    if not items:
        print(f"No items found in {args.items}")
        sys.exit(1)

    client = None
    if args.no_dry_run:
        if OpenAI is None:
            print("Error: openai package is not installed.")
            print("Install it with: pip install -U openai")
            sys.exit(1)

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("Error: DASHSCOPE_API_KEY is not set.")
            print('In zsh/bash, run: export DASHSCOPE_API_KEY="your_key_here"')
            sys.exit(1)

        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
        )

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
                result = call_qwen(
                    client=client,
                    model_name=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    enable_thinking=args.enable_thinking,
                    disable_thinking=args.disable_thinking,
                    thinking_budget=args.thinking_budget,
                )
            else:
                result = dry_run_output(item, args.model)

        except Exception as e:
            print(
                f"[{idx}/{total_items}] {item_id} ERROR: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
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
