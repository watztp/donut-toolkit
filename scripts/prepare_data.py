#!/usr/bin/env python3
"""Split receipt JSONL data and optionally inspect target token lengths."""

from __future__ import annotations

import argparse
import json

from donut_training.io_utils import analyze_text_lengths, split_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Create train/validation/test JSONL files")
    split_parser.add_argument("--source", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    split_parser.add_argument("--seed", type=int, default=42)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze target token lengths")
    analyze_parser.add_argument("--source", required=True)
    analyze_parser.add_argument("--tokenizer", required=True)
    analyze_parser.add_argument("--text-field", default="gt")
    analyze_parser.add_argument("--max-tokens", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        paths = split_jsonl(args.source, args.output_dir, tuple(args.ratios), args.seed)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return
    # Keep the split command dependency-light; Transformers is needed only here.
    from donut_training.modeling import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer, args.max_tokens)
    result = analyze_text_lengths(args.source, tokenizer, args.text_field, args.max_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
