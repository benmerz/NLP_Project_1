#!/usr/bin/env python3
"""
Randomly sample 1/4 of records from a JSONL file and write to a new file.
"""

import json
import random
import argparse
from pathlib import Path


def sample_jsonl(input_path: str, output_path: str, keep_fraction: float = 0.25, seed: int = 42):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    random.seed(seed)
    sample_size = round(len(lines) * keep_fraction)
    sampled = random.sample(lines, sample_size)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(sampled)

    print(f"Input records:  {len(lines)}")
    print(f"Output records: {len(sampled)}")
    print(f"Saved to:       {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample 1/4 of records from a JSONL file")
    parser.add_argument('input', type=str, help='Input JSONL file')
    parser.add_argument('output', type=str, help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        raise SystemExit(1)

    sample_jsonl(args.input, args.output, seed=args.seed)
