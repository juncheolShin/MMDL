#!/usr/bin/env python3
"""Measure how long the rendered assistant turns are, in real tokens.

G2 holds the answer to at most 45 tokens. For the answer_only style that is
trivially met - the target is one letter. For the ekra style the target is
evidence, optionally knowledge, reasoning and then the answer, and whether that
fits is a question about the DATA, not about the prompt: an AI2D food web cites
six edges and a ChartQA lookup cites two cells.

This has to use the model's own tokenizer. A word count or a characters/4 rule
would be off by enough to matter at a 45-token limit, and the number decides
whether the evidence format needs shortening before any arm trains on it.

    python code/data/measure_target_tokens.py training_data/sft/ekra_partial/train.jsonl
    python code/data/measure_target_tokens.py <jsonl> --budget 45 --show-worst 5
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics


def percentile(sorted_values, q):
    n = len(sorted_values)
    return sorted_values[min(int(q * n), n - 1)] if n else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("--model", default="/workspace/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--budget", type=int, default=45)
    ap.add_argument("--show-worst", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in args.jsonl.open(encoding="utf-8") if l.strip()]
    by_source = collections.defaultdict(list)
    worst = []
    for r in rows:
        target = r["conversations"][1]["value"]
        n = len(tok(target, add_special_tokens=False)["input_ids"])
        by_source[r.get("source", "?")].append(n)
        by_source["ALL"].append(n)
        worst.append((n, r.get("uid", ""), target))

    head = (f'{"source":18}{"n":>7}{"mean":>8}{"median":>8}{"p90":>7}'
            f'{"p99":>7}{"max":>7}{"<=" + str(args.budget):>9}')
    print(head)
    print("-" * len(head))
    for key in sorted(by_source, key=lambda k: (k == "ALL", k)):
        v = sorted(by_source[key])
        within = 100 * sum(1 for x in v if x <= args.budget) / len(v)
        print(f"{key:18}{len(v):>7,}{statistics.mean(v):>8.1f}"
              f"{statistics.median(v):>8.0f}{percentile(v, 0.90):>7}"
              f"{percentile(v, 0.99):>7}{max(v):>7}{within:>8.1f}%")

    worst.sort(reverse=True)
    if args.show_worst:
        print(f"\nlongest {args.show_worst}:")
        for n, uid, target in worst[:args.show_worst]:
            print(f"\n  [{n} tokens] {uid}")
            for line in target.split("\n"):
                print(f"    {line[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
