#!/usr/bin/env python
"""
Quick checker for boxed-answer equivalence.

Example:
python quick_check_boxed_equivalence.py \
  --gold "-\\frac{11}{8}" \
  --response "...\boxed{-1.375}..."
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly check boxed answer equivalence.")
    parser.add_argument("--gold", type=str, required=True, help="Gold answer string, e.g. -\\frac{11}{8}")
    parser.add_argument("--response", type=str, required=True, help="Model response containing \\boxed{...}")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["string", "math"],
        default="math",
        help="Comparison mode for gold vs boxed answer.",
    )
    return parser.parse_args()


def extract_last_boxed(text: str) -> str:
    if not text:
        return ""
    pos = 0
    last = ""
    while True:
        idx = text.find("\\boxed", pos)
        if idx < 0:
            break
        brace_start = text.find("{", idx)
        if brace_start < 0:
            pos = idx + 6
            continue

        depth = 0
        for i in range(brace_start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace_start + 1 : i].strip()
                    break
        pos = idx + 6
    return last


def normalize_answer(text: str) -> str:
    s = text.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\u2212", "-")
    s = "".join(s.split())
    return s


def main() -> None:
    args = parse_args()
    boxed = extract_last_boxed(args.response)
    if not boxed:
        print("boxed_found=False")
        print("is_correct=False")
        return

    if args.mode == "string":
        is_correct = normalize_answer(boxed) == normalize_answer(args.gold)
    else:
        from math_verify import parse, verify

        gold_parsed = parse(args.gold)
        pred_parsed = parse(boxed)
        is_correct = len(gold_parsed) > 0 and len(pred_parsed) > 0 and bool(verify(gold_parsed, pred_parsed))

    print(f"boxed_found=True")
    print(f"boxed_value={boxed}")
    print(f"is_correct={is_correct}")


if __name__ == "__main__":
    main()
