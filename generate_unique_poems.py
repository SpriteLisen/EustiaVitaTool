# -*- coding: utf-8 -*-
"""
Generate 19 poems, each 14 lines x 14 characters (196 chars per poem),
using 3724 unique Unicode characters (letters only, no punctuation).

The script:
1. Collects 3724 unique characters from CJK ranges (and other Unicode ranges if needed).
2. Arranges them into 19 poems of 14x14 characters.
3. Saves the poems to "poems_19_x14x14.txt".
"""

import unicodedata
from pathlib import Path

TARGET_COUNT = 19 * 14 * 14  # 3724
assert TARGET_COUNT == 3724

# Candidate ranges to pull characters from (inclusive ranges)
RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F (if needed)
    (0x30000, 0x3134F),  # Extension G (if needed)
]


def is_suitable_char(ch: str) -> bool:
    """Check if a character is suitable for inclusion (letter only)."""
    cat = unicodedata.category(ch)
    if not cat.startswith('L'):
        return False
    if ch.isspace():
        return False
    return True


if __name__ == "__main__":
    collected = []
    seen = set()

    for start, end in RANGES:
        for code in range(start, end + 1):
            try:
                ch = chr(code)
            except ValueError:
                continue
            if ch in seen:
                continue
            if is_suitable_char(ch):
                collected.append(ch)
                seen.add(ch)
                if len(collected) >= TARGET_COUNT:
                    break
        if len(collected) >= TARGET_COUNT:
            break

    if len(collected) < TARGET_COUNT:
        for code in range(0x2E80, 0x4E00):
            if len(collected) >= TARGET_COUNT:
                break
            ch = chr(code)
            if ch in seen:
                continue
            if is_suitable_char(ch):
                collected.append(ch)
                seen.add(ch)

        if len(collected) < TARGET_COUNT:
            for code in range(0xA000, 0x3134F):
                if len(collected) >= TARGET_COUNT:
                    break
                ch = chr(code)
                if ch in seen:
                    continue
                if is_suitable_char(ch):
                    collected.append(ch)
                    seen.add(ch)

    if len(collected) < TARGET_COUNT:
        raise RuntimeError(f"Could not collect enough unique characters, got {len(collected)} of {TARGET_COUNT}")

    poems = []
    idx = 0
    for poem_index in range(19):
        lines = []
        for _ in range(14):
            line_chars = collected[idx: idx + 14]
            if len(line_chars) < 14:
                raise RuntimeError("Ran out of characters while building lines")
            lines.append(''.join(line_chars))
            idx += 14
        poem_text = '\n'.join(lines)
        poems.append(poem_text)

    all_chars = ''.join(poems).replace('\n', '')
    assert len(all_chars) == TARGET_COUNT
    assert len(set(all_chars)) == TARGET_COUNT

    out_path = Path('poems_19_x14x14.txt')
    with out_path.open('w', encoding='utf-8') as f:
        for i, poem in enumerate(poems, start=1):
            f.write(poem)
            if i != len(poems):
                f.write('\n\n')

    print(f"Generated {len(poems)} poems, each 14 lines x 14 chars ({TARGET_COUNT} unique characters total).")
    print("Saved to:", str(out_path))
    print("\n--- First poem preview ---\n")
    print(poems[0])
