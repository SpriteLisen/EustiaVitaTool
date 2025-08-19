from pathlib import Path


def load_psv_script():
    file_path = Path("game_script/extract_text/psv_version.txt")

    lines = []
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            lines.append(line)

    all_text = "".join(lines)

    unique_chars = sorted(set(all_text))

    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)


if __name__ == "__main__":
    load_psv_script()
