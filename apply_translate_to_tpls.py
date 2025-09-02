import re
import csv
import json
from pathlib import Path


def map_string_with_dict(mapping: dict, text: str) -> str:
    result_chars = []
    for ch in text:
        if ch not in mapping:
            raise KeyError(f"字符 '{ch}' 没有在映射表中定义！")
        result_chars.append(mapping[ch])
    return "".join(result_chars)


def load_translations(csv_file):
    """加载翻译对照表 (原文 -> 译文)，保持顺序"""
    json_file = open("glyphTable/character-mapping.json", mode="r")
    mapping_data = json.load(json_file)
    json_file.close()

    translations = []
    with open(csv_file, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                src, tgt = row[0], row[1]

                # 替换成映射后的字符
                mapping_text = map_string_with_dict(mapping_data, tgt)

                translations.append((src, mapping_text))
    return translations


def apply_translate_to_script(psv_script_path, csv_file, output_dir):
    file_path = Path(psv_script_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载翻译
    translations = load_translations(csv_file)
    trans_idx = 0  # 翻译表索引指针

    processed_count = 0

    # 定义正则表达式模式
    rtth_pattern = re.compile(r'!_RTTH\([^,]*,([^)]*)\)\)')
    zmyyyy_pattern = re.compile(r'<\w+>_ZM\w+\(([^)]*)\)')
    mtlk_pattern = re.compile(r'!_MTLK\([^,]*,\s*([^)]*)\)')
    selr_pattern = re.compile(r'<\w+>_SELR\([^;]*;/([^)]*)\)\)')

    files = sorted(file_path.glob('*.tpl'), key=lambda x: x.name.lower())
    for psv_script in files:
        if psv_script.name == "entry_200.tpl":
            continue

        with open(psv_script, "r", encoding="shift_jis", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            orig_line = line
            line = line.strip()

            replaced = False
            match_text = None

            if line.startswith('!_RTTH'):
                match = rtth_pattern.search(line)
                if match:
                    match_text = match.group(1)

            elif '_ZM' in line and '(' in line and ')' in line:
                match = zmyyyy_pattern.search(line)
                if match:
                    match_text = match.group(1)

            elif line.startswith('!_MTLK'):
                match = mtlk_pattern.search(line)
                if match:
                    match_text = match.group(1)

            elif '_SELR(' in line:
                match = selr_pattern.search(line)
                if match:
                    match_text = match.group(1)

            # 如果匹配到文本，就尝试替换
            if match_text is not None and trans_idx < len(translations):
                src, tgt = translations[trans_idx]
                if src == match_text.strip().replace("」.", "」"):
                    has_dot = match_text.strip().endswith(".")
                    # 替换文本
                    line = orig_line.replace(match_text, tgt + "." if has_dot else tgt)
                    replaced = True
                    trans_idx += 1
                    processed_count += 1
                else:
                    raise ValueError(
                        f"翻译对照不一致:\n"
                        f"tpl 文件中: {match_text}\n"
                        f"csv 文件中: {src}\n"
                        f"文件: {psv_script.name}"
                    )

            new_lines.append(line if replaced else orig_line)

        # 写入新文件
        out_file = output_path / psv_script.name
        with open(out_file, "w", encoding="shift_jis", errors="ignore") as f:
            f.writelines(new_lines)

        print(f"Processed {psv_script.name}")

    print(f"All tpl files processed. {processed_count} count.")


if __name__ == "__main__":
    apply_translate_to_script(
        "game_script/psv",
        "game_script/translate.csv",
        "game_script/psv/translated",
    )
