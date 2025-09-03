import re
import csv
import json
from pathlib import Path


def map_string_with_dict(mapping: dict, text: str) -> str:
    """使用映射表替换字符"""
    result_chars = []
    for ch in text:
        if ch not in mapping:
            raise KeyError(f"字符 '{ch}' 没有在映射表中定义！")
        result_chars.append(mapping[ch])
    return "".join(result_chars)


def load_translations(csv_file):
    """加载翻译对照表 (原文 -> 译文)，并映射字符"""
    with open("glyphTable/character-mapping.json", mode="r", encoding="utf-8") as json_file:
        mapping_data = json.load(json_file)

    # 生成反向映射
    reverse_mapping = {v: k for k, v in mapping_data.items()}

    translations = []
    with open(csv_file, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                src, tgt = row[0], row[1]
                mapping_text = map_string_with_dict(mapping_data, tgt)
                translations.append((src, mapping_text))
    return translations, reverse_mapping


def apply_translate_to_script(psv_script_path, csv_file, output_dir):
    file_path = Path(psv_script_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载翻译
    translations, reverse_mapping = load_translations(csv_file)
    trans_idx = 0  # 翻译表索引指针

    processed_count = 0

    # 定义正则表达式模式
    rtth_pattern = re.compile(r'!_RTTH\([^,]*,([^)]*)\)\)')
    zmyyyy_pattern = re.compile(r'<\w+>_ZM\w+\(([^)]*)\)')
    mtlk_pattern = re.compile(r'!_MTLK\([^,]*,\s*([^)]*)\)')
    selr_pattern = re.compile(r'<\w+>_SELR\([^;]*;/([^)]*)\)\)')

    over_count = 0

    files = sorted(file_path.glob('*.tpl'), key=lambda x: x.name.lower())
    for psv_script in files:
        if psv_script.name == "entry_200.tpl":
            continue

        with open(psv_script, "r", encoding="shift_jis", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            orig_line = line
            line_changed = False
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
                if src in match_text:
                    # 判断是否以 "." 结尾
                    has_dot = tgt.endswith(".")
                    if has_dot:
                        tgt_body = tgt[:-1]  # 去掉最后的 "."
                    else:
                        tgt_body = tgt

                    # 按字节长度补齐, 游戏对文案的长度有及其苛刻的要求, 不能长也不能短, 否则会影响控制语句
                    encoding = "cp932"
                    orig_bytes = match_text.encode(encoding)
                    tgt_bytes = tgt_body.encode(encoding)

                    # 计算长度差
                    diff_len = len(orig_bytes) - len(tgt_bytes)

                    # 按字节补齐
                    while diff_len != 0:
                        if diff_len > 0:
                            # 补空格
                            if diff_len % 2 == 0:
                                tgt_bytes += "　".encode(encoding)  # 全角
                                diff_len -= 2
                            else:
                                tgt_bytes += " ".encode(encoding)  # 半角
                                diff_len -= 1
                        else:
                            # 不能超出原文的长度
                            print(
                                f"({len(match_text)})" + map_string_with_dict(
                                    reverse_mapping, tgt)
                            )
                            # print(
                            #     f"长度超出原文, 预期: {len(match_text)} 实际  => ({len(tgt)}) " + map_string_with_dict(reverse_mapping, tgt)
                            # )
                            over_count += 1
                            # 长度超出时截断
                            tgt_bytes = tgt_bytes[:diff_len]  # 去掉多余字节
                            diff_len = 0

                    # 补回 "."
                    if has_dot:
                        tgt_bytes += ".".encode(encoding)

                    # 解码回字符串
                    tgt_final = tgt_bytes.decode(encoding, errors="ignore")

                    # 替换文本
                    line = orig_line.replace(match_text, tgt_final)
                    line_changed = True
                    trans_idx += 1
                    processed_count += 1

            new_lines.append(line if line_changed else orig_line)

        # 写入新文件
        out_file = output_path / psv_script.name
        with open(out_file, "w", encoding="shift_jis", errors="ignore") as f:
            f.writelines(new_lines)

    print(f"All tpl files processed. {processed_count} count.")
    print(f"总超出文案: {over_count}")


if __name__ == "__main__":
    apply_translate_to_script(
        "game_script/psv",
        "game_script/translate.csv",
        "game_script/psv/translated",
    )
