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


target_script = [
    "entry_005.scr", "entry_006.scr", "entry_013.scr", "entry_020.scr", "entry_029.scr",
    "entry_032.scr", "entry_058.scr", "entry_064.scr", "entry_065.scr", "entry_091.scr",
    "entry_095.scr", "entry_111.scr", "entry_115.scr", "entry_129.scr", "entry_151.scr",
    "entry_200.scr",
]


def apply_translate_to_script(psv_script_path, csv_file, output_dir, verbose=False, allow_truncate_non_target=False):
    """
    参数:
      verbose: 如果 True，打印每行替换前后的长度与 offset 变化（便于调试）
      allow_truncate_non_target: 如果 True，非目标脚本也会在必要时裁切超出的翻译
    """
    file_path = Path(psv_script_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    translations, reverse_mapping = load_translations(csv_file)
    trans_idx = 0  # 全局翻译索引

    processed_count = 0

    rtth_pattern = re.compile(r'_RTTH\([^,]*,([^)]*)\)\)')
    zmyyyy_pattern = re.compile(r'_ZM\w+\(([^)]*)\)')
    mtlk_pattern = re.compile(r'_MTLK\([^,]*,\s*([^)]*)\)')
    selr_pattern = re.compile(r'_SELR\([^;]*;/([^)]*)\)\)')

    over_count = 0

    files = sorted(file_path.glob('*.scr'), key=lambda x: x.name.lower())
    for psv_script in files:
        if psv_script.name == "entry_199.scr":
            continue

        with open(psv_script, "r", encoding="shift_jis", errors="ignore") as f:
            lines = f.readlines()

        need_fix_count = 0
        has_more_limit = False
        has_select = False
        has_goto = False
        has_define = False

        origin_len_count = 0
        tgt_len_count = 0

        is_target = psv_script.name in target_script
        if is_target:
            print("-" * 20 + f"{psv_script.name} 开始" + "-" * 20)

        # -----------------
        # 预扫描这一文件：收集所有将要替换的段（并不修改文本）
        # -----------------
        scan_idx = trans_idx
        matches = []  # 每项: dict(line_idx, match_text, src, tgt_body, orig_bytes, tgt_bytes)
        for i, line in enumerate(lines):
            match_text = None
            if line.startswith('_RTTH'):
                m = rtth_pattern.search(line)
                if m:
                    match_text = m.group(1)
            elif '_ZM' in line and '(' in line and ')' in line:
                m = zmyyyy_pattern.search(line)
                if m:
                    match_text = m.group(1)
            elif line.startswith('_MTLK'):
                m = mtlk_pattern.search(line)
                if m:
                    match_text = m.group(1)
            elif '_SELR(' in line:
                m = selr_pattern.search(line)
                if m:
                    has_select = True
                    match_text = m.group(1)
            elif "_GOTO" in line and "SELECT" in line:
                has_goto = True
            elif "_ZZ" in line and "SELECT" in line:
                has_define = True

            if match_text is not None and scan_idx < len(translations):
                src, tgt_body = translations[scan_idx]
                if src in match_text:
                    # 判断是否以 "." 结尾
                    has_dot = tgt_body.endswith(".") and is_target

                    encoding = "cp932"
                    orig_bytes = match_text.encode(encoding)
                    tgt_bytes = tgt_body.encode(encoding)
                    matches.append({
                        "line_idx": i,
                        "has_dot": has_dot,
                        "match_text": match_text,
                        "src": src,
                        "tgt_body": tgt_body[:-1] if has_dot else tgt_body,
                        "orig_len": len(orig_bytes),
                        "tgt_len": len(tgt_bytes),
                        "scan_idx": scan_idx
                    })
                    scan_idx += 1

        # 统计预扫描总和，便于判断是否可能完全抵消
        total_orig = sum(m["orig_len"] for m in matches)
        total_tgt = sum(m["tgt_len"] for m in matches)
        if verbose:
            print(f"[SCAN] {psv_script.name} matches={len(matches)} total_orig={total_orig} total_tgt={total_tgt}")

        # 如果总翻译长度大于原文长度且不允许裁切 -> 无法完全对齐
        if not is_target and (total_tgt > total_orig) and (not allow_truncate_non_target):
            print(f"警告: {psv_script.name} 整体翻译字节 (total_tgt={total_tgt}) 大于原文 (total_orig={total_orig})，"
                  "且未允许裁切；无法完全对齐。将保留超出内容并尽量使用后面短行抵消。")

        # -----------------
        # 现在逐行真正替换（使用 matches 列表），维护 offset
        # -----------------
        offset = 0  # 表示到当前处理为止 (orig_total_processed - final_tgt_total_processed)
        match_ptr = 0

        new_lines = []
        for i, orig_line in enumerate(lines):
            line_changed = False
            # 如果当前行是匹配项
            if match_ptr < len(matches) and matches[match_ptr]["line_idx"] == i:
                m = matches[match_ptr]
                has_dot = m["has_dot"]
                match_text = m["match_text"]
                tgt_body = m["tgt_body"]
                encoding = "cp932"

                orig_bytes = match_text.encode(encoding)
                tgt_bytes = tgt_body.encode(encoding)

                diff_len = len(orig_bytes) - len(tgt_bytes)  # 正: 需要补齐；负: 翻译超出（但非target不裁切）
                # 有 dot 则 -1, 因为开头去掉了一个字符
                if has_dot and is_target:
                    diff_len -= 1

                if is_target:
                    # 目标脚本：严格对齐（全角优先补齐；超出则裁切）
                    while diff_len != 0:
                        if diff_len > 0:
                            if diff_len >= 2:
                                tgt_bytes += "　".encode(encoding)
                                diff_len -= 2
                            else:
                                tgt_bytes += " ".encode(encoding)
                                diff_len -= 1
                        else:
                            # 超出则裁切（target 必须对齐）
                            has_more_limit = True
                            print(
                                f"长度超出原文 (target), 预期: {len(match_text)} 实际 => ({len(tgt_body)}) "
                                + map_string_with_dict(reverse_mapping, tgt_body)
                            )
                            tgt_bytes = tgt_bytes[:len(orig_bytes)]
                            diff_len = 0
                else:
                    # 非目标脚本：不裁切超出，使用 offset 抵扣后再补齐
                    if diff_len > 0:
                        # 如果之前有 surplus（offset < 0），先用它抵消
                        if offset < 0:
                            consume = min(diff_len, -offset)
                        else:
                            consume = 0
                        pad_needed = diff_len - consume  # bytes to actually add
                        # 按全角(2 bytes) + 半角(1 byte) 填充
                        if pad_needed > 0:
                            full = pad_needed // 2
                            half = pad_needed % 2
                            if full > 0:
                                tgt_bytes += "　".encode(encoding) * full
                            if half > 0:
                                tgt_bytes += " ".encode(encoding) * half
                        # offset 通过下面的统一公式更新（更稳健）
                    elif diff_len < 0:
                        # 翻译超出：**不裁切**，直接保留，offset 会在下面收录这个超出
                        pass

                # 统一更新 offset（使用最终的 tgt_bytes 长度）
                new_diff = len(orig_bytes) - len(tgt_bytes)  # 正表示 orig 更长（需要补），负表示 tgt 超出
                prev_offset = offset
                offset += new_diff

                # verbose 调试
                if verbose:
                    print(f"[{psv_script.name}] line {i}: orig_len={len(orig_bytes)} "
                          f"tgt_before={m['tgt_len']} tgt_after={len(tgt_bytes)} "
                          f"new_diff={new_diff} offset: {prev_offset} -> {offset}")

                # 补回 "."
                if has_dot and is_target:
                    tgt_bytes += ".".encode(encoding)

                # 替换并统计（用 final 的 tgt_bytes）
                tgt_final = tgt_bytes.decode(encoding, errors="ignore")
                origin_len_count += len(orig_bytes)
                tgt_len_count += len(tgt_bytes)
                line = orig_line.replace(match_text, tgt_final)
                line_changed = True
                match_ptr += 1
                trans_idx += 1
                processed_count += 1
            else:
                line = orig_line

            new_lines.append(line if line_changed else orig_line)

        # 文件结束后打印 offset 与统计
        if not is_target and offset != 0:
            print(f"警告: {psv_script.name} 全局字节差 (offset) = {offset} "
                  f"(total_orig={total_orig} total_tgt={total_tgt})")
            for m in matches:
                if m["orig_len"] < m["tgt_len"]:
                    print(map_string_with_dict(reverse_mapping, m["tgt_body"]))

        if origin_len_count == tgt_len_count:
            print(f"{psv_script.name} 已对齐")
        else:
            print(f"{psv_script.name} 不完全对齐, 总差值: {origin_len_count - tgt_len_count}")

        if has_more_limit and has_goto and has_select and has_define:
            print(f"有分支、跳转、定义 => {psv_script.name}")
        elif has_more_limit and has_select and has_define:
            print(f"只有分支和定义，无跳转 => {psv_script.name}")
        elif has_more_limit and has_goto and has_define:
            print(f"只有跳转和定义，无分支 => {psv_script.name}")
        elif not has_more_limit and has_goto and has_select and has_define:
            print(f"没超限, 但有跳转、分支、定义 => {psv_script.name}")

        if is_target:
            if need_fix_count > 0:
                print(f"总计需修复: {need_fix_count}")
            print("-" * 20 + f"{psv_script.name} 结束" + "-" * 20 + "\n")

        # 写入新文件
        out_file = output_path / psv_script.name
        with open(out_file, "w", encoding="shift_jis", errors="ignore") as f:
            f.writelines(new_lines)

    print(f"All scr files processed. {processed_count} count.")
    print(f"总超出文案: {over_count}")


if __name__ == "__main__":
    # verbose=True 可以看到每条替换的长度信息，方便定位问题
    apply_translate_to_script(
        "game_script/psv",
        "game_script/translate.csv",
        "game_script/psv/translated",
        verbose=False,
        allow_truncate_non_target=False
    )
