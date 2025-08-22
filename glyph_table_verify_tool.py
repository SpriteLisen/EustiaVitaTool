import os


def print_glyph_table_with_tpl(folder_path):
    # 读取所有 txt 文件
    nowIndex = 0
    for filename in os.listdir(folder_path):
        nowIndex += 1
        print(f"<0109>_ZM3e912(-----{nowIndex}-----)")
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines():
                    none_line = line.replace("\n", "").replace("\r", "")
                    print(f"<0109>_ZM3e912({none_line})")


def process_txt_files(folder_path):
    all_texts = []

    # 读取所有 txt 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().replace("\n", "").replace("\r", "")
                all_texts.append(text)

    # 把所有文本合并成一个字符串
    combined_text = "".join(all_texts)

    # 检查重复字符
    seen = set()
    duplicates = set()
    for ch in combined_text:
        if ch in seen:
            duplicates.add(ch)
        else:
            seen.add(ch)

    if duplicates:
        print("发现重复字符:")
        print("".join(sorted(duplicates)))
    else:
        print("没有重复字符")

    # 去重后做统计
    # unique_chars = list(seen)
    final_text = sorted(set(combined_text))

    print("\n字符统计（去重后）:")
    print(final_text)

    print(f"\n总字数: {len(final_text)}")

    if len(final_text) == 3627:
        print("-------------字符数不满足要求!!!!")


if __name__ == "__main__":
    folder = "glyphTable/characterTable"
    process_txt_files(folder)
    print_glyph_table_with_tpl(folder)
