import csv
from Constants import normal_char


def remove_normal_char(text: str):
    final = text
    for char in normal_char:
        final = final.replace(char, "")
    return final


# noinspection DuplicatedCode
def load_final_script():
    script_file_path = "game_script/translate.csv"
    eboot_translate_file_path = "eboot/translate-info.csv"

    final_translate_file_path = "glyphTable/translate-character.txt"

    original = []
    translated = []

    with open(script_file_path, "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        for row_num, row in enumerate(reader, 1):
            if len(row) >= 2:
                original.append(row[0])
                translated.append(row[1])

    with open(eboot_translate_file_path, "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        for row_num, row in enumerate(reader, 1):
            if len(row) >= 2:
                original.append(row[0])
                translated.append(row[1])

    all_original_text = "".join(original)
    all_original_text = remove_normal_char(all_original_text)
    unique_chars = sorted(set(all_original_text))

    print("原始总字符数:", len(unique_chars))
    print("原始字符列表:", unique_chars)

    print("-" * 80)

    all_translated_text = "".join(translated)
    all_translated_text = remove_normal_char(all_translated_text)
    unique_chars = sorted(set(all_translated_text))

    print("最终总字符数:", len(unique_chars))
    print("最终字符列表:", unique_chars)

    with open(final_translate_file_path, "w", encoding='utf-8') as f:
        for char in unique_chars:
            f.write(char)


if __name__ == "__main__":
    load_final_script()
