import csv
import json


def map_string_with_dict(mapping: dict, text: str) -> str:
    result_chars = []
    for ch in text:
        if ch not in mapping:
            raise KeyError(f"字符 '{ch}' 没有在映射表中定义！")
        result_chars.append(mapping[ch])
    return "".join(result_chars)


if __name__ == "__main__":
    translation_path = "eboot/translate-info.csv"  # 翻译文件

    json_file = open("glyphTable/character-mapping.json", mode="r", encoding='utf-8')
    mapping_data = json.load(json_file)
    json_file.close()

    with open(translation_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        original = []
        translated = []

        for row_num, row in enumerate(reader, 1):
            if len(row) >= 2:
                original.append(row[0])
                translated.append(row[1])

        for index, text in enumerate(original):
            original_len = len(original[index].encode('CP932'))

            translated_text = translated[index]
            mapping_text = map_string_with_dict(mapping_data, translated_text)
            translated_len = len(
                mapping_text.encode('CP932')
            )

            print(f"译文: {translated_text}, 映射结果: {mapping_text}")

            if translated_len <= original_len:
                print(f"{index} 翻译文案检测通过")
            else:
                print(f"❌ {index} 检测不通过, 原始 = {original_len}, 翻译后 = {translated_len}, 翻译结果 => {translated[index]}")
