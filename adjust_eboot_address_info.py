import csv


def calculate_shift_jis_length_precise(text):
    """
    使用实际的CP932编码来计算字节长度
    """
    try:
        # 尝试用Shift-JIS编码
        byte_data = text.encode('CP932')
        char_count = len(text)
        byte_length = len(byte_data)
        return char_count, byte_length
    except UnicodeEncodeError:
        # 如果编码失败，回退到基于字符范围的估算
        return calculate_shift_jis_length(text)


def calculate_shift_jis_length(text):
    """
    基于字符范围估算Shift-JIS字节长度
    """
    byte_length = 0
    char_count = len(text)

    for char in text:
        char_code = ord(char)

        # Shift-JIS编码范围判断
        if (0x3040 <= char_code <= 0x309F or  # 平假名
                0x30A0 <= char_code <= 0x30FF or  # 片假名
                0x4E00 <= char_code <= 0x9FFF or  # 常用汉字
                0xFF00 <= char_code <= 0xFFEF or  # 全角符号
                char_code >= 0x10000 or  # 其他扩展字符
                # 一些常见的全角英文字符和符号
                char in 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ' or
                char in 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ' or
                char in '０１２３４５６７８９' or
                char in '！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～'):
            byte_length += 2
        else:
            byte_length += 1

    return char_count, byte_length


def fix_csv_lengths(csv_file_path, output_csv_path):
    """
    修正CSV文件中的长度字段
    """
    print("=" * 80)
    print("CSV文件长度修正工具")
    print("=" * 80)
    print(f"输入文件: {csv_file_path}")
    print(f"输出文件: {output_csv_path}")
    print("=" * 80)

    fixed_rows = 0
    total_rows = 0

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as infile, \
                open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row_num, row in enumerate(reader, 1):
                if len(row) < 3:
                    # 保持不完整的行不变
                    writer.writerow(row)
                    continue

                total_rows += 1

                address = row[0]
                text = row[1]
                old_length = row[2]

                # 计算正确的字符数和字节长度
                char_count, byte_length = calculate_shift_jis_length_precise(text)

                # 新的长度应该是字符数（不是字节数）
                # new_length = char_count
                new_length = byte_length

                # 检查是否需要修正
                try:
                    old_length_int = int(old_length)
                    needs_fix = old_length_int != new_length
                except ValueError:
                    needs_fix = True

                if needs_fix:
                    fixed_rows += 1
                    print(f"第 {row_num} 行: 长度 {old_length} -> {new_length}")
                    print(f"  文本: {text}")
                    print(f"  字符数: {char_count}, 字节长度: {byte_length}")

                    # 创建修正后的行
                    fixed_row = [address, text, str(new_length)]
                    # 保留原始行的其他列（如果有）
                    if len(row) > 3:
                        fixed_row.extend(row[3:])

                    writer.writerow(fixed_row)
                else:
                    # 不需要修正，保持原样
                    writer.writerow(row)

        print("=" * 80)
        print(f"处理完成!")
        print(f"总行数: {total_rows}")
        print(f"修正行数: {fixed_rows}")
        print(f"输出文件: {output_csv_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file_path}")
    except Exception as e:
        print(f"处理文件时出错: {e}")


def analyze_csv_lengths(csv_file_path):
    """
    分析CSV文件中的长度问题
    """
    print("=" * 80)
    print("CSV文件长度分析")
    print("=" * 80)

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)

            issues_found = 0

            for row_num, row in enumerate(reader, 1):
                if len(row) < 3:
                    continue

                address = row[0].strip()
                text = row[1].strip()
                csv_length = row[2].strip()

                # 计算正确的字符数
                char_count, byte_length = calculate_shift_jis_length_precise(text)

                try:
                    csv_length_int = int(csv_length)
                    if csv_length_int != char_count:
                        issues_found += 1
                        print(f"第 {row_num} 行: 地址 {address}")
                        print(f"  CSV长度: {csv_length}")
                        print(f"  实际字符数: {char_count}")
                        print(f"  字节长度: {byte_length}")
                        print(f"  文本: {text}")
                        print("-" * 40)
                except ValueError:
                    issues_found += 1
                    print(f"第 {row_num} 行: 无效的长度值 '{csv_length}'")

        print("=" * 80)
        print(f"分析完成! 发现 {issues_found} 个长度问题")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file_path}")


def main():
    # 配置参数
    input_csv = "eboot/decode-info.csv"  # 原始CSV文件
    output_csv = "eboot/decode-info-fixed.csv"  # 修正后的CSV文件

    print("选择操作:")
    print("1. 分析CSV文件长度问题")
    print("2. 修正CSV文件长度")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        analyze_csv_lengths(input_csv)
    elif choice == "2":
        fix_csv_lengths(input_csv, output_csv)
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()
