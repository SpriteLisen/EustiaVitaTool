import csv
from elftools.elf.elffile import ELFFile


def text_to_shift_jis_hex(text):
    """
    将文本转换为Shift-JIS编码的十六进制字符串
    """
    try:
        # 将文本编码为Shift-JIS字节
        byte_data = text.encode('shift_jis')
        # 转换为十六进制字符串（空格分隔，大写）
        hex_data = byte_data.hex(' ').upper()
        return hex_data, byte_data, None
    except UnicodeEncodeError as e:
        return None, None, f"Shift-JIS编码失败: {e}"


def va_to_file_offset(elf_file_path, virtual_address):
    """
    将虚拟地址转换为ELF文件中的偏移量
    """
    with open(elf_file_path, 'rb') as f:
        elf_file = ELFFile(f)

        for segment in elf_file.iter_segments():
            seg_vaddr = segment['p_vaddr']
            seg_memsz = segment['p_memsz']
            seg_offset = segment['p_offset']
            seg_filesz = segment['p_filesz']

            if seg_vaddr <= virtual_address < seg_vaddr + seg_memsz:
                offset_in_segment = virtual_address - seg_vaddr
                if offset_in_segment < seg_filesz:
                    file_offset = seg_offset + offset_in_segment
                    return file_offset
        return None


def read_elf_bytes(elf_file_path, virtual_address, byte_length):
    """
    从ELF文件中读取指定长度的原始字节数据
    """
    file_offset = va_to_file_offset(elf_file_path, virtual_address)

    if file_offset is None:
        return None, None, f"无法转换地址: 0x{virtual_address:X}"

    try:
        with open(elf_file_path, 'rb') as f:
            f.seek(file_offset)

            # 读取指定长度的字节数据
            data = f.read(byte_length)

            if len(data) < byte_length:
                return data, data.hex(' ').upper(), f"读取数据不足: 期望 {byte_length} 字节，实际 {len(data)} 字节"

            hex_data = data.hex(' ').upper()
            return data, hex_data, None

    except Exception as e:
        return None, None, f"读取文件时出错: {e}"


def compare_hex_data(csv_file_path, elf_file_path):
    """
    比对CSV字符串转换的十六进制与ELF文件中的原始字节
    """
    print("=" * 100)
    print("十六进制数据比对工具")
    print("=" * 100)
    print(f"CSV文件: {csv_file_path}")
    print(f"ELF文件: {elf_file_path}")
    print("=" * 100)

    results = []
    total_count = 0
    match_count = 0
    mismatch_count = 0
    error_count = 0

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)

            for row_num, row in enumerate(reader, 1):
                if len(row) < 3:
                    print(f"第 {row_num} 行数据不完整: {row}")
                    continue

                total_count += 1

                try:
                    # 解析CSV行数据
                    address_hex = row[0].strip()
                    expected_text = row[1].strip()
                    expected_length = int(row[2].strip())

                    # 转换地址
                    virtual_address = int(address_hex, 16)

                    # 将CSV文本转换为Shift-JIS十六进制
                    expected_hex, expected_bytes, encode_error = text_to_shift_jis_hex(expected_text)

                    if encode_error:
                        status = "编码错误"
                        error_count += 1
                        result_msg = encode_error
                        actual_data = None
                        actual_hex = None
                        byte_length = 0
                    else:
                        # 计算需要读取的字节长度
                        byte_length = len(expected_bytes)

                        # 从ELF文件中读取原始字节
                        actual_data, actual_hex, read_error = read_elf_bytes(elf_file_path, virtual_address,
                                                                             byte_length)

                        if read_error:
                            status = "读取错误"
                            error_count += 1
                            result_msg = read_error
                        else:
                            # 比对十六进制数据
                            if actual_data == expected_bytes:
                                status = "匹配"
                                match_count += 1
                                result_msg = "十六进制数据完全一致"
                            else:
                                status = "不匹配"
                                mismatch_count += 1
                                result_msg = "十六进制数据不一致"

                    # 保存结果
                    result = {
                        'row': row_num,
                        'address': address_hex,
                        'expected_text': expected_text,
                        'expected_hex': expected_hex,
                        'actual_hex': actual_hex,
                        'byte_length': byte_length,
                        'status': status,
                        'message': result_msg,
                        'error': encode_error if encode_error else None
                    }
                    results.append(result)

                except ValueError as e:
                    error_count += 1
                    results.append({
                        'row': row_num,
                        'address': row[0] if len(row) > 0 else 'N/A',
                        'expected_text': row[1] if len(row) > 1 else 'N/A',
                        'expected_hex': 'N/A',
                        'actual_hex': 'N/A',
                        'byte_length': 0,
                        'status': '错误',
                        'message': f"数据格式错误: {e}",
                        'error': str(e)
                    })

        # 打印详细结果
        print(f"\n比对结果汇总:")
        print(f"总条目: {total_count}")
        print(f"匹配: {match_count}")
        print(f"不匹配: {mismatch_count}")
        print(f"错误: {error_count}")
        print("-" * 100)

        # 打印每个条目的详细结果
        for result in results:
            print(f"\n条目 {result['row']}:")
            print(f"  地址: 0x{result['address']}")
            print(f"  预期文本: \"{result['expected_text']}\"")
            print(f"  读取字节: {result['byte_length']} 字节")

            if result['status'] == '错误':
                print(f"  状态: ❌ {result['status']}")
                print(f"  错误信息: {result['message']}")
            elif result['status'] == '编码错误':
                print(f"  状态: ❌ {result['status']}")
                print(f"  错误信息: {result['message']}")
            elif result['status'] == '读取错误':
                print(f"  预期十六进制: {result['expected_hex']}")
                print(f"  状态: ❌ {result['status']}")
                print(f"  错误信息: {result['message']}")
            elif result['status'] == '匹配':
                print(f"  预期十六进制: {result['expected_hex']}")
                print(f"  实际十六进制: {result['actual_hex']}")
                print(f"  状态: ✅ {result['status']} - {result['message']}")
            else:  # 不匹配
                print(f"  预期十六进制: {result['expected_hex']}")
                print(f"  实际十六进制: {result['actual_hex']}")
                print(f"  状态: ❌ {result['status']} - {result['message']}")

                # 显示差异详情
                if result['expected_hex'] and result['actual_hex']:
                    expected_list = result['expected_hex'].split()
                    actual_list = result['actual_hex'].split()

                    diff_count = 0
                    for i, (exp, act) in enumerate(zip(expected_list, actual_list)):
                        if exp != act:
                            diff_count += 1
                            if diff_count <= 3:  # 只显示前3个差异
                                print(f"    字节 {i}: 预期 {exp} ≠ 实际 {act}")

                    if diff_count > 3:
                        print(f"    还有 {diff_count - 3} 个差异...")

        # 打印统计信息
        print("\n" + "=" * 100)
        print("最终统计:")
        print("=" * 100)
        if total_count > 0:
            print(f"✅ 匹配: {match_count}/{total_count} ({match_count / total_count * 100:.1f}%)")
            print(f"❌ 不匹配: {mismatch_count}/{total_count} ({mismatch_count / total_count * 100:.1f}%)")
            print(f"⚡ 错误: {error_count}/{total_count} ({error_count / total_count * 100:.1f}%)")
        else:
            print("没有数据可处理")

        return results

    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 {csv_file_path}")
        return []
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return []


def main():
    # 配置参数
    csv_file_path = "eboot/decode-info-fixed.csv"  # CSV文件路径
    elf_file_path = "eboot/vita3k/eboot.elf"  # ELF文件路径

    # 执行比对
    results = compare_hex_data(csv_file_path, elf_file_path)

    # 可选：将结果保存到文件
    if results:
        with open("hex_comparison_results.txt", "w", encoding="utf-8") as f:
            f.write("十六进制数据比对结果\n")
            f.write("=" * 60 + "\n")
            for result in results:
                f.write(f"地址: 0x{result['address']}\n")
                f.write(f"文本: {result['expected_text']}\n")
                f.write(f"预期十六进制: {result['expected_hex']}\n")
                f.write(f"实际十六进制: {result['actual_hex']}\n")
                f.write(f"状态: {result['status']} - {result['message']}\n")
                f.write("-" * 60 + "\n")
        print(f"\n详细结果已保存到: hex_comparison_results.txt")


if __name__ == "__main__":
    main()