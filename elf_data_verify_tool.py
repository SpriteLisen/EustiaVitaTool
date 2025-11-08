import csv
from elftools.elf.elffile import ELFFile


def safe_shift_jis_decode(data):
    """
    安全的Shift-JIS解码函数，处理无法解码的字符
    """
    try:
        # 首先尝试正常解码
        return data.decode('shift_jis'), None
    except UnicodeDecodeError as e:
        # 如果解码失败，尝试逐个字节处理
        result = []
        i = 0
        error_positions = []

        while i < len(data):
            try:
                # 尝试解码单个或双字节字符
                if i + 1 < len(data):
                    # 尝试双字节字符
                    try:
                        char = data[i:i + 2].decode('shift_jis')
                        result.append(char)
                        i += 2
                        continue
                    except UnicodeDecodeError:
                        pass

                # 尝试单字节字符
                try:
                    char = data[i:i + 1].decode('shift_jis')
                    result.append(char)
                    i += 1
                except UnicodeDecodeError:
                    # 无法解码的字节，用转义序列表示
                    result.append(f"\\x{data[i]:02x}")
                    error_positions.append(i)
                    i += 1

            except Exception:
                result.append(f"\\x{data[i]:02x}")
                error_positions.append(i)
                i += 1

        decoded_text = ''.join(result)
        error_msg = f"部分字符解码失败" if error_positions else "解码失败"
        return decoded_text, error_msg


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


def read_elf_string(elf_file_path, virtual_address, expected_length):
    """
    从ELF文件中读取字符串内容
    """
    file_offset = va_to_file_offset(elf_file_path, virtual_address)

    if file_offset is None:
        return None, None, f"无法转换地址: 0x{virtual_address:X}"

    try:
        with open(elf_file_path, 'rb') as f:
            f.seek(file_offset)

            # 直接读取指定长度的字节数
            bytes_to_read = expected_length
            data = f.read(bytes_to_read)

            if len(data) < bytes_to_read:
                return None, data.hex(' ').upper(), f"读取数据不足: 期望 {bytes_to_read} 字节，实际 {len(data)} 字节"

            # 使用安全的Shift-JIS解码
            decoded_text, error_msg = safe_shift_jis_decode(data)

            if error_msg:
                return decoded_text, data.hex(' ').upper(), error_msg
            else:
                return decoded_text, data.hex(' ').upper(), None

    except Exception as e:
        return None, None, f"读取文件时出错: {e}"


def compare_strings(csv_file_path, elf_file_path):
    """
    比对CSV中的字符串数据与ELF文件中的实际内容
    """
    print("=" * 100)
    print("字符串内容比对工具")
    print("=" * 100)
    print(f"CSV文件: {csv_file_path}")
    print(f"ELF文件: {elf_file_path}")
    print("=" * 100)

    results = []
    total_count = 0
    match_count = 0
    mismatch_count = 0
    error_count = 0
    partial_count = 0

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
                    address_hex = row[0]
                    expected_text = row[1]
                    expected_length = int(row[2])

                    # 转换地址
                    virtual_address = int(address_hex, 16)

                    # 从ELF文件中读取实际内容
                    actual_text, hex_data, error = read_elf_string(elf_file_path, virtual_address, expected_length)

                    if error:
                        if "部分字符解码失败" in error:
                            status = "部分解码"
                            partial_count += 1
                            result_msg = error
                        else:
                            status = "错误"
                            error_count += 1
                            result_msg = error
                    else:
                        # 比对内容
                        if actual_text == expected_text:
                            status = "匹配"
                            match_count += 1
                            result_msg = "内容一致"
                        else:
                            status = "不匹配"
                            mismatch_count += 1
                            result_msg = f"内容不一致"

                    # 保存结果
                    result = {
                        'row': row_num,
                        'address': address_hex,
                        'expected': expected_text,
                        'expected_length': expected_length,
                        'actual': actual_text if actual_text else hex_data,
                        'actual_length': len(actual_text) if isinstance(actual_text, str) else 0,
                        'hex_data': hex_data,
                        'status': status,
                        'message': result_msg,
                        'error': error
                    }
                    results.append(result)

                except ValueError as e:
                    error_count += 1
                    results.append({
                        'row': row_num,
                        'address': row[0] if len(row) > 0 else 'N/A',
                        'expected': row[1] if len(row) > 1 else 'N/A',
                        'actual': 'N/A',
                        'hex_data': 'N/A',
                        'status': '错误',
                        'message': f"数据格式错误: {e}",
                        'error': str(e)
                    })

        # 打印详细结果
        print(f"\n比对结果汇总:")
        print(f"总条目: {total_count}")
        print(f"匹配: {match_count}")
        print(f"不匹配: {mismatch_count}")
        print(f"部分解码: {partial_count}")
        print(f"错误: {error_count}")
        print("-" * 100)

        # 打印每个条目的详细结果
        for result in results:
            print(f"\n条目 {result['row']}:")
            print(f"  地址: 0x{result['address']}")
            print(f"  预期文本: \"{result['expected']}\"")
            print(f"  预期长度: {result['expected_length']} 字符")
            print(f"  读取字节: {result['expected_length']} 字节")

            if result['status'] == '错误':
                print(f"  状态: ❌ {result['status']}")
                print(f"  错误信息: {result['message']}")
                print(f"  十六进制: {result['hex_data']}")
            elif result['status'] == '部分解码':
                print(f"  实际文本: \"{result['actual']}\"")
                print(f"  实际长度: {result['actual_length']} 字符")
                print(f"  十六进制: {result['hex_data']}")
                print(f"  状态: ⚠️ {result['status']} - {result['message']}")
            elif result['status'] == '匹配':
                print(f"  实际文本: \"{result['actual']}\"")
                print(f"  十六进制: {result['hex_data']}")
                print(f"  状态: ✅ {result['status']} - {result['message']}")
            else:  # 不匹配
                print(f"  实际文本: \"{result['actual']}\"")
                print(f"  实际长度: {result['actual_length']} 字符")
                print(f"  十六进制: {result['hex_data']}")
                print(f"  状态: ❌ {result['status']} - {result['message']}")

        # 打印统计信息
        print("\n" + "=" * 100)
        print("最终统计:")
        print("=" * 100)
        if total_count > 0:
            print(f"✅ 匹配: {match_count}/{total_count} ({match_count / total_count * 100:.1f}%)")
            print(f"⚠️  部分解码: {partial_count}/{total_count} ({partial_count / total_count * 100:.1f}%)")
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
    elf_file_path = "eboot/device/eboot.elf"  # ELF文件路径

    # 执行比对
    results = compare_strings(csv_file_path, elf_file_path)

    # 可选：将结果保存到文件
    if results:
        with open("comparison_results.txt", "w", encoding="utf-8") as f:
            f.write("字符串比对结果\n")
            f.write("=" * 50 + "\n")
            for result in results:
                f.write(f"地址: 0x{result['address']}\n")
                f.write(f"预期: {result['expected']} (长度: {result['expected_length']})\n")
                f.write(f"实际: {result['actual']} (长度: {result['actual_length']})\n")
                f.write(f"十六进制: {result['hex_data']}\n")
                f.write(f"状态: {result['status']} - {result['message']}\n")
                f.write("-" * 50 + "\n")
        print(f"\n详细结果已保存到: comparison_results.txt")


if __name__ == "__main__":
    main()