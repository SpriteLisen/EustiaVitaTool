import csv
import shutil
from elftools.elf.elffile import ELFFile


def text_to_shift_jis_bytes(text):
    """
    将文本转换为Shift-JIS编码的字节
    """
    try:
        return text.encode('CP932'), None
    except UnicodeEncodeError as e:
        return None, f"Shift-JIS编码失败: {e}"


def get_space_padding(original_bytes, target_length):
    """
    生成适当的空格填充
    根据上下文选择半角或全角空格
    """
    # padding_bytes = b''
    # remaining = target_length - len(original_bytes)
    #
    # if remaining <= 0:
    #     return original_bytes
    #
    # # 检查原始字节的最后一个字符是否为全角字符
    # use_fullwidth = False
    # if len(original_bytes) >= 2:
    #     # 如果最后一个字符是全角字符，使用全角空格
    #     last_char = original_bytes[-2:]
    #     if len(last_char) == 2 and last_char[0] >= 0x81:
    #         use_fullwidth = True
    #
    # if use_fullwidth:
    #     # 使用全角空格 (81 40)
    #     fullwidth_space = b'\x81\x40'
    #     padding_count = remaining // 2
    #     padding_bytes = fullwidth_space * padding_count
    #
    #     # 如果还有剩余1字节，使用半角空格
    #     if remaining % 2 == 1:
    #         padding_bytes += b'\x20'
    # else:
    #     # 使用半角空格 (20)
    #     padding_bytes = b'\x20' * remaining
    #
    # return original_bytes + padding_bytes

    remaining = target_length - len(original_bytes)
    if remaining <= 0:
        return original_bytes

    # 尝试找到一个可以用来填充的安全字节, 00, ff 都不行
    safe_byte = b'\x00'
    padding_bytes = safe_byte * remaining

    return original_bytes + padding_bytes


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
        return None, f"无法转换地址: 0x{virtual_address:X}"

    try:
        with open(elf_file_path, 'rb') as f:
            f.seek(file_offset)
            data = f.read(byte_length)
            return data, None
    except Exception as e:
        return None, f"读取文件时出错: {e}"


def write_elf_bytes(elf_file_path, virtual_address, data):
    """
    将数据写入ELF文件的指定位置
    """
    file_offset = va_to_file_offset(elf_file_path, virtual_address)

    if file_offset is None:
        return f"无法转换地址: 0x{virtual_address:X}"

    try:
        with open(elf_file_path, 'r+b') as f:
            f.seek(file_offset)
            f.write(data)
        return None
    except Exception as e:
        return f"写入文件时出错: {e}"


def load_translation_dict(translation_file):
    """
    加载翻译对照表
    """
    translation_dict = {}
    try:
        with open(translation_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) >= 2:
                    original = row[0].strip()
                    translated = row[1].strip()
                    translation_dict[original] = translated
        return translation_dict, None
    except FileNotFoundError:
        return None, f"找不到翻译文件: {translation_file}"
    except Exception as e:
        return None, f"读取翻译文件时出错: {e}"


def analyze_original_data(original_data):
    """
    分析原始数据的填充模式
    """
    if not original_data:
        return "半角空格"

    # 检查是否以null终止
    if original_data.endswith(b'\x00'):
        return "null终止"

    # 检查是否以空格填充
    if original_data.endswith(b'\x20'):
        return "半角空格"

    if original_data.endswith(b'\x81\x40'):
        return "全角空格"

    # 检查混合模式
    if b'\x20' in original_data[-10:] or b'\x81\x40' in original_data[-10:]:
        return "混合空格"

    return "未知模式"


def process_elf_with_translation(original_elf_path, output_elf_path, decode_info_path, translation_path):
    """
    处理ELF文件，用翻译文本替换原始文本
    """
    print("=" * 80)
    print("ELF文件翻译替换工具 (使用空格填充)")
    print("=" * 80)
    print(f"原始ELF文件: {original_elf_path}")
    print(f"输出ELF文件: {output_elf_path}")
    print(f"地址信息文件: {decode_info_path}")
    print(f"翻译文件: {translation_path}")
    print("=" * 80)

    # 加载翻译对照表
    translation_dict, error = load_translation_dict(translation_path)
    if error:
        print(f"错误: {error}")
        return False

    print(f"加载了 {len(translation_dict)} 条翻译")

    # 创建输出文件的副本
    try:
        shutil.copy2(original_elf_path, output_elf_path)
        print(f"已创建输出文件副本: {output_elf_path}")
    except Exception as e:
        print(f"创建文件副本时出错: {e}")
        return False

    processed_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    unchanged_count = 0

    try:
        with open(decode_info_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)

            for row_num, row in enumerate(reader, 1):
                if len(row) < 3:
                    continue

                processed_count += 1

                address_hex = row[0].strip()
                original_text = row[1].strip()
                char_length = int(row[2].strip())
                byte_length = char_length  # 转换为字节长度

                # 检查是否有翻译
                if original_text not in translation_dict:
                    skipped_count += 1
                    continue

                translated_text = translation_dict[original_text]
                virtual_address = int(address_hex, 16)

                print(f"\n处理条目 {row_num}:")
                print(f"  地址: 0x{address_hex}")
                print(f"  原始文本: \"{original_text}\"")
                print(f"  翻译文本: \"{translated_text}\"")
                print(f"  原始字节长度: {byte_length}")

                # 如果翻译文本与原始文本相同，则跳过
                if translated_text == original_text:
                    print(f"  ⏭️  翻译文本与原始文本相同，跳过")
                    unchanged_count += 1
                    continue

                # 读取原始数据用于分析填充模式
                original_data, read_error = read_elf_bytes(original_elf_path, virtual_address, byte_length)
                if read_error:
                    print(f"  ❌ 读取原始数据错误: {read_error}")
                    error_count += 1
                    continue

                # 分析原始数据的填充模式
                padding_mode = analyze_original_data(original_data)
                print(f"  原始填充模式: {padding_mode}")

                # 将翻译文本转换为Shift-JIS字节
                translated_bytes, encode_error = text_to_shift_jis_bytes(translated_text)
                if encode_error:
                    print(f"  ❌ 编码错误: {encode_error}")
                    error_count += 1
                    continue

                translated_byte_length = len(translated_bytes)
                # translated_byte_length = len(translated_bytes) * 2
                print(f"  翻译字节长度: {translated_byte_length}")

                # 检查长度
                if translated_byte_length > byte_length:
                    print(f"  ❌ 错误: 翻译文本过长 ({translated_byte_length} > {byte_length})")
                    error_count += 1
                    continue

                # 使用适当的填充
                if translated_byte_length < byte_length:
                    padding_length = byte_length - translated_byte_length
                    print(f"  需要填充 {padding_length} 字节")

                    # 使用智能填充函数
                    translated_bytes = get_space_padding(translated_bytes, byte_length)

                # 写入翻译数据
                write_error = write_elf_bytes(output_elf_path, virtual_address, translated_bytes)
                if write_error:
                    print(f"  ❌ 写入错误: {write_error}")
                    error_count += 1
                    continue

                # 验证写入
                written_data, verify_error = read_elf_bytes(output_elf_path, virtual_address, byte_length)
                if verify_error:
                    print(f"  ❌ 验证错误: {verify_error}")
                    error_count += 1
                    continue

                if written_data == translated_bytes:
                    print(f"  ✅ 成功写入并验证")
                    success_count += 1
                else:
                    print(f"  ❌ 验证失败: 写入的数据不匹配")
                    print(f"    期望: {translated_bytes.hex()}")
                    print(f"    实际: {written_data.hex()}")
                    error_count += 1

        # 打印统计信息
        print("\n" + "=" * 80)
        print("处理完成!")
        print("=" * 80)
        print(f"总处理条目: {processed_count}")
        print(f"成功替换: {success_count}")
        print(f"跳过(无翻译): {skipped_count}")
        print(f"跳过(相同文本): {unchanged_count}")
        print(f"错误: {error_count}")

        if error_count == 0:
            print(f"\n✅ 所有翻译已成功应用到: {output_elf_path}")
            return True
        else:
            print(f"\n⚠️  有 {error_count} 个错误发生，请检查输出")
            return False

    except FileNotFoundError:
        print(f"错误: 找不到文件 {decode_info_path}")
        return False
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return False


def main():
    # 配置参数
    original_elf_path = "eboot/vita3k/eboot.elf"  # 原始ELF文件
    output_elf_path = "eboot/vita3k/eboot_patched.elf"  # 输出ELF文件
    decode_info_path = "eboot/decode-info-fixed.csv"  # 地址信息文件
    translation_path = "eboot/translate-info.csv"  # 翻译文件

    # 执行处理
    success = process_elf_with_translation(
        original_elf_path,
        output_elf_path,
        decode_info_path,
        translation_path
    )

    if success:
        print("\n🎉 处理完成! 新的ELF文件已生成。")
    else:
        print("\n❌ 处理失败!")


if __name__ == "__main__":
    main()
