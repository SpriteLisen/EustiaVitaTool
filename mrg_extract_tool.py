import sys

assert sys.version_info >= (3, 7), "Python 3.7 or higher is required"

import os
import shutil
from Constants import *
import argparse
from pathlib import Path
from struct import unpack, unpack_from, pack


class HelpAction(argparse._HelpAction):
    # noinspection PyProtectedMember
    def __call__(self, parser_instance, namespace, values, option_string=None):
        parser_instance.print_help()
        for subparser_action in parser_instance._actions:
            if isinstance(subparser_action, argparse._SubParsersAction):
                for choice, subparser in subparser_action.choices.items():
                    print("\n\n== {} ==".format(choice))
                    print(subparser.format_help())
        parser_instance.exit()


def parse_args():
    parser_instance = argparse.ArgumentParser(description=__doc__, add_help=False)
    subparsers = parser_instance.add_subparsers(title='subcommands', dest='subcommand')

    parser_unpack = subparsers.add_parser('unpack',
                                          help='unpack mrg and optionally create a filelist, it can auto parse .head & .nam file')
    parser_unpack.add_argument('input', metavar='input.mrg', help='Input .mrg file [REQUIRED]')

    parser_repack = subparsers.add_parser('repack', help='generate a mrg file from an existing filelist')
    parser_repack.add_argument('-l', '--list',
                               required=True, dest='file_list', type=Path,
                               help='Input filelist path [REQUIRED]')
    parser_repack.add_argument(
        '--with-hed',
        action='store_true',
        help='Enable re-packing output .hed file support'
    )
    parser_repack.add_argument('input', metavar='./input', type=Path, help='Input repack dir [REQUIRED]')
    parser_repack.add_argument('output', metavar='output.mrg', type=Path, help='Output .mrg file [REQUIRED]')

    parser_instance.add_argument('-h', '--help',
                                 action=HelpAction, default=argparse.SUPPRESS,
                                 help='show this help message and exit')

    return parser_instance, parser_instance.parse_args()


class HeadEntry:
    def __init__(self, in_hed):
        log_info("Parse .hed file...")
        self.hed_file = open(in_hed, 'rb')
        self.entry_data = {}

        _, first_entry_high = unpack('<HH', self.hed_file.read(0x04))
        self.entry_length = 8 if (first_entry_high & 0x0FFF) == 0 else 4

        # 总大小需要减去末尾 16 字符的结束 FF 空数据, 才能正确的计算文件大小
        self.entry_count = (in_hed.stat().st_size - HED_END_BLOCK_SIZE) // self.entry_length
        log_info(".nam archive has {0} entries".format(self.entry_count))

    class Data:
        def __init__(self, raw_data=None, size=0, offset=0):
            if raw_data is None:
                self.size = size
                self.offset = offset

            elif len(raw_data) == 8:
                ofs_low, ofs_high, size_sect, size_low = unpack(HEADER_FORMAT, raw_data)
                self.offset = DEFAULT_SECTOR_SIZE * (ofs_low | ((ofs_high & 0xF000) << 4))
                self.rounded_size = self.size = DEFAULT_SECTOR_SIZE * size_sect
                if size_low == 0:
                    self.size = self.rounded_size
                else:
                    self.size = size_low | ((DEFAULT_SECTOR_SIZE * (size_sect - 1)) & 0xFFFF0000)

            elif len(raw_data) == 4:
                ofs_low, ofs_high = unpack('<HH', raw_data)
                self.offset = DEFAULT_SECTOR_SIZE * (ofs_low | ((ofs_high & 0xF000) << 4))
                self.rounded_size = self.size = DEFAULT_SECTOR_SIZE * (ofs_high & 0x0FFF)

            else:
                raise ValueError('Must a 4-byte or 8-byte binary block, source file may be incomplete')

        def to_block(self, blocksize, entry_index):
            if blocksize == 8:
                ofs_aligned = self.offset // DEFAULT_SECTOR_SIZE
                ofs_low = ofs_aligned & 0xFFFF
                ofs_high = (ofs_aligned & 0xF0000) >> 4

                # 判断 size_sect
                # 纠正扇区的取整, 如果不足一个扇区则按一个扇区算
                if self.size % DEFAULT_SECTOR_SIZE == 0:
                    size_sect = self.size // DEFAULT_SECTOR_SIZE
                else:
                    size_sect = self.size // DEFAULT_SECTOR_SIZE + 1

                # 判断 size_low
                # 若文件大小是 65536 的倍数, 低 16 位为 0 的情况下, 则看当前是在 16 字节的前半段偏移还是后半段偏移
                # 根据具体的偏移位置给结束的标记
                size_low = self.size & 0xFFFF
                if size_low == 0:
                    byte_pos_in_line = (entry_index * 8) % 16
                    if byte_pos_in_line < 8:
                        size_low = 0xA000
                    else:
                        size_low = 0xB000

                return pack(HEADER_FORMAT, ofs_low, ofs_high, size_sect, size_low)

            elif blocksize == 4:
                ofs_aligned = self.offset // DEFAULT_SECTOR_SIZE
                ofs_low = ofs_aligned & 0xFFFF
                ofs_high = (ofs_aligned & 0xF0000) >> 4
                return pack('<HH', ofs_low, ofs_high)

            return None

    def parse_data(self):
        self.hed_file.seek(0)

        for i in range(self.entry_count):
            blob = self.hed_file.read(self.entry_length)

            # 防止溢出读到结尾空字符的情况, 跳过空字符, 填充一个空数据进去, 这样能保障后续的流程不出错
            (first_word,) = unpack_from('<L', blob)
            if first_word == 0xFFFFFFFF:
                self.entry_data[i] = None
                continue

            self.entry_data[i] = HeadEntry.Data(blob)

    def release(self):
        self.hed_file.close()


class NameEntry:
    def __init__(self, in_nam):
        log_info("Parse .nam file...")
        self.nam_file = open(in_nam, 'rb')

        self.is_index_mode = True if self.nam_file.read(0x7) == NAM_MAGIC else False
        if not self.is_index_mode:
            self.nam_length = DEFAULT_BLOCK_SIZE if in_nam.name.find('voice') >= 0 else NAM_ENTRY_BLOCK_SIZE
            # 实际的数量需要减去一个数据块的长度, 因为尾部会多填充一个数据块的空数据作为结尾标识
            self.entry_count = (in_nam.stat().st_size - NAM_ENTRY_BLOCK_SIZE) // self.nam_length
            log_info(
                "{0} is not index mode file, block size => {1}, entry count => {2}".format(
                    in_nam.name,
                    self.nam_length,
                    self.entry_count
                )
            )
        else:
            # Parse name count
            self.nam_file.seek(0x10)
            (self.entry_count,) = unpack("<I", self.nam_file.read(0x4))

            log_info("{0} is index mode file, name count => {1}".format(in_nam.name, self.entry_count))

            # Parse name index
            self.nam_file.seek(NAM_ENTRY_BLOCK_SIZE)
            self.nam_index = {}
            for i in range(self.entry_count):
                self.nam_index[i] = unpack("<I", self.nam_file.read(0x4))
            self.nam_index[self.entry_count] = os.path.getsize(in_nam)

    def get_name(self, index):
        if self.is_index_mode:
            length = self.nam_index[index + 1] - self.nam_index[index] - 4
            self.nam_file.seek(self.nam_index[index])
            in_count, = unpack("<I", self.nam_file.read(4))
            if in_count == index:
                name = self.nam_file.read(length)
            else:
                raise Exception(
                    '{0} can not get name from index {1}, the in-header index is {2}'.format(TAG_ERR, index, in_count)
                )
        else:
            self.nam_file.seek(self.nam_length * index)
            name = self.nam_file.read(self.nam_length)

        try:
            return name[0:name.index(b'\x00')].decode(DEFAULT_ENCODING)
        except ValueError:
            return name.decode(DEFAULT_ENCODING)

    def release(self):
        self.nam_file.close()


class ArchiveEntry:
    def __init__(self, sector_offset, offset, sector_size_upper_boundary, size, number_of_entries):
        self.sector_offset = sector_offset
        self.offset = offset
        self.sector_size_upper_boundary = sector_size_upper_boundary
        self.size = size
        self.real_size = (sector_size_upper_boundary - 1) // NAM_ENTRY_BLOCK_SIZE * 0x10000 + size
        data_start_offset = 6 + 2 + number_of_entries * 8
        self.real_offset = data_start_offset + self.sector_offset * DEFAULT_SECTOR_SIZE + self.offset


class MergedPack:
    def __init__(self, in_mrg: Path = None, in_list: Path = None):
        if in_mrg is not None:
            assert in_mrg.suffix.lower() == SUFFIX_MRG, "Input file must be .mrg!"

            self.mrg_file = open(in_mrg, 'rb')
            suffix_isupper = in_mrg.suffix.isupper()

            in_hed = in_mrg.with_suffix(".HED" if suffix_isupper else SUFFIX_HED)
            in_nam = in_mrg.with_suffix(".NAM" if suffix_isupper else SUFFIX_NAM)

            self.head_entry = HeadEntry(in_hed) if in_hed.exists() and in_hed.is_file() else None
            self.name_entry = NameEntry(in_nam) if in_nam.exists() and in_nam.is_file() else None

            if self.head_entry is not None and self.name_entry is not None:
                assert self.head_entry.entry_count == self.name_entry.entry_count, "The entry count in .hed and .nam must be the same."

            self.output_dir = in_mrg.with_name(in_hed.stem + '-unpacked')

            if self.output_dir.exists():
                log_warn("Removing existing directory: {0}".format(self.output_dir))
                shutil.rmtree(self.output_dir)

            self.output_dir.mkdir(parents=True)

            self.list_file = open(
                self.output_dir.joinpath(LIST_FILE_NAME), 'w', encoding='utf-8'
            )

            # Single mrg file
            if self.head_entry is None and self.name_entry is None:
                self.is_single_file = True
                magic, self.entry_count = unpack("<6sH", self.mrg_file.read(DEFAULT_BLOCK_SIZE))
                assert magic == MRG_MAGIC, "Invalid mrg file => {0}".format(in_mrg.name)
            else:
                assert self.head_entry is not None, "Must have a .hed file!"

                self.is_single_file = False
                self.entry_count = self.head_entry.entry_count

            log_info("Start with extract mode.")
        elif in_list is not None:
            assert in_list.name == LIST_FILE_NAME, "list file name must be {0}".format(LIST_FILE_NAME)

            list_file_path = Path(in_list)
            with open(list_file_path, 'r', encoding='utf-8') as f:
                self.file_names = [line.strip() for line in f.readlines() if line.strip()]

            assert len(self.file_names) > 0, "file list must not be empty"

            log_info("Start with re-pack mode.")
            log_info("Will re-pack {0} files.".format(len(self.file_names)))
        else:
            raise Exception("MergedPack init error, must input .mrg or list file.")

    def release(self):
        mrg_file = getattr(self, 'mrg_file', None)
        if mrg_file is not None:
            mrg_file.close()

        head_entry = getattr(self, 'head_entry', None)
        if head_entry is not None:
            head_entry.release()

        name_entry = getattr(self, 'name_entry', None)
        if name_entry is not None:
            name_entry.release()

    def extract_single_archive(self, entry_count):
        # 跳过文件头, 直接从数据位开始 load
        self.mrg_file.seek(8)

        # Parse Entries desc
        entries_descriptors = []
        for i in range(entry_count):
            sector_offset, offset, sector_size_upper_boundary, size = unpack(HEADER_FORMAT, self.mrg_file.read(8))
            entries_descriptors.append(
                ArchiveEntry(
                    sector_offset=sector_offset, offset=offset,
                    sector_size_upper_boundary=sector_size_upper_boundary,
                    size=size, number_of_entries=entry_count
                )
            )

        # Parse name
        file_names = []
        for i in range(entry_count):
            file_name = "entry_{i:03}{suffix}".format(
                i=i, suffix=SUFFIX_MZX
            )
            file_names.append(file_name)

        # Do extract action
        extract_count = 0
        for index, entry in enumerate(entries_descriptors):
            self.mrg_file.seek(entry.real_offset)
            data = self.mrg_file.read(entry.real_size)
            suffix = detect_file_extension(data)

            real_file_name = file_names[index].replace(SUFFIX_MZX, suffix)

            output_file_name = self.output_dir.joinpath(real_file_name)
            with open(output_file_name, 'wb') as output_file:
                output_file.write(data)
                extract_count += 1

            self.list_file.write(real_file_name + "\n")
            log_info("Extract {0} succeed, size => {1}".format(real_file_name, output_file_name.stat().st_size))

        log_succeed("Successfully extracted a total of {0} files.".format(extract_count))

    def extract_combin_archive(self):
        extract_count = 0
        self.head_entry.parse_data()

        for i in range(self.entry_count):
            entry_data = self.head_entry.entry_data[i]

            # 跳过空的数据不进行写入
            if entry_data is None:
                log_warn("Pass empty entry data, index => {0}".format(i))
                continue

            self.mrg_file.seek(entry_data.offset)
            mrg_entry_data = self.mrg_file.read(entry_data.size)

            file_name = "entry_{i:03}.{suffix}".format(
                i=i, suffix=detect_file_extension(mrg_entry_data)
            ) if self.name_entry is None else self.name_entry.get_name(i)

            path = self.output_dir.joinpath(file_name)
            # 存在相同文件名的情况下自动重命名, 防止数据被覆盖
            if path.exists():
                root, ext = os.path.splitext(file_name)
                file_name = root + '-' + str(i) + ext
                path = self.output_dir.joinpath(file_name)

            with open(path, 'wb') as f:
                # can parse file magic head in this, replace the suffix
                # like .mzp or .mzx and other suffix
                f.write(mrg_entry_data)
                extract_count += 1

            self.list_file.write(file_name + "\n")
            log_info("Extract {0} succeed, size => {1}".format(file_name, path.stat().st_size))

        log_succeed("Successfully extracted a total of {0} files.".format(extract_count))

    def extract(self):
        if self.is_single_file:
            self.mrg_file.seek(0)
            file_head = self.mrg_file.read(DEFAULT_BLOCK_SIZE)

            suffix = detect_file_extension(file_head)

            (entry_count,) = unpack('<H', file_head[6:8])
            log_info("Found {0} entries in archive.".format(entry_count))

            if suffix == SUFFIX_MRG:
                log_info("This mrg file is a single archive.")
                self.extract_single_archive(entry_count)
            else:
                log_error("Unsupported this archive file => {0}".format(self.mrg_file.name))
        else:
            self.extract_combin_archive()

    def repack_single_mrg_file(self, input_path: Path, output_file: Path):
        log_info("Start re-packing single .mrg file")

        mrg_file = open(output_file, "wb")

        sections = []
        for file_name in self.file_names:
            with open(input_path.joinpath(file_name), "rb") as f:
                sections.append(f.read())

        entry_count = len(sections)

        # 写入头部信息
        mrg_file.write(pack("<6sH", MRG_MAGIC, entry_count))

        # === 正确计算真实 offset ===
        current_offset = 0

        for section in sections:
            sector_offset = current_offset // DEFAULT_SECTOR_SIZE
            byte_offset = current_offset % DEFAULT_SECTOR_SIZE

            raw_sectors = (len(section) + DEFAULT_SECTOR_SIZE - 1) // DEFAULT_SECTOR_SIZE

            if byte_offset + len(section) > DEFAULT_SECTOR_SIZE * raw_sectors:
                size_sectors = raw_sectors + 1
            else:
                size_sectors = raw_sectors

            mrg_file.write(
                pack(
                    "<HHHH", sector_offset, byte_offset,
                    size_sectors, len(section) & 0xFFFF
                )
            )

            current_offset += len(section)
            current_offset += padding_bytes_needed(current_offset)

        # 写入实际文件数据
        for section in sections:
            mrg_file.write(section)

            # 写入 section 之前先补齐
            current_offset = mrg_file.tell()

            pad_len = padding_bytes_needed(current_offset)
            if pad_len > 0:
                mrg_file.write(b'\xFF' * pad_len)

        mrg_file.close()

        log_succeed(f"Successfully re-packed archive file => {output_file.name}")

    def repack_triple_mrg(self, input_path: Path, output_path: Path):
        # 删掉旧的 .hed 输出文件
        hed_path = output_path.with_suffix(SUFFIX_HED)
        if hed_path.exists():
            hed_path.unlink()
            log_warn("Removing existing output file: {0}".format(hed_path.name))

        # 删掉旧的 .nam 输出文件
        nam_path = output_path.with_suffix(SUFFIX_NAM)
        if nam_path.exists():
            nam_path.unlink()
            log_warn("Removing existing output file: {0}".format(nam_path.name))

        log_info("Start re-packing triple files: .hed, .mrg, .nam")

        hed_entries = []

        # 写出 .mrg 文件 (.hed 单独处理, 此处仅写入文件压缩内容)
        mrg_file = open(output_path, "wb")

        for file_name in self.file_names:
            file_path = input_path.joinpath(file_name)
            entry_file = open(file_path, "rb")

            entry = HeadEntry.Data(
                size=file_path.stat().st_size,
                offset=mrg_file.tell()
            )
            hed_entries.append(entry)

            remaining = entry.size
            while remaining > 32768:
                mrg_file.write(entry_file.read(32768))
                remaining -= 32768
            if remaining > 0:
                mrg_file.write(entry_file.read(remaining))
            complement = entry.size & 0x7FF
            if complement > 0:
                while complement & 0xF != 0:
                    mrg_file.write(b'\x00')
                    complement += 1
                while complement & 0x7FF != 0:
                    mrg_file.write(b'\x0C' + 15 * b'\x00')
                    complement += 0x10

            entry_file.close()

            log_info(f"Packed {file_name} => offset={entry.offset:08X}, size={entry.size}")

        mrg_file.close()

        # 写出 .hed 文件
        with open(hed_path, "wb") as f_hed:
            for index, entry in enumerate(hed_entries):
                f_hed.write(entry.to_block(DEFAULT_BLOCK_SIZE, index))
            # 末尾写 16字节 0xFF 标记 .hed 文件结束
            f_hed.write(END_PADDING_DATA * 16)

        # 写出 .nam 文件
        with open(nam_path, 'wb') as f_nam:
            for i, file_name in enumerate(self.file_names):
                # 还原的时候需要去掉提取时因文件重名而增加的后缀名称
                file_name = file_name.replace('-' + str(i), "")
                # 限制文件名最长 32 字节（含后缀）
                lite_name = file_name.encode('ascii')
                lite_name = lite_name[:NAM_ENTRY_BLOCK_SIZE]
                f_nam.write(lite_name)

                # 补齐到 32 字节, 前面都是 00, 后面 2个字节是换行符
                pad_len = NAM_ENTRY_BLOCK_SIZE - 2 - len(lite_name)
                if pad_len > 0:
                    f_nam.write(EMPTY_PADDING_DATA * pad_len)

                # 写入换行符 CR LF
                f_nam.write(b'\x0D\x0A')

            # 文件结尾补全 32 字节 0
            f_nam.write(EMPTY_PADDING_DATA * NAM_ENTRY_BLOCK_SIZE)

        log_succeed(f"Successfully re-packed: {hed_path.name}, {output_path.name}, {nam_path.name}")

    def repack(self, with_hed: bool, input_path: Path, output_file: Path):
        if output_file.exists():
            output_file.unlink()
            log_warn("Removing existing output file: {0}".format(output_file.name))

        if with_hed:
            self.repack_triple_mrg(input_path, output_file)
        else:
            self.repack_single_mrg_file(input_path, output_file)


def do_unpack(input_args):
    merged_pack = MergedPack(in_mrg=Path(input_args.input))
    merged_pack.extract()
    merged_pack.release()


def do_repack(input_args):
    assert input_args.input is not None, "Input dir required."
    assert input_args.output is not None, "Output file name required."

    merged_pack = MergedPack(in_list=input_args.file_list)
    merged_pack.repack(with_hed=input_args.with_hed, input_path=input_args.input, output_file=input_args.output)
    merged_pack.release()


if __name__ == '__main__':
    parser, args = parse_args()
    if args.subcommand == "unpack":
        do_unpack(args)
    elif args.subcommand == "repack":
        do_repack(args)
    else:
        parser.print_usage()
        sys.exit(EXIT_WITH_HELP)
    sys.exit(0)
