import sys

assert sys.version_info >= (3, 7), "Python 3.7 or higher is required"

import os
import shutil
from Constants import *
import argparse
from pathlib import Path
from struct import unpack, unpack_from, pack


class HelpAction(argparse._HelpAction):
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
    parser_unpack.add_argument('-l', '--list',
                               default=None, dest='file list',
                               help='Output filelist path (default: none -- only unpack files)')
    parser_unpack.add_argument('input', metavar='input.mrg', help='Input .mrg file')

    parser_repack = subparsers.add_parser('repack', help='generate a hed|nam|mrg file from an existing filelist')
    parser_repack.add_argument('-l', '--list',
                               required=True, dest='file list', type=argparse.FileType('r'),
                               help='Input filelist path [REQUIRED]')
    parser_repack.add_argument('output', metavar='output.hed',
                               help='Output .hed file. Same basename is used for .nam/.mrg')

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
        self.entry_count = in_hed.stat().st_size // self.entry_length
        log_info("Estimate archive has {0} entries".format(self.entry_count))

    class Data:
        def __init__(self, raw_data):
            if len(raw_data) == 8:
                ofs_low, ofs_high, size_sect, size_low = unpack('<HHHH', raw_data)
                self.offset = 0x800 * (ofs_low | ((ofs_high & 0xF000) << 4))
                self.rounded_size = self.size = 0x800 * size_sect
                if size_low == 0:
                    self.size = self.rounded_size
                else:
                    self.size = size_low | ((0x800 * (size_sect - 1)) & 0xFFFF0000)

            elif len(raw_data) == 4:
                ofs_low, ofs_high = unpack('<HH', raw_data)
                self.offset = 0x800 * (ofs_low | ((ofs_high & 0xF000) << 4))
                self.rounded_size = self.size = 0x800 * (ofs_high & 0x0FFF)

            else:
                raise ValueError('Must a 4-byte or 8-byte binary block, source file may be incomplete')

        def to_block(self, blocksize):
            if blocksize == 8:
                ofs_aligned = self.offset // 0x800
                ofs_low = ofs_aligned & 0xFFFF
                ofs_high = (ofs_aligned & 0xF0000) >> 4
                size_low = self.size & 0xFFFF
                if size_low == 0:
                    size_sect = self.size // 0x800
                else:
                    size_sect = self.size // 0x800 + 1
                return pack('<HHHH', ofs_low, ofs_high, size_sect, size_low)

            elif blocksize == 4:
                ofs_aligned = self.offset // 0x800
                ofs_low = ofs_aligned & 0xFFFF
                ofs_high = (ofs_aligned & 0xF0000) >> 4
                return pack('<HH', ofs_low, ofs_high)

            return None

    def parse_data(self):
        self.hed_file.seek(0)

        # index = 0
        # while True:
        #     blob = self.hed_file.read(self.entry_length)
        #     # 文件已经读完
        #     if len(blob) < self.entry_length:
        #         break
        #
        #     (first_word,) = unpack_from('<L', blob)
        #     # 跳过空字符, 填充一个空数据进去, 这样能保障后续的流程不出错
        #     if first_word == 0xFFFFFFFF:
        #         self.entry_data[index] = None
        #         index += 1
        #         continue
        #
        #     # 跳过无效的数据, 这样可以不用进行统计
        #     entry = HeadEntry.Data(blob)
        #     # if entry.size == 0 or entry.offset == 0:
        #     #     continue
        #
        #     self.entry_data[index] = entry
        #     index += 1
        #
        # self.entry_count = len(self.entry_data)
        # log_info("Actual archive has {0} entries".format(self.entry_count))

        for i in range(self.entry_count):
            blob = self.hed_file.read(self.entry_length)

            # 跳过空字符, 填充一个空数据进去, 这样能保障后续的流程不出错
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
            self.nam_length = 0x8 if in_nam.name.find('voice') >= 0 else 0x20
            self.entry_count = in_nam.stat().st_size // self.nam_length
            log_info(
                "{0} is not index mode file, calc length => {1}, name count => {2}".format(
                    in_nam.name,
                    self.nam_length,
                    self.entry_count
                )
            )
        else:
            # Parse name count
            self.nam_file.seek(0x10)
            self.nam_count, = unpack("<I", self.nam_file.read(0x4))

            log_info("{0} is index mode file, name count => {1}".format(in_nam.name, self.nam_count))

            # Parse name index
            self.nam_file.seek(0x20)
            self.nam_index = {}
            for i in range(self.nam_count):
                self.nam_index[i] = unpack("<I", self.nam_file.read(0x4))
            self.nam_index[self.nam_count] = os.path.getsize(in_nam)

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


class MergedPack:
    def __init__(self, in_mrg):
        assert in_mrg.suffix.lower() == '.mrg', "Input file must be .mrg!"

        self.mrg_file = open(in_mrg, 'rb')
        suffix_isupper = in_mrg.suffix.isupper()

        in_hed = in_mrg.with_suffix(".HED" if suffix_isupper else ".hed")
        in_nam = in_mrg.with_suffix(".NAM" if suffix_isupper else ".nam")

        self.head_entry = HeadEntry(in_hed) if in_hed.exists() and in_hed.is_file() else None
        self.name_entry = NameEntry(in_nam) if in_nam.exists() and in_nam.is_file() else None

        self.output_dir = in_mrg.with_name(in_hed.stem + '-unpacked')

        # Single mrg file
        if self.head_entry is None and self.name_entry is None:
            self.is_single_file = True
            magic, self.entry_count = unpack("<6sH", self.mrg_file.read(0x8))
            assert magic == MRG_MAGIC, "Invalid mrg file => {0}".format(in_mrg.name)
        else:
            assert self.head_entry is not None, "Must have a .hed file!"

            self.is_single_file = False
            self.entry_count = self.head_entry.entry_count

    def release(self):
        self.mrg_file.close()

        if self.head_entry is not None:
            self.head_entry.release()
        if self.name_entry is not None:
            self.name_entry.release()

    def detect_file_extension(self, data):
        if data.startswith(MZX_MAGIC):
            return '.mzx'
        # elif data.startswith(b'mrgd00'):  # 之前的 mzp 头是 mrgd00... 但无法唯一判定, 因为 mrg 也是这个头
        #    return '.mzp'
        else:
            return '.bin'  # fallback

    def extract(self):
        if self.output_dir.exists():
            log_warn("Removing existing directory: {0}".format(self.output_dir))
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True)

        if self.is_single_file:
            pass
        else:
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
                    i=i, suffix=self.detect_file_extension(mrg_entry_data)
                ) if self.name_entry is None else self.name_entry.get_name(i)

                path = self.output_dir.joinpath(file_name)
                # 存在相同文件名的情况下自动重命名, 防止数据被覆盖
                if path.exists():
                    root, ext = os.path.splitext(file_name)
                    newname = root + '-' + str(i) + ext
                    path = self.output_dir.joinpath(newname)

                with open(path, 'wb') as f:
                    # can parse file magic head in this, replace the suffix
                    # like .mzp or .mzx and other suffix
                    f.write(mrg_entry_data)
                    extract_count += 1

                log_info("Extract {0} succeed, size => {1}".format(file_name, path.stat().st_size))

            log_info("Successfully extracted a total of {0} files.".format(extract_count))


def do_unpack(input_args):
    merged_pack = MergedPack(Path(input_args.input))
    merged_pack.extract()
    merged_pack.release()


if __name__ == '__main__':
    parser, args = parse_args()
    if args.subcommand == "unpack":
        do_unpack(args)
    elif args.subcommand == "repack":
        # repack_verb(args)
        print("repack_mrg")
    else:
        parser.print_usage()
        sys.exit(20)
    sys.exit(0)
