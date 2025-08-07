import argparse
import io
import json
import math
import zlib
from pathlib import Path
from struct import unpack, pack

import numpy as np
from PIL import Image

from Constants import *

output_dir = "mzp-unpacked"

OPEN_DEBUG_FILE = False


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
                                          help='unpack mzp file')
    parser_unpack.add_argument('input', metavar='input.mzp', type=Path, help='Input .mzp file [REQUIRED]')

    parser_repack = subparsers.add_parser('repack', help='generate a mzp file from an existing file')
    parser_repack.add_argument('input', metavar='./input.*', type=Path, help='Input repack file [REQUIRED]')
    parser_repack.add_argument('output', metavar='output.mzp', type=Path, help='Output .mzp file [REQUIRED]')

    parser_instance.add_argument('-h', '--help',
                                 action=HelpAction, default=argparse.SUPPRESS,
                                 help='show this help message and exit')

    return parser_instance, parser_instance.parse_args()


def is_indexed_bitmap(bmp_info):
    return bmp_info == 0x01


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def write_pngsig(f):
    f.write(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A')


def write_pngchunk_withcrc(f, t, data):
    f.write(pack(">I", len(data)))
    f.write(t)
    f.write(data)
    f.write(pack(">I", zlib.crc32(t + data, 0) & 0xffffffff))


def write_ihdr(f, width, height, depth, color):
    chunk = pack(">IIBB", width, height, depth, color) + b'\0\0\0'
    write_pngchunk_withcrc(f, b"IHDR", chunk)


def write_plte(f, palettebin):
    write_pngchunk_withcrc(f, b"PLTE", palettebin)


def write_trns(f, transparencydata):
    write_pngchunk_withcrc(f, b"tRNS", transparencydata)


def write_idat(f, pixels):
    write_pngchunk_withcrc(f, b"IDAT", zlib.compress(pixels))


def write_iend(f):
    write_pngchunk_withcrc(f, b"IEND", b"")


class MzxCmd:
    RLE = 0
    BACKREF = 1
    RINGBUF = 2
    LITERAL = 3


class RingBuffer:
    def __init__(self, size: int, base_value: int = None) -> None:
        self._size = size

        if base_value is not None:
            self._file = io.BytesIO(bytes([base_value]) * size)
        else:
            self._file = io.BytesIO()

    def append(self, buffer: bytes) -> None:
        pos = self._file.tell()

        if pos + len(buffer) <= self._size:
            self._file.write(buffer)
        else:
            split_index = self._size - pos
            self._file.write(buffer[:split_index])
            self._file.seek(0)
            self._file.write(buffer[split_index:])

    def get(self, index: int, size: int) -> bytes:
        if self._file.tell() == 0:
            return b''

        pos = self._file.tell()
        self._file.seek(index)
        result = self._file.read(size)
        self._file.seek(pos)
        return result

    def peek_back(self, size: int) -> bytes:
        pos = self._file.tell()
        read_pos = pos - size
        if read_pos >= 0:
            self._file.seek(read_pos)
            return self._file.read(size)
        else:
            self._file.seek(read_pos, 2)
            part1 = self._file.read(-read_pos)
            self._file.seek(0)
            return part1 + self._file.read(pos)

    def word_position(self, word: bytes, word_size: int):
        pos = self._file.tell()
        self._file.seek(0)
        index = -1
        while self._file.tell() < self._size:
            stored = self._file.read(word_size)
            if not stored:
                break
            if stored == word:
                index = self._file.tell() - word_size
                break
        self._file.seek(pos)
        return index


def rle_compress_length(words: np.ndarray, cursor: int, clear_count: int,
                        invert_bytes: bool):
    word = words[cursor]
    if clear_count <= 0 or clear_count == 0x1000:
        last = 0xFFFF if invert_bytes else 0x0000
    else:
        last = words[cursor - 1]
    if word == last:
        max_chunk_size = min(64, words.size - cursor)
        max_chunk_size = min(max_chunk_size, clear_count)
        length = 1
        while length < max_chunk_size and words[cursor + length] == last:
            length += 1
        return length
    else:
        return 0


def backref_compress_length(words: np.ndarray, cursor: int, clear_count: int):
    start = max(cursor - 255, 0)
    occurrences = np.where(words[start:cursor] == words[cursor])[0]
    max_length = min(64, clear_count)
    max_dist = 0x1000 - clear_count
    best_index = -1
    best_len = 0
    for o in occurrences:
        distance = cursor - (start + o)
        if distance > max_dist or distance < 0:
            continue
        length = 1
        while (length < max_length and
               cursor + length < words.size and
               words[start + o + length] == words[cursor + length]):
            length += 1
        if (length > best_len or (length == best_len and
                                  distance < (cursor - (start + best_index)))):
            best_index = o
            best_len = length
    if best_len > 0:
        distance = cursor - (start + best_index)
        return distance, best_len
    else:
        return 0, 0


def literal_compress(output: io.BytesIO, words: np.ndarray, start: int,
                     length: int, invert: bool):
    assert length <= 64
    cmd = (MzxCmd.LITERAL | ((length - 1) << 2))
    output.write(cmd.to_bytes(1, 'little'))
    if invert:
        chunk = words[start:start + length] ^ 0xFFFF
    else:
        chunk = words[start:start + length]
    output.write(chunk.tobytes())


# noinspection PyUnboundLocalVariable
def mzx_compress(src: io.BytesIO, invert=False, level=2):
    src.seek(0)

    # 读取原始数据
    data = src.getvalue()
    words = np.frombuffer(data, dtype=np.uint8)
    if words.size % 2 == 1:
        words = np.append(words, 0x00)
    words = words.view(dtype="<u2")

    inversion_xor = 0xFFFF if invert else 0x0000
    end = words.size

    output = io.BytesIO()

    # 写文件头 MZX0 + 4字节原始长度
    output.write(MZX_MAGIC)
    output.write(len(data).to_bytes(4, 'little'))

    if level == 0:
        cursor = 0
        while cursor < end:
            chunk_size = min(end - cursor, 64)
            cmd = (MzxCmd.LITERAL | ((chunk_size - 1) << 2))
            output.write(cmd.to_bytes(1, 'little'))
            chunk = (words[cursor:cursor + chunk_size] ^ inversion_xor).tobytes()
            output.write(chunk)
            cursor += chunk_size
    else:
        clear_count = 0x1000
        ring_buf = RingBuffer(128)
        lit_start = 0
        lit_len = 0
        cursor = 0

        while cursor < end:
            best_len = 0
            best_type = 0  # LITERAL
            current_word = (words[cursor] ^ inversion_xor).tobytes()

            rle_len = rle_compress_length(words, cursor, clear_count, invert)
            if rle_len > 0:
                best_len = rle_len
                best_type = 1  # RLE

            if cursor > 0:
                if best_len < 64 and level >= 2:
                    br_dist, br_len = backref_compress_length(words, cursor, clear_count)
                    if br_len > best_len + 1 and not (
                            best_len > 0 and best_len * 2 + 1 >= br_len and 0x1000 - clear_count + best_len >= br_dist):
                        best_len = br_len
                        best_type = 2  # BACKREF

                if best_len == 0 and level >= 2:
                    rb_index = ring_buf.word_position(current_word, 2)
                    if rb_index >= 0:
                        best_len = rb_len = 1
                        best_type = 3  # RINGBUF
                        rb_index //= 2

            if best_type == 0:
                # LITERAL 批量写入
                if lit_len == 0:
                    lit_start = cursor
                    lit_len = 1
                elif lit_len == 63:
                    literal_compress(output, words, lit_start, 64, invert)
                    if clear_count <= 0:
                        clear_count = 0x1000
                    lit_len = 0
                else:
                    lit_len += 1

                if level >= 2:
                    ring_buf.append(current_word)

                cursor += 1
                clear_count -= 1
                if clear_count <= 0 < lit_len:
                    clear_count = 0x1000
                    literal_compress(output, words, lit_start, lit_len, invert)
                    lit_len = 0
            else:
                if lit_len > 0:
                    literal_compress(output, words, lit_start, lit_len, invert)
                    if clear_count <= 0:
                        clear_count = 0x1000
                    lit_len = 0

                if best_type == 1:
                    cmd = (MzxCmd.RLE | ((rle_len - 1) << 2))
                    output.write(cmd.to_bytes(1, 'little'))
                    clear_count -= rle_len
                elif best_type == 2:
                    cmd = (MzxCmd.BACKREF | (br_len - 1) << 2)
                    output.write(cmd.to_bytes(1, 'little'))
                    output.write(int(br_dist - 1).to_bytes(1, 'little'))
                    clear_count -= br_len
                elif best_type == 3:
                    cmd = (MzxCmd.RINGBUF | (rb_index << 2))
                    output.write(cmd.to_bytes(1, 'little'))
                    clear_count -= rb_len
                else:
                    raise RuntimeError("未知压缩类型")

                if clear_count <= 0:
                    clear_count = 0x1000
                cursor += best_len

        if lit_len > 0:
            literal_compress(output, words, lit_start, lit_len, invert)

    output.seek(0)
    return output


def mzx_decompress(src, invert_bytes=False):
    input_file = open(src, 'rb') if isinstance(src, str) else src

    output_data = io.BytesIO()

    start = input_file.tell()
    end = input_file.seek(0, 2)
    input_file.seek(start)

    filler_2bytes = b'\xFF\xFF' if invert_bytes else b'\x00\x00'
    ring_buf = RingBuffer(128, 0xFF if invert_bytes else 0)

    clear_count = 0

    while input_file.tell() < end:
        byte = input_file.read(1)
        if not byte:
            break
        flags = byte[0]
        cmd = flags & 0x03
        arg = flags >> 2

        if clear_count <= 0:
            clear_count = 0x1000

        if cmd == MzxCmd.RLE:
            if clear_count == 0x1000:
                last = filler_2bytes
            else:
                output_data.seek(-2, 1)
                last = output_data.read(2)
            output_data.write(last * (arg + 1))

        elif cmd == MzxCmd.BACKREF:
            pos = output_data.tell()
            k_byte = input_file.read(1)
            if not k_byte:
                break
            k = 2 * (k_byte[0] + 1)
            length = 2 * (arg + 1)
            output_data.seek(pos - k)
            if k < length:
                buffer = (output_data.read(k) * math.ceil(length / k))[:length]
            else:
                buffer = output_data.read(length)
            output_data.seek(pos)
            output_data.write(buffer)

        elif cmd == MzxCmd.RINGBUF:
            output_data.write(ring_buf.get(arg * 2, 2))

        else:  # MzxCmd.LITERAL
            count = (arg + 1) * 2
            buffer = input_file.read(count)
            if len(buffer) != count:
                break  # 数据不足，避免越界
            if invert_bytes:
                buffer = bytes([b ^ 0xFF for b in buffer])
            output_data.write(buffer)
            ring_buf.append(buffer)

        clear_count -= 1 if cmd == MzxCmd.RINGBUF else (arg + 1)

    output_data.seek(0)
    input_file.seek(start)

    if isinstance(src, str):
        input_file.close()

    return output_data


class Byte(object):
    def __init__(self, number):
        self.number = number

    @property
    def high(self):
        return self.number >> 4

    @property
    def low(self):
        return self.number & 0x0F


class MzpEntry:
    KEY_ENTRY_COUNT = "entry_count"
    KEY_BITMAP_BPP = "bitmap_bpp"
    KEY_WIDTH = "width"
    KEY_HEIGHT = "height"
    KEY_TILE_WIDTH = "tile_width"
    KEY_TILE_HEIGHT = "tile_height"
    KEY_TILE_X_COUNT = "tile_x_count"
    KEY_TILE_Y_COUNT = "tile_y_count"
    KEY_TILE_CROP = "tile_crop"
    KEY_ORIGIN_HEAD_DATA = "origin_head_data"

    def __init__(self, in_mzp: Path = None, in_png: Path = None, out_mzp: Path = None):
        if in_mzp is not None:
            log_info("Parse {0} file...".format(in_mzp.name))
            self.in_mzp = in_mzp
            mzp_file = in_mzp.open("rb")
            data = mzp_file.read()
            mzp_file.close()

            suffix = detect_file_extension_with_bytes(data)
            assert suffix.lower() == SUFFIX_MZP, "Target file {0} is not mzp file!".format(in_mzp.name)

            self.mzp_data = io.BytesIO(data)
            # 跳过文件头
            self.mzp_data.seek(6)
            self.entry_count, = unpack('<H', self.mzp_data.read(2))
            log_info("Found {0} entries.".format(self.entry_count))

            self.entries_descriptors = []
            for i in range(self.entry_count):
                sector_offset, offset, sector_size_upper_boundary, size = unpack('<HHHH', self.mzp_data.read(8))
                self.entries_descriptors.append(
                    ArchiveEntry(
                        sector_offset, offset,
                        sector_size_upper_boundary, size, self.entry_count
                    )
                )

            # 解析描述信息
            self.paletteblob = b''
            self.palettepng = b''
            self.transpng = b''
            self.width = 0
            self.height = 0
            self.tile_width = 0
            self.tile_height = 0
            self.tile_x_count = 0
            self.tile_y_count = 0
            self.bmp_type = None
            self.bmp_depth = None
            self.bitmap_bpp = 0
            self.palette_count = 0
            self.tile_crop = 0
            self.tile_size = 0
            self.extract_desc()

            self.bytesprepx = self.bitmap_bpp // 8
            if self.bytesprepx == 0:
                self.bytesprepx = 1
            self.debug_format()

            self.rows = [b''] * (self.height - self.tile_y_count * self.tile_crop)
        else:
            assert in_png is not None and out_mzp is not None

            json_file_path = in_png.with_suffix(".json")
            if not json_file_path.exists():
                raise FileNotFoundError(json_file_path)

            log_info("Parse {0} file...".format(json_file_path.name))
            json_file = json_file_path.open("r")
            json_data = json.load(json_file)
            json_file.close()

            self.in_png = in_png
            self.out_mzp = out_mzp
            self.entry_count = json_data[MzpEntry.KEY_ENTRY_COUNT]
            self.bitmap_bpp = json_data[MzpEntry.KEY_BITMAP_BPP]
            self.width = json_data[MzpEntry.KEY_WIDTH]
            self.height = json_data[MzpEntry.KEY_HEIGHT]
            self.tile_width = json_data[MzpEntry.KEY_TILE_WIDTH]
            self.tile_height = json_data[MzpEntry.KEY_TILE_HEIGHT]
            self.tile_x_count = json_data[MzpEntry.KEY_TILE_X_COUNT]
            self.tile_y_count = json_data[MzpEntry.KEY_TILE_Y_COUNT]
            self.tile_crop = json_data[MzpEntry.KEY_TILE_CROP]
            self.origin_head_data_hex = bytes.fromhex(json_data[MzpEntry.KEY_ORIGIN_HEAD_DATA])

    def dump_to_json(self, save_file: Path):
        log_info("Start dumps key data to json.")

        with open(save_file, "w") as json_file:
            self.mzp_data.seek(0)
            entry0_offset = self.entries_descriptors[0].real_offset
            origin_head_data_hex = self.mzp_data.read(entry0_offset).hex()

            json.dump(
                {
                    MzpEntry.KEY_ENTRY_COUNT: self.entry_count,
                    MzpEntry.KEY_BITMAP_BPP: self.bitmap_bpp,
                    MzpEntry.KEY_WIDTH: self.width,
                    MzpEntry.KEY_HEIGHT: self.height,
                    MzpEntry.KEY_TILE_WIDTH: self.tile_width,
                    MzpEntry.KEY_TILE_HEIGHT: self.tile_height,
                    MzpEntry.KEY_TILE_X_COUNT: self.tile_x_count,
                    MzpEntry.KEY_TILE_Y_COUNT: self.tile_y_count,
                    MzpEntry.KEY_TILE_CROP: self.tile_crop,
                    MzpEntry.KEY_ORIGIN_HEAD_DATA: origin_head_data_hex,
                },
                json_file, indent=4
            )

        log_info("Dumps key data to {0} succeed.".format(save_file.name))

    def dump_act_from_png(self, png_file: Path):
        if is_indexed_bitmap(self.bmp_type):
            img = Image.open(png_file)
            if img.mode == "P":
                log_info("Start dumps palette to act file.")
                palette = img.getpalette()[:256 * 3]
                palette_file_path = png_file.with_suffix(".act")
                with open(palette_file_path, "wb") as palette_file:
                    palette_file.write(bytes(palette))
                    palette_file.write(b'\x00\x00')  # ACT 文件尾部需2个空字节
                img.close()
                log_info("Dumps palette to {0} succeed.".format(palette_file_path.name))
        else:
            log_info("Png file not index mode, pass act dumps.")

    def extract_desc(self):
        self.mzp_data.seek(self.entries_descriptors[0].real_offset)
        self.width, self.height, self.tile_width, self.tile_height, self.tile_x_count, self.tile_y_count, \
            self.bmp_type, self.bmp_depth, self.tile_crop = unpack('<HHHHHHHBB', self.mzp_data.read(0x10))
        self.tile_size = self.tile_width * self.tile_height
        if self.bmp_type not in [0x01, 0x03, 0x08, 0x0B]:
            log_error("Unknown type 0x{:02X}".format(self.bmp_type))
            sys.exit(EXIT_WITH_ERROR)

        # 有索引
        if is_indexed_bitmap(self.bmp_type):
            if self.bmp_depth == 0x01:
                self.bitmap_bpp = 8
                self.palette_count = 0x100
            elif self.bmp_depth == 0x00 or self.bmp_depth == 0x10:
                self.bitmap_bpp = 4
                self.palette_count = 0x10
            elif self.bmp_depth == 0x11 or self.bmp_depth == 0x91:
                self.bitmap_bpp = 8
                self.palette_count = 0x100
            else:
                log_error("Unknown depth 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)

            if self.bmp_depth in [0x00, 0x10]:
                for i in range(self.palette_count):
                    r = self.mzp_data.read(1)
                    g = self.mzp_data.read(1)
                    b = self.mzp_data.read(1)

                    # a = self.data.read(1)
                    # Experimental
                    # 4bpp Alpha-channel to 8bpp
                    # Author: Hintay <hintay@me.com>
                    temp_a, = unpack('B', self.mzp_data.read(1))
                    a = (temp_a << 1) + (temp_a >> 6) if (temp_a < 0x80) else 255
                    a = pack('B', a)

                    self.paletteblob += (b + g + r + a)
                    self.palettepng += (r + g + b)
                    self.transpng += a

            # :PalType:RGBATim2:
            # Author: caoyang131
            elif self.bmp_depth in [0x11, 0x91, 0x01]:
                pal_start = self.mzp_data.tell()
                for h in range(0, self.palette_count * 4 // 0x80, 1):
                    for i in range(2):
                        for j in range(2):
                            self.mzp_data.seek(h * 0x80 + (i + j * 2) * 0x20 + pal_start)
                            for k in range(8):
                                r = self.mzp_data.read(1)
                                g = self.mzp_data.read(1)
                                b = self.mzp_data.read(1)

                                # Experimental
                                # 4bpp Alpha-channel to 8bpp
                                # Author: Hintay <hintay@me.com>
                                temp_a, = unpack('B', self.mzp_data.read(1))
                                a = (temp_a << 1) + (temp_a >> 6) if (temp_a < 0x80) else 255
                                a = pack('B', a)

                                self.paletteblob += (b + g + r + a)
                                self.palettepng += (r + g + b)
                                self.transpng += a
            else:
                log_error("Unsupported palette type 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)

            # 补全索引
            for i in range(self.palette_count, 0x100):
                self.paletteblob += b'\x00\x00\x00\xFF'
                self.palettepng += b'\x00\x00\x00'
                self.transpng += b'\xFF'
        elif self.bmp_type == 0x08:
            if self.bmp_depth == 0x14:
                self.bitmap_bpp = 24
            else:
                log_error("Unknown depth 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)
        elif self.bmp_type == 0x0B:
            if self.bmp_depth == 0x14:
                self.bitmap_bpp = 32
            else:
                log_error("Unknown depth 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)
        elif self.bmp_type == 0x03:  # 'PEH' 8bpp + palette
            log_error("Unsupported type 0x{:02X} (PEH)".format(self.bmp_type))
            sys.exit(EXIT_WITH_ERROR)

        del self.entries_descriptors[0]

    def debug_format(self):
        log_info(
            'MZP Format: Width = %s, Height = %s, Bitmap type = %s, Bitmap depth = %s, Bits per pixel = %s, '
            'Bytes Pre pixel = %s' % (
                self.width, self.height, self.bmp_type, self.bmp_depth, self.bitmap_bpp, self.bytesprepx))
        log_info('Tile Format: Width = %s, Height = %s, X count = %s, Y count = %s, Tile crop = %s' % (
            self.tile_width, self.tile_height, self.tile_x_count, self.tile_y_count, self.tile_crop))
        if self.tile_crop:
            width = self.width - self.tile_x_count * self.tile_crop * 2
            height = self.height - self.tile_y_count * self.tile_crop * 2
            log_info('MZP Cropped Size: Width = %s, Height = %s' % (width, height))

    def extract_tile(self, index):
        entry = self.entries_descriptors[index]
        self.mzp_data.seek(entry.real_offset)
        head_data = io.BytesIO(self.mzp_data.read(0x8))
        # log_info("tile head: {0}".format(head_data.read(0x8)))
        sig, size = unpack('<LL', head_data.getvalue())
        tile_data = io.BytesIO(self.mzp_data.read(size))

        if OPEN_DEBUG_FILE:
            debug_file_name = "{0}_{1}.tile.mzx0".format(self.in_mzp.name, index)
            with open(self.in_mzp.with_name(debug_file_name), 'wb') as debug_file:
                debug_file.write(head_data.getvalue())
                debug_file.write(tile_data.getvalue())
            log_info("Open debug file, writing {0} file succeed.".format(debug_file_name))

        dec_buf = mzx_decompress(src=tile_data)

        dec_buf = dec_buf.read()
        if self.bitmap_bpp == 4:
            tile_data = b''
            for octet in dec_buf:
                the_byte = Byte(octet)
                tile_data += pack('BB', the_byte.low, the_byte.high)
            dec_buf = tile_data

        # RGB/RGBA true color for 0x08 and 0x0B bmp type
        elif self.bitmap_bpp in [24, 32] and self.bmp_type in [0x08, 0x0B]:
            # 16bpp
            tile_data = b''
            for index in range(self.tile_size):
                P = dec_buf[index * 2]
                Q = dec_buf[(index * 2) + 1]
                b = (P & 0x1f) << 3
                g = (Q & 0x07) << 5 | (P & 0xe0) >> 3
                r = (Q & 0xf8)

                # Offset for 16bpp to 24bpp
                offset_byte = dec_buf[self.tile_size * 2 + index]
                r_offset = offset_byte >> 5
                g_offset = (offset_byte & 0x1f) >> 3
                b_offset = offset_byte & 0x7

                # Alpha
                if self.bitmap_bpp == 32:
                    a = dec_buf[self.tile_size * 3 + index]
                    tile_data += pack('BBBB', r + r_offset, g + g_offset, b + b_offset, a)
                else:
                    tile_data += pack('BBB', r + r_offset, g + g_offset, b + b_offset)
            dec_buf = tile_data
        return dec_buf

    def loop_data(self):
        for y in range(self.tile_y_count):
            start_row = y * (self.tile_height - self.tile_crop * 2)  # 上下切边
            rowcount = min(self.height, start_row + self.tile_height) - start_row - self.tile_crop * 2  # 共几行
            self.loop_x(y, start_row, rowcount)

    def loop_x(self, y, start_row, rowcount):
        # Tile 块处理
        for x in range(self.tile_x_count):
            dec_buf = self.extract_tile(self.tile_x_count * y + x)

            for i, tile_row_px in enumerate(chunks(dec_buf, self.tile_width * self.bytesprepx)):
                if i < self.tile_crop:
                    continue
                if (i - self.tile_crop) >= rowcount:
                    break
                cur_width = len(self.rows[start_row + i - self.tile_crop])
                px_count = min(self.width, cur_width + self.tile_width) * self.bytesprepx - cur_width
                try:
                    temp_row = tile_row_px[:px_count]
                    self.rows[start_row + i - self.tile_crop] += temp_row[self.tile_crop * self.bytesprepx: len(
                        temp_row) - self.tile_crop * self.bytesprepx]
                except IndexError:
                    log_error(start_row + i - self.tile_crop)

    def extract_to_png(self):
        output_target_dir = self.in_mzp.with_name(output_dir)
        # if output_target_dir.exists():
        #     log_warn("Removing existing directory: {0}".format(output_target_dir))
        #     shutil.rmtree(output_target_dir)

        if not output_target_dir.exists():
            output_target_dir.mkdir(parents=True)

        self.loop_data()

        png_file_name = self.in_mzp.with_suffix(".png").name
        png_path = output_target_dir / png_file_name
        with png_path.open('wb') as png:
            write_pngsig(png)
            width = self.width - self.tile_x_count * self.tile_crop * 2
            height = self.height - self.tile_y_count * self.tile_crop * 2
            if is_indexed_bitmap(self.bmp_type):
                write_ihdr(png, width, height, 8, 3)  # 8bpp (PLTE)
                write_plte(png, self.palettepng)
                write_trns(png, self.transpng)

            elif self.bitmap_bpp == 24:  # RGB True-color
                write_ihdr(png, width, height, 8, 2)  # 24bpp

            elif self.bitmap_bpp == 32:  # RGBA True-color
                write_ihdr(png, width, height, 8, 6)  # 32bpp

            # split into rows and add png filtering info (mandatory even with no filter)
            row_data = b''
            for row in self.rows:
                row_data += b'\x00' + row

            write_idat(png, row_data)
            write_iend(png)

        self.dump_to_json(
            png_path.with_suffix(".json"),
        )
        self.dump_act_from_png(png_path)

        log_succeed("Extracted {0} to {1} succeed.".format(self.in_mzp.name, png_file_name))

    def _png_tile_to_mzp_bytes(self, tile_img: Image.Image) -> bytes:
        tile_img = tile_img.convert("P")

        # create tile size img, copy palette
        padded_img = Image.new("P", (self.tile_width, self.tile_height), color=0)
        padded_img.putpalette(tile_img.getpalette())

        # paste left top
        crop_width, crop_height = tile_img.size
        region = tile_img.crop((0, 0, crop_width, crop_height))
        padded_img.paste(region, (0, 0))

        # save test img (optional)
        # padded_img.save("debug_padded.png")

        raw_data = padded_img.tobytes()
        padded_img.close()

        if self.bitmap_bpp == 8:
            return raw_data
        elif self.bitmap_bpp == 4:
            res = bytearray()
            for i in range(0, len(raw_data), 2):
                low = raw_data[i] & 0x0F
                high = raw_data[i + 1] & 0x0F if i + 1 < len(raw_data) else 0
                res.append((high << 4) | low)
            return bytes(res)

        raise Exception("Not supported bitmap bpp => {0}".format(self.bitmap_bpp))

    def repack_to_mzp(self):
        img = Image.open(self.in_png)

        # Compare input img size
        expected_width = self.width - self.tile_x_count * self.tile_crop * 2
        expected_height = self.height - self.tile_y_count * self.tile_crop * 2
        if img.width != expected_width or img.height != expected_height:
            raise ValueError(
                f"PNG size changed, expected ({expected_width}, {expected_height}), now ({img.width}, {img.height})"
            )

        # Calc tile size
        tile_full_w = self.tile_width
        tile_full_h = self.tile_height

        out_mzp_file = self.out_mzp.open('w+b')

        # copy origin head data to new mzp file
        out_mzp_file.write(self.origin_head_data_hex)

        # 用于记录每个 tile 的新的 entry 描述信息
        new_entry_descriptors = []

        # Compress per tile data
        for tile_idx in range(self.entry_count - 1):
            # Calc tile in png position
            tile_x = tile_idx % self.tile_x_count
            tile_y = tile_idx // self.tile_x_count

            # clip tile boundary
            x0 = tile_x * (tile_full_w - 2 * self.tile_crop)
            y0 = tile_y * (tile_full_h - 2 * self.tile_crop)

            # include tile crop boundary
            x_start = max(0, x0 - self.tile_crop)
            y_start = max(0, y0 - self.tile_crop)
            x_end = x_start + tile_full_w
            y_end = y_start + tile_full_h

            # 修正裁剪边界不超过原图宽高
            # x_end = min(x_end, img.width)
            # y_end = min(y_end, img.height)

            log_info(f"Start tile{tile_idx}, start_x: {x_start}, start_y: {y_start}, end_x: {x_end}, end_y: {y_end}")
            # Crop tile pixel
            tile_img = img.crop((x_start, y_start, x_end, y_end))
            img.close()

            # Convert mzp need bytes
            tile_bytes = self._png_tile_to_mzp_bytes(tile_img)

            tile_buf = io.BytesIO(tile_bytes)
            compressed_tile = mzx_compress(tile_buf)
            # log_info("tile head: {0}".format(compressed_tile.read(0x8)))
            out_mzp_file.write(compressed_tile.getvalue())

            padding_size = padding_bytes_needed(out_mzp_file.tell())

            # 实际的 offset 需要用当前的 offset - (头 size + entry count * sector_size)
            data_offset = out_mzp_file.tell() - (6 + 2 + self.entry_count * 8)
            compressed_len = len(compressed_tile.getvalue()) + padding_size

            # 这里需要用当前的 offset 减回压缩后的数据大小 + 补位的大小
            sector_offset, offset_in_sector, upper_bound_sector = calculate_entry_descriptor(
                data_offset - compressed_len + padding_size,
                compressed_len
            )

            out_mzp_file.write(padding_size * END_PADDING_DATA)

            index = tile_idx + 1
            desc: ArchiveEntry = ArchiveEntry(
                sector_offset, offset_in_sector,
                upper_bound_sector, compressed_len, self.entry_count
            )
            new_entry_descriptors.append(desc)
            log_info(f"Tile{tile_idx} compress successfully.")

            if OPEN_DEBUG_FILE:
                out_mzp_file.seek(desc.real_offset)
                tile_debug_file = self.out_mzp.with_name("{0}_{1}.tile".format(self.out_mzp.name, index))
                with open(tile_debug_file, 'wb') as tile_file:
                    tile_file.write(out_mzp_file.read(0x8))
                    tile_file.write(out_mzp_file.read(compressed_len))
                log_info("Save debug file => {0}".format(tile_debug_file.name))

        # Seek entry0 head info, override new entry info
        out_mzp_file.seek(8 + 8 * 1)  # 跳过 header 和 entry0 描述信息
        for desc in new_entry_descriptors:
            out_mzp_file.write(
                pack(
                    "<4H",
                    desc.sector_offset, desc.offset,
                    desc.sector_size_upper_boundary, desc.size
                )
            )

        out_mzp_file.close()
        log_succeed(f"Successfully re-packed archive file => {out_mzp_file.name}")


def do_unpack(input_args):
    mzp_entry = MzpEntry(in_mzp=input_args.input)
    mzp_entry.extract_to_png()


def do_repack(input_args):
    mzp_entry = MzpEntry(in_png=input_args.input, out_mzp=input_args.output)
    mzp_entry.repack_to_mzp()


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
