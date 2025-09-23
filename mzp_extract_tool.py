import io
import json
import os.path
import zlib
from pathlib import Path
from struct import unpack, pack

import numpy as np
from PIL import Image

from Constants import *
from mzx_tool import mzx_compress, mzx_decompress

output_dir = "mzp-unpacked"
output_mzp_dir = "mzp-repacked"

OPEN_DEBUG_FILE = False


def parse_args():
    parser_instance = argparse.ArgumentParser(description=__doc__, add_help=False)
    subparsers = parser_instance.add_subparsers(title='subcommands', dest='subcommand')

    parser_unpack = subparsers.add_parser('unpack',
                                          help='unpack mzp file')
    parser_unpack.add_argument('input', metavar='input.mzp', type=Path, help='Input .mzp file or dir [REQUIRED]')

    parser_repack = subparsers.add_parser('repack', help='generate a mzp file from an existing file')
    parser_repack.add_argument('input', metavar='./input.*', type=Path, help='Input .png file [REQUIRED]')
    parser_repack.add_argument('output', type=Path, nargs='?', help='Output .mzp file')

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


class Byte(object):
    def __init__(self, number):
        self.number = number

    @property
    def high(self):
        return self.number >> 4

    @property
    def low(self):
        return self.number & 0x0F


# 更精确的逆向算法
def alpha_8bpp_to_4bpp_precise(a_8bpp):
    """
    更精确的逆向转换
    """
    if a_8bpp >= 254:  # 接近255的值
        return 0x80

    # 寻找最接近的4bpp值
    best_match = 0
    min_diff = 256

    for test_4bpp in range(0, 128):  # 0-127
        # 应用正向算法
        test_8bpp = (test_4bpp << 1) + (test_4bpp >> 6)
        diff = abs(a_8bpp - test_8bpp)

        if diff < min_diff:
            min_diff = diff
            best_match = test_4bpp

    return best_match


def get_rgba_palette(img: Image.Image, palette_count: int):
    """
    从不同格式的图转换回调色板 rgba 信息
    """
    # 取出原始色板
    if img.mode == "P":
        # 获取调色板信息
        palette = img.getpalette()

        # 检查是否有透明度信息
        transparency = img.info.get('transparency')

        # 将调色板转换为 RGBA 格式
        rgba_palette = []
        for i in range(0, min(len(palette), palette_count * 3), 3):
            r, g, b = palette[i], palette[i + 1], palette[i + 2]
            alpha = 255  # 默认 alpha

            # 如果有透明度信息，检查当前索引是否在透明度表中
            if transparency is not None:
                if isinstance(transparency, dict):
                    # 透明度是字典形式 {index: alpha}
                    alpha = transparency.get(i // 3, 255)
                elif isinstance(transparency, (bytes, tuple, list)):
                    # 透明度是字节序列或列表形式
                    if i // 3 < len(transparency):
                        alpha = transparency[i // 3] if isinstance(transparency[i // 3], int) else 255
                elif isinstance(transparency, int):
                    # 单个透明度值（通常表示透明色的索引）
                    alpha = 0 if i // 3 == transparency else 255

            # 重新解包时是平滑的, 因为解包时做了转换, 所以重新封回时也需要进行逆转换
            # 配合游戏引擎中奇怪的 alpha 换算
            alpha = alpha_8bpp_to_4bpp_precise(alpha)
            rgba_palette.append((r, g, b, alpha))

        # 用 (0,0,0,0) 补全到 palette_count
        if len(rgba_palette) < palette_count:
            rgba_palette += [(0, 0, 0, 0)] * (palette_count - len(rgba_palette))

        return img, rgba_palette

    # 取出 RGB 调色板
    elif img.mode == "RGBA":
        p_img = img.convert("P")
        rgba_palette = [None] * len(p_img.palette.colors)
        for rgba, idx in p_img.palette.colors.items():
            # rgba_palette[idx] = rgba
            r = rgba[0]
            g = rgba[1]
            b = rgba[2]

            # 重新解包时是平滑的, 因为解包时做了转换, 所以重新封回时也需要进行逆转换
            # 配合游戏引擎中奇怪的 alpha 换算
            a = alpha_8bpp_to_4bpp_precise(rgba[3])

            rgba_palette[idx] = (r, g, b, a)

        # 用 (0,0,0,0) 补全到 palette_count
        if len(rgba_palette) < palette_count:
            rgba_palette += [(0, 0, 0, 0)] * (palette_count - len(rgba_palette))

        return p_img, rgba_palette
    else:
        raise Exception(f"Unsupported image mode: {img.mode}")


class MzpEntry:
    KEY_ENTRY_0_START_OFFSET = "entry_0_start_offset"
    KEY_ENTRY_COUNT = "entry_count"
    KEY_BITMAP_BPP = "bitmap_bpp"
    KEY_BMP_TYPE = "bmp_type"
    KEY_BMP_DEPTH = "bmp_depth"
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
            self.entry_0_start_offset = 0
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
            self.entry_0_start_offset = json_data[MzpEntry.KEY_ENTRY_0_START_OFFSET]
            self.entry_count = json_data[MzpEntry.KEY_ENTRY_COUNT]
            self.bitmap_bpp = json_data[MzpEntry.KEY_BITMAP_BPP]
            self.bmp_type = json_data[MzpEntry.KEY_BMP_TYPE]
            self.bmp_depth = json_data[MzpEntry.KEY_BMP_DEPTH]
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
                    MzpEntry.KEY_ENTRY_0_START_OFFSET: self.entry_0_start_offset,
                    MzpEntry.KEY_ENTRY_COUNT: self.entry_count,
                    MzpEntry.KEY_BITMAP_BPP: self.bitmap_bpp,
                    MzpEntry.KEY_BMP_TYPE: self.bmp_type,
                    MzpEntry.KEY_BMP_DEPTH: self.bmp_depth,
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

    def extract_desc(self):
        self.mzp_data.seek(self.entries_descriptors[0].real_offset)
        self.width, self.height, self.tile_width, self.tile_height, self.tile_x_count, self.tile_y_count, \
            self.bmp_type, self.bmp_depth, self.tile_crop = unpack('<HHHHHHHBB', self.mzp_data.read(0x10))

        # 记录 entry0 当前的位置
        self.entry_0_start_offset = self.mzp_data.tell()

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
        # sig, size = unpack('<LL', head_data.getvalue())
        # tile_data = io.BytesIO(self.mzp_data.read(size))
        # 减去头部的大小就是正确的 size
        tile_data = io.BytesIO(self.mzp_data.read(entry.real_size - 8))

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

        png_file_name = self.in_mzp.with_suffix(".png").name
        png_path = output_target_dir / png_file_name

        if os.path.isfile(png_path):
            log_warn(f"{png_path} exist, ignore.")
            return

        self.loop_data()

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

        log_succeed("Extracted {0} to {1} succeed.".format(self.in_mzp.name, png_file_name))

    def _png_tile_to_mzp_bytes(self, tile_img: Image.Image) -> bytes:
        mode = tile_img.mode
        tile_size = self.tile_width * self.tile_height

        if mode == "P":
            raw_data = tile_img.tobytes()
            if self.bitmap_bpp == 8:
                return raw_data
            elif self.bitmap_bpp == 4:
                res = bytearray()
                for i in range(0, len(raw_data), 2):
                    low = raw_data[i] & 0x0F
                    high = raw_data[i + 1] & 0x0F if i + 1 < len(raw_data) else 0
                    res.append((high << 4) | low)
                return bytes(res)
            else:
                raise Exception(f"Unsupported bitmap_bpp: {self.bitmap_bpp}")

        elif mode in ("RGB", "RGBA"):
            # 不再进行BGR转换，直接使用原始RGB顺序
            if self.bitmap_bpp == 24 and mode == "RGB":
                data = np.frombuffer(tile_img.tobytes(), dtype=np.uint8).reshape(-1, 3)
            elif self.bitmap_bpp == 32 and mode == "RGBA":
                data = np.frombuffer(tile_img.tobytes(), dtype=np.uint8).reshape(-1, 4)
            else:
                raise Exception(f"Incompatible bitmap_bpp {self.bitmap_bpp} for mode {mode}")

            P_list = []
            Q_list = []
            offset_list = []
            alpha_list = [] if self.bitmap_bpp == 32 else None

            for i in range(tile_size):
                if self.bitmap_bpp == 24:
                    r, g, b = data[i]
                else:
                    r, g, b, a = data[i]

                # 计算基色值 (去除低位)
                r_base = (r >> 3) << 3  # 保留高5位
                g_base = (g >> 2) << 2  # 保留高6位 (关键修复)
                b_base = (b >> 3) << 3  # 保留高5位

                # 计算偏移量 (直接差值)
                r_offset = r - r_base
                g_offset = g - g_base
                b_offset = b - b_base

                # 提取g的位分量
                g_high3 = (g_base >> 5) & 0x07  # g的高3位 (位5-7)
                g_low3 = (g_base >> 2) & 0x07  # g的中间3位 (位2-4)

                # 构造P值: b低5位 | g低3位(高位)
                P_val = ((b_base >> 3) & 0x1F) | (g_low3 << 5)

                # 构造Q值: r高5位(高位) | g高3位(低位)
                Q_val = ((r_base >> 3) << 3) | g_high3

                P_list.append(P_val)
                Q_list.append(Q_val)

                # 打包偏移字节: r_offset(3位) | g_offset(2位) | b_offset(3位)
                offset_byte = (r_offset << 5) | ((g_offset & 0x03) << 3) | b_offset
                offset_list.append(offset_byte)

                if self.bitmap_bpp == 32:
                    # noinspection PyUnboundLocalVariable
                    alpha_list.append(a)

            result = bytearray()
            # 写入P/Q对
            for p, q in zip(P_list, Q_list):
                result.append(p)
                result.append(q)
            # 写入偏移量
            result.extend(offset_list)
            # 写入alpha通道 (32bpp)
            if self.bitmap_bpp == 32:
                result.extend(alpha_list)

            return bytes(result)

        else:
            raise Exception(f"Unsupported image mode: {mode}")

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

        head_end = out_mzp_file.tell()

        # 记录 entry0 当前的位置
        out_mzp_file.seek(self.entry_0_start_offset)

        self.tile_size = self.tile_width * self.tile_height
        if self.bmp_type not in [0x01, 0x03, 0x08, 0x0B]:
            log_error("Unknown type 0x{:02X}".format(self.bmp_type))
            sys.exit(EXIT_WITH_ERROR)

        # 索引模式则往回覆盖调色板信息
        if is_indexed_bitmap(self.bmp_type):
            if self.bmp_depth == 0x01:
                self.palette_count = 0x100
            elif self.bmp_depth == 0x00 or self.bmp_depth == 0x10:
                self.palette_count = 0x10
            elif self.bmp_depth == 0x11 or self.bmp_depth == 0x91:
                self.palette_count = 0x100
            else:
                log_error("Unknown depth 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)

            p_img, palette = get_rgba_palette(img, self.palette_count)
            img = p_img

            if self.bmp_depth in [0x00, 0x10]:
                for i in range(self.palette_count):
                    r, g, b, a = palette.pop(0)
                    out_mzp_file.write(pack("BBBB", r, g, b, a))

            elif self.bmp_depth in [0x11, 0x91, 0x01]:
                pal_start = self.entry_0_start_offset
                for h in range(0, self.palette_count * 4 // 0x80, 1):
                    for i in range(2):
                        for j in range(2):
                            out_mzp_file.seek(h * 0x80 + (i + j * 2) * 0x20 + pal_start)
                            for k in range(8):
                                r, g, b, a = palette.pop(0)
                                out_mzp_file.write(pack("BBBB", r, g, b, a))
            else:
                log_error("Unsupported palette type 0x{:02X}".format(self.bmp_depth))
                sys.exit(EXIT_WITH_ERROR)

        out_mzp_file.seek(head_end)

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

        img.close()

        # Seek entry0 head info, override new entry info
        out_mzp_file.seek(8 + 8 * 1)  # 跳过 header 和 entry0 描述信息
        for desc in new_entry_descriptors:
            fields = [
                ("sector_offset", desc.sector_offset),
                ("offset", desc.offset),
                ("sector_size_upper_boundary", desc.sector_size_upper_boundary),
                ("size", desc.size)
            ]
            for name, value in fields:
                if value > 0xFFFF:
                    log_error(f"字段 {name} 超出 0xFFFF: {value}")

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
    input_path: Path = input_args.input

    if input_path.is_dir():
        target_files = list(input_path.glob('*.[Mm][Zz][Pp]'))
        file_count = len(target_files)

        for index, mzp_file_path in enumerate(target_files):
            mzp_entry = MzpEntry(in_mzp=mzp_file_path)
            mzp_entry.extract_to_png()

            percentage = ((index + 1) / file_count) * 100
            log_prog(f"Progress: {percentage:.2f}%, Total: {file_count}, now: {index + 1}")
    else:
        mzp_entry = MzpEntry(in_mzp=input_args.input)
        mzp_entry.extract_to_png()


def do_repack(input_args):
    input_path: Path = input_args.input

    if input_path.is_dir():
        target_files = list(input_path.glob('*.[Pp][Nn][Gg]'))
        file_count = len(target_files)

        for index, png_file_path in enumerate(target_files):
            mzp_output_dir = input_args.output if input_args.output is not None \
                else input_args.input / output_mzp_dir
            mzp_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = mzp_output_dir / png_file_path.with_suffix(SUFFIX_MZP).name

            mzp_entry = MzpEntry(in_png=png_file_path, out_mzp=output_file)
            mzp_entry.repack_to_mzp()

            percentage = ((index + 1) / file_count) * 100
            log_prog(f"Progress: {percentage:.2f}%, Total: {file_count}, now: {index + 1}")
    else:
        mzp_output_dir = input_args.output if input_args.output is not None \
            else input_args.input / output_mzp_dir
        mzp_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = mzp_output_dir / input_args.input.with_suffix(SUFFIX_MZP).name

        mzp_entry = MzpEntry(in_png=input_args.input, out_mzp=output_file)
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
