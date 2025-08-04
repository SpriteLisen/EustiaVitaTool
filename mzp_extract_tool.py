import io
import zlib
import argparse
from Constants import *
from typing import Tuple
from pathlib import Path
from struct import unpack, pack

output_dir = "mzp-unpacked"


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


###############################################
# struct TGAHeader
# {
#   uint8   idLength,           // Length of optional identification sequence.
#           paletteType,        // Is a palette present? (1=yes)
#           imageType;          // Image data type (0=none, 1=indexed, 2=rgb,
#                               // 3=grey, +8=rle packed).
#   uint16  firstPaletteEntry,  // First palette index, if present.
#           numPaletteEntries;  // Number of palette entries, if present.
#   uint8   paletteBits;        // Number of bits per palette entry.
#   uint16  x,                  // Horiz. pixel coord. of lower left of image.
#           y,                  // Vert. pixel coord. of lower left of image.
#           width,              // Image width in pixels.
#           height;             // Image height in pixels.
#   uint8   depth,              // Image color depth (bits per pixel).
#           descriptor;         // Image attribute flags.
# };

def is_indexed_bitmap(bmp_info):
    return bmp_info == 0x01


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def write_pngsig(f):
    f.write(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A')


def write_pngchunk_withcrc(f, t, data):
    f.write(pack(">I", len(data)))
    f.write(t)
    f.write(data)
    f.write(pack(">I", zlib.crc32(t + data, 0) & 0xffffffff))


"""
    color = 1 (palette used), 2 (color used), and 4 (alpha channel used). Valid values are 0, 2, 3, 4, and 6. 

    Color    Allowed    Interpretation
    Type    Bit Depths

    0       1,2,4,8,16  Each pixel is a grayscale sample.

    2       8,16        Each pixel is an R,G,B triple.

    3       1,2,4,8     Each pixel is a palette index;
                       a PLTE chunk must appear.

    4       8,16        Each pixel is a grayscale sample,
                       followed by an alpha sample.

    6       8,16        Each pixel is an R,G,B triple,
                       followed by an alpha sample.
"""


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


def mzx0_decompress(f, inlen, exlen, xorff=False) -> Tuple[str, io.BytesIO]:
    """
    Decompress a block of data.
    """
    key = 0xFF

    out_data = io.BytesIO()  # slightly overprovision for writes past end of buffer
    ring_buf = [b'\xFF\xFF'] * 64 if xorff else [b'\x00\x00'] * 64
    ring_wpos = 0

    clear_count = 0
    max_len = f.tell() + inlen
    last = b'\xFF\xFF' if xorff else b'\x00\x00'

    while out_data.tell() < exlen:
        if f.tell() >= max_len:
            break
        if clear_count <= 0:
            clear_count = 0x1000
            last = b'\xFF\xFF' if xorff else b'\x00\x00'
        flags = ord(f.read(1))
        # print("+ %X %X %X" % (flags, f.tell(), out_data.tell()))

        clear_count -= 1 if (flags & 0x03) == 2 else flags // 4 + 1

        if flags & 0x03 == 0:
            out_data.write(last * ((flags // 4) + 1))

        elif flags & 0x03 == 1:
            k = 2 * (ord(f.read(1)) + 1)
            for i in range(flags // 4 + 1):
                out_data.seek(-k, 1)
                last = out_data.read(2)
                out_data.seek(0, 2)
                out_data.write(last)

        elif flags & 0x03 == 2:
            last = ring_buf[flags // 4]
            out_data.write(last)

        else:
            for i in range(flags // 4 + 1):
                last = ring_buf[ring_wpos] = bytes([byte ^ key for byte in f.read(2)]) if xorff else f.read(2)
                out_data.write(last)

                ring_wpos += 1
                ring_wpos %= 64
    status = "OK"

    out_data.truncate(exlen)  # Resize stream to decompress size
    out_data.seek(0)
    return status, out_data


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
    def __init__(self, in_mzp: Path):
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
        sig, size = unpack('<LL', self.mzp_data.read(0x8))
        status, dec_buf = mzx0_decompress(self.mzp_data, entry.real_size - 8, size)
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

        log_succeed("Extracted {0} to {1} succeed.".format(self.in_mzp.name, png_file_name))


def do_unpack(input_args):
    mzp_entry = MzpEntry(in_mzp=input_args.input)
    mzp_entry.extract_to_png()


if __name__ == '__main__':
    parser, args = parse_args()
    if args.subcommand == "unpack":
        do_unpack(args)
    elif args.subcommand == "repack":
        # do_repack(args)
        pass
    else:
        parser.print_usage()
        sys.exit(EXIT_WITH_HELP)
    sys.exit(0)
