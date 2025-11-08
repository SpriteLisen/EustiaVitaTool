import io
import re
from Constants import *
from pathlib import Path
from struct import unpack, pack
from mzx_tool import mzx_decompress

scr_output_dir = "mzx-scr"
mzx_output_dir = "scr-mzx"

open_origin_file = False

match_command = [
    "_LVSV",
    "_STTI",
    "_ZM",
    "_MSAD",
    "_MSA\d",
    "SELR"
]


def parse_args():
    parser_instance = argparse.ArgumentParser(description=__doc__, add_help=False)
    subparsers = parser_instance.add_subparsers(title='subcommands', dest='subcommand')

    parser_unpack = subparsers.add_parser('unpack', help='unpack mzx file to scr file')
    parser_unpack.add_argument('input', metavar='input.mzx', type=Path, help='Input .mzx file or dir [REQUIRED]')

    parser_repack = subparsers.add_parser('repack', help='generate a mzx file from an .scr file')
    parser_repack.add_argument('input', metavar='./input.*', type=Path, help='Input .scr file or dir[REQUIRED]')

    parser_instance.add_argument('-h', '--help',
                                 action=HelpAction, default=argparse.SUPPRESS,
                                 help='show this help message and exit')

    return parser_instance, parser_instance.parse_args()


def mzx0_compress(f, inlen, xorff=False):
    """Compress a block of data.
    """
    dout = bytearray(b'MZX0')
    dout.extend(pack('<L', inlen))

    key32 = 0xFFFFFFFF if xorff else 0
    while inlen >= 0x80:
        dout.append(0xFF)
        for _ in range(0x20):
            chunk = f.read(4)
            if len(chunk) < 4:
                raise EOFError("Unexpected end of file while reading 128-byte block")
            dout.extend(pack('<L', unpack('<L', chunk)[0] ^ key32))
        inlen -= 0x80

    key8 = 0xFF if xorff else 0
    if inlen >= 2:
        dout.append(((inlen >> 1) - 1) * 4 + 3)
        for _ in range((inlen >> 1) * 2):
            b = f.read(1)
            if not b:
                raise EOFError("Unexpected end of file while reading remainder")
            dout.append(b[0] ^ key8)
        inlen -= (inlen >> 1) * 2

    if inlen == 1:
        dout.append(0x03)
        b = f.read(1)
        if not b:
            raise EOFError("Unexpected end of file while reading last byte")
        dout.append(b[0] ^ key8)
        dout.append(0x00)

    return bytes(dout)


class ScrEntry:
    def __init__(self, in_mzx: Path = None, in_scr: Path = None):
        if in_mzx is not None:
            self.in_mzx = in_mzx
            mzx_file = in_mzx.open("rb")
            data = mzx_file.read()
            mzx_file.close()

            suffix = detect_file_extension_with_bytes(data)
            assert suffix.lower() == SUFFIX_MZX, "Target file {0} is not mzx file!".format(in_mzx.name)

            self.mzx_data = io.BytesIO(data)
        elif in_scr is not None:
            self.in_scr = in_scr

            scr_file = in_scr.open("rt", encoding=MZX_ENCODING, errors="surrogateescape")
            self.scr_lines = scr_file.readlines()
            scr_file.close()
        else:
            raise ValueError("Must init a input file.")

    def extract_to_scr(self):
        log_info("Start extracting {0} to scr...".format(self.in_mzx.name))
        head_data = io.BytesIO(self.mzx_data.read(0x8))
        # log_info("tile head: {0}".format(head_data.read(0x8)))
        sig, size = unpack('<LL', head_data.getvalue())
        content_data = io.BytesIO(self.mzx_data.read(self.in_mzx.stat().st_size))

        dec_buf = mzx_decompress(src=content_data, invert_bytes=True)
        dec_buf.truncate(size)

        origin_data = []
        out_text = []
        for index, instr in enumerate(dec_buf.read().split(b';')):
            instr_text = instr.decode(MZX_ENCODING, errors="surrogateescape")

            origin_data.append(instr_text)

            if re.search(rf'{"|".join(match_command)}', instr_text) is not None:
                out_text.append(
                    instr_text.replace("^", "_r")
                        .replace("@n", "_n")
                        .replace(",", ";/")
                )
            else:
                out_text.append(instr_text)

        if out_text:
            scr_out_path = self.in_mzx.with_name(scr_output_dir)
            scr_out_path.mkdir(parents=True, exist_ok=True)

            out_scr_file_path = scr_out_path / f"{self.in_mzx.stem}{SUFFIX_SCR}"
            if out_scr_file_path.exists():
                log_warn("Remove exists scr file {0}".format(out_scr_file_path.name))
                out_scr_file_path.unlink()

            out_origin_file_path = scr_out_path / f"{self.in_mzx.stem}{SUFFIX_BIN}"

            if open_origin_file:
                with out_origin_file_path.open('wt', encoding=MZX_ENCODING, errors="surrogateescape") as origin_file:
                    origin_file.write(";".join(origin_data))

            with out_scr_file_path.open('wt', encoding=MZX_ENCODING, errors="surrogateescape") as scr_file:
                scr_file.write('\n'.join(out_text))

            log_succeed(f"Successfully extracted scr file => {out_scr_file_path.name}")
        else:
            log_error(f"Failed to extract scr file of => {self.in_mzx.name}")

    def repack_to_mzx(self):
        log_info("Start repacking {0} to mxz...".format(self.in_scr.name))

        processed_lines = []

        line_num = 1
        for line in self.scr_lines:
            line = line.rstrip('\r\n')

            # 看是否包含我们需要的指令
            m = re.search(rf'{"|".join(match_command)}', line)
            if m is not None:
                line = line.replace(
                    ";/", ","
                ).replace(
                    "_n", "@n"
                ).replace(
                    "_r", "^"
                )

                processed_lines.append(line)
            else:
                processed_lines.append(line)
            line_num += 1

        result_bytes = ';'.join(processed_lines).encode(MZX_ENCODING, errors="surrogateescape")

        mzx_out_path = self.in_scr.with_name(mzx_output_dir)
        mzx_out_path.mkdir(parents=True, exist_ok=True)

        # out_origin_file_path = mzx_out_path / f"{self.in_scr.stem}{SUFFIX_BIN}"
        # with out_origin_file_path.open('wb') as out_origin_file:
        #     out_origin_file.write(result_bytes)

        out_mzx_file_path = mzx_out_path / f"{self.in_scr.stem}{SUFFIX_MZX}"
        if out_mzx_file_path.exists():
            log_warn("Remove exists mzx file {0}".format(out_mzx_file_path.name))
            out_mzx_file_path.unlink()

        mzx_data = mzx0_compress(io.BytesIO(result_bytes), len(result_bytes), xorff=True)
        with open(out_mzx_file_path, 'wb') as outfile:
            outfile.write(mzx_data)

        log_succeed(f"Successfully repacked mxz file => {out_mzx_file_path.name}")


def do_unpack(input_args):
    input_path: Path = input_args.input

    if input_path.is_dir():
        for mzx_file_path in input_path.glob('*.[Mm][Zz][Xx]'):
            scr_entry = ScrEntry(in_mzx=mzx_file_path)
            scr_entry.extract_to_scr()
    else:
        scr_entry = ScrEntry(in_mzx=input_path)
        scr_entry.extract_to_scr()


def do_repack(input_args):
    input_path: Path = input_args.input

    if input_path.is_dir():
        for scr_file_path in input_path.glob('*.[Ss][Cc][Rr]'):
            scr_entry = ScrEntry(in_scr=scr_file_path)
            scr_entry.repack_to_mzx()
    else:
        scr_entry = ScrEntry(in_scr=input_path)
        scr_entry.repack_to_mzx()


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
