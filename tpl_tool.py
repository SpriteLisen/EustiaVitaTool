import io
import re
from Constants import *
from pathlib import Path
from struct import unpack, pack
from mzx_tool import mzx_decompress

tpl_output_dir = "mzx-tpl"
mzx_output_dir = "tpl-mzx"


def parse_args():
    parser_instance = argparse.ArgumentParser(description=__doc__, add_help=False)
    subparsers = parser_instance.add_subparsers(title='subcommands', dest='subcommand')

    parser_unpack = subparsers.add_parser('unpack', help='unpack mzx file to tpl file')
    parser_unpack.add_argument('input', metavar='input.mzx', type=Path, help='Input .mzx file or dir [REQUIRED]')

    parser_repack = subparsers.add_parser('repack', help='generate a mzx file from an .tpl file')
    parser_repack.add_argument('input', metavar='./input.*', type=Path, help='Input .tpl file or dir[REQUIRED]')

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


class TplEntry:
    def __init__(self, in_mzx: Path = None, in_tpl: Path = None):
        if in_mzx is not None:
            self.in_mzx = in_mzx
            mzx_file = in_mzx.open("rb")
            data = mzx_file.read()
            mzx_file.close()

            suffix = detect_file_extension_with_bytes(data)
            assert suffix.lower() == SUFFIX_MZX, "Target file {0} is not mzx file!".format(in_mzx.name)

            self.mzx_data = io.BytesIO(data)
        elif in_tpl is not None:
            self.in_tpl = in_tpl

            tpl_file = in_tpl.open("rt", encoding=MZX_ENCODING)
            self.tpl_lines = tpl_file.readlines()
            tpl_file.close()
        else:
            raise ValueError("Must init a input file.")

    def extract_to_tpl(self):
        log_info("Start extracting {0} to tpl...".format(self.in_mzx.name))
        head_data = io.BytesIO(self.mzx_data.read(0x8))
        # log_info("tile head: {0}".format(head_data.read(0x8)))
        sig, size = unpack('<LL', head_data.getvalue())
        content_data = io.BytesIO(self.mzx_data.read(self.in_mzx.stat().st_size))

        dec_buf = mzx_decompress(src=content_data, invert_bytes=True)
        dec_buf.truncate(size)

        origin_data = []
        out_text = []
        for index, instr in enumerate(dec_buf.read().split(b';')):
            instr_text = instr.decode(MZX_ENCODING)
            # out_text.append(instr_text)
            origin_data.append(instr_text)
            if re.search(r'_LVSV|_STTI|_MSAD|_ZM|SEL[R]', instr_text) is not None:
                out_text.append(
                    "<{0:04d}>".format(index) + instr_text.replace("^", "_r")
                    .replace("@n", "_n").replace(",", ";/")
                )  # replace order significant
            elif len(re.sub('[ -~]', '', instr_text)) > 0:
                out_text.append(u"!" + instr_text)  # flag missing matches containing non-ASCII characters
            else:
                out_text.append(u"~" + instr_text + u"~")  # non-localizable

        if out_text:
            tpl_out_path = self.in_mzx.with_name(tpl_output_dir)
            tpl_out_path.mkdir(parents=True, exist_ok=True)

            out_tpl_file_path = tpl_out_path / f"{self.in_mzx.stem}{SUFFIX_TPL}"
            if out_tpl_file_path.exists():
                log_warn("Remove exists tpl file {0}".format(out_tpl_file_path.name))
                out_tpl_file_path.unlink()

            out_origin_file_path = tpl_out_path / f"{self.in_mzx.stem}{SUFFIX_BIN}"

            with out_origin_file_path.open('wt', encoding=MZX_ENCODING) as origin_file:
                origin_file.write(";".join(origin_data))

            with out_tpl_file_path.open('wt', encoding=MZX_ENCODING) as tpl_file:
                tpl_file.write('\n'.join(out_text))

            log_succeed(f"Successfully extracted tpl file => {out_tpl_file_path.name}")
        else:
            log_error(f"Failed to extract tpl file of => {self.in_mzx.name}")

    def repack_to_mzx(self):
        log_info("Start repacking {0} to mxz...".format(self.in_tpl.name))

        processed_lines = []

        line_num = 1
        for line in self.tpl_lines:
            line = line.rstrip('\r\n')
            m = re.search(r'^<[0-9]+>(.+)', line)
            if m is not None:
                parts = m.group(1).split('=', 2)
                line = parts[0] if len(parts) < 2 else parts[1]
                expr = re.search(r'^([^(]+)\((.+)\)', line)

                if expr is not None:
                    before = expr.group(1) + "("
                    subj = expr.group(2)
                    after = ")"
                else:
                    before = after = ""
                    subj = line

                line = before + subj.replace(
                    ";/", ","
                ).replace(
                    "_n", "@n"
                ).replace(
                    "_r", "^"
                ) + after

                processed_lines.append(line)
            elif len(line) > 1 and line[0] == '!':
                processed_lines.append(line[1:])
            else:
                if len(line) > 0:
                    m = re.search(r'^~(.*)~$', line)
                    if m is None:
                        log_warn(
                            f"{self.in_tpl.name} line {line_num} - text should be enclosed in ~~ {line}"
                        )
                    else:
                        processed_lines.append(m.group(1))
            line_num += 1

        result_bytes = ';'.join(processed_lines).encode(MZX_ENCODING)

        mzx_out_path = self.in_tpl.with_name(mzx_output_dir)
        mzx_out_path.mkdir(parents=True, exist_ok=True)

        # out_origin_file_path = mzx_out_path / f"{self.in_tpl.stem}{SUFFIX_BIN}"
        # with out_origin_file_path.open('wb') as out_origin_file:
        #     out_origin_file.write(result_bytes)

        out_mzx_file_path = mzx_out_path / f"{self.in_tpl.stem}{SUFFIX_MZX}"
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
            tpl_entry = TplEntry(in_mzx=mzx_file_path)
            tpl_entry.extract_to_tpl()
    else:
        tpl_entry = TplEntry(in_mzx=input_path)
        tpl_entry.extract_to_tpl()


def do_repack(input_args):
    input_path: Path = input_args.input

    if input_path.is_dir():
        for tpl_file_path in input_path.glob('*.[Tt][Pp][Ll]'):
            tpl_entry = TplEntry(in_tpl=tpl_file_path)
            tpl_entry.repack_to_mzx()
    else:
        tpl_entry = TplEntry(in_tpl=input_path)
        tpl_entry.repack_to_mzx()


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
