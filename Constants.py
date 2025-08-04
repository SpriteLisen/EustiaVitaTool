import sys

assert sys.version_info >= (3, 7), "Python 3.7 or higher is required"

TAG_SUCCEED = "[SUCC]: "
TAG_INFO = "[INFO]: "
TAG_ERR = "[ERRO]: "
TAG_WARN = "[Warn]: "
TAG_SAME = "[SAME]: "
TAG_DIFF = "[DIFF]: "

RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLACK = "\033[90m"
GREEN = "\033[92m"


def log_succeed(msg):
    print(f"{GREEN}{TAG_SUCCEED} {msg}{RESET}")


def log_info(msg):
    # print(f"{BLACK}{TAG_INFO} {msg}{RESET}")
    print(f"{TAG_INFO} {msg}")


def log_warn(msg):
    print(f"{YELLOW}{TAG_WARN} {msg}{RESET}")


def log_error(msg):
    print(f"{RED}{TAG_ERR} {msg}{RESET}")


def log_same(msg):
    print(f"{GREEN}{TAG_SAME} {msg}{RESET}")


def log_diff(msg):
    print(f"{RED}{TAG_DIFF} {msg}{RESET}")


# 打印帮助信息
EXIT_WITH_HELP = 999
# 出错退出的 code
EXIT_WITH_ERROR = 888

# 默认编码
DEFAULT_ENCODING = "Shift_JIS"

# 默认的扇区大小
DEFAULT_SECTOR_SIZE = 0x800
# 默认的数据块大小
DEFAULT_BLOCK_SIZE = 0x8

# .nam 文件结尾的空数据长度, 默认是 32 个字节, 全部填充 FF
NAM_ENTRY_BLOCK_SIZE = 0x20

# .hed 文件结尾的空数据长度, 默认是 16 个字节, 全部填充 FF
HED_END_BLOCK_SIZE = 0x10

# 用来对齐结尾数据的字节
END_PADDING_DATA: bytes = b'\xFF'
# 用来对齐空数据的字节
EMPTY_PADDING_DATA: bytes = b"\x00"

LIST_FILE_NAME = "list.txt"

HEADER_FORMAT = "<HHHH"

# MRG 的文件头
MRG_MAGIC = b"mrgd00"

# MZX 的文件头
MZX_MAGIC = b"MZX0"

# NAM 的文件头
NAM_MAGIC = b"MRG.NAM"

SUFFIX_HED = ".hed"
SUFFIX_NAM = ".nam"
SUFFIX_MRG = ".mrg"
SUFFIX_MZX = ".mzx"
SUFFIX_MZP = ".mzp"
SUFFIX_BIN = ".bin"


def padding_bytes_needed(offset: int) -> int:
    # 目前发现会根据最后一位的 offset 进行补位, 总结规则为：
    # 刚好是 8 | 16 位, 则直接补 8 位
    # 小于 8 位补齐到 8 位
    # 大于 8 小于 16 位则补齐 16 位

    tail = offset % 16

    if tail == 8 or tail == 0:
        # offset 刚好在 8 or 16，要继续补 8 位
        return 8
    elif 0 < tail < 8:
        # tail in [1, 7]
        return 8 - tail
    else:
        # tail in [9, 15]
        return 16 - tail


def detect_file_extension_with_bytes(data):
    count_mrg = data.count(MRG_MAGIC)
    count_mzx0 = data.count(MZX_MAGIC)
    entry_count = int.from_bytes(data[6:8], 'little')

    log_info(f"entry_count => {entry_count}, 出现次数: mrgd00 = {count_mrg}, MZX0 = {count_mzx0}")

    if data.startswith(MZX_MAGIC):
        return SUFFIX_MZX

    # 目前观察, 如果是一个 MZP 文件, 则会以 mrgd00 开头, 然后 MZX0 出现的次数为 entry_count - 1
    if data.startswith(MRG_MAGIC):

        if count_mzx0 == entry_count - 1:
            return SUFFIX_MZP
        else:
            # 其他情况暂时看作就是一个 mrg 文件
            return SUFFIX_MRG

    return SUFFIX_BIN


def detect_file_extension(data):
    if data.startswith(MZX_MAGIC):
        return SUFFIX_MZX

    # 不准确, 特别在去掉 .hed 文件的单 mrg 文件上, 这样判定会出错
    if data.startswith(MRG_MAGIC) and len(data) >= 8:
        # 第 6~8 字节为 entry count（2 字节 little-endian）
        entry_count = int.from_bytes(data[6:8], 'little')

        # 每个 entry 8 字节 + 8 字节头 = 至少需要 6+2+8×n 字节
        # expected_min_size = 6 + 2 + (entry_count * 8)
        # if len(data) >= expected_min_size:
        #     return '.mzp'

        # 基于启发式判断：entry_count > 0 且不过大
        if 0 < entry_count < 0x1000:
            # return SUFFIX_MZP
            # 其实本质是就是一个 mrg 文件, 最后按照 mzp 的格式去解构
            return SUFFIX_MRG

    return SUFFIX_BIN


class ArchiveEntry:
    def __init__(self, sector_offset, offset, sector_size_upper_boundary, size, number_of_entries):
        self.sector_offset = sector_offset
        self.offset = offset
        self.sector_size_upper_boundary = sector_size_upper_boundary
        self.size = size
        self.real_size = (sector_size_upper_boundary - 1) // NAM_ENTRY_BLOCK_SIZE * 0x10000 + size
        data_start_offset = 6 + 2 + number_of_entries * 8
        self.real_offset = data_start_offset + self.sector_offset * DEFAULT_SECTOR_SIZE + self.offset
