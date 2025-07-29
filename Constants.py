TAG_SUCCEED = "[SUCC]: "
TAG_INFO = "[INFO]: "
TAG_ERR = "[Error]: "
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

# 默认编码
DEFAULT_ENCODING = "Shift_JIS"

# 默认的扇区大小
DEFAULT_SECTOR_SIZE = 0x800

# MRG 的文件头
MRG_MAGIC = b"mrgd00"

# MZX 的文件头
MZX_MAGIC = b"MZX0"

# NAM 的文件头
NAM_MAGIC = b"MRG.NAM"

SUFFIX_MRG = ".mrg"
SUFFIX_MZX = ".mzx"
SUFFIX_MZP = ".mzp"
SUFFIX_BIN = ".bin"


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
