TAG_INFO = "[INFO]: "
TAG_ERR = "[Error]: "
TAG_WARN = "[Warn]: "

RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLACK = "\033[90m"


def log_info(msg):
    # print(f"{BLACK}{TAG_INFO} {msg}{RESET}")
    print(f"{TAG_INFO} {msg}")


def log_warn(msg):
    print(f"{YELLOW}{TAG_WARN} {msg}{RESET}")


def log_error(msg):
    print(f"{RED}{TAG_ERR} {msg}{RESET}")


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
