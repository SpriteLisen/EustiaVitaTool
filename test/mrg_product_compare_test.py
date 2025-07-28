import os
import sys
from pathlib import Path

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from Constants import *


def binary_equal(file1, file2):
    """逐字节比较两个文件是否完全一致"""
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            b1 = f1.read(4096)
            b2 = f2.read(4096)
            if b1 != b2:
                return False
            if not b1:  # EOF
                return True


def compare_directories(dir1, dir2):
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    files1 = {f.name for f in dir1.iterdir() if f.is_file()}
    files2 = {f.name for f in dir2.iterdir() if f.is_file()}

    common_files = files1 & files2

    assert len(files1) == len(files2), "File count mismatch: the number of files in both folders must be the same."

    log_info("== Comparing files ==")
    for filename in sorted(common_files):
        f1 = dir1 / filename
        f2 = dir2 / filename
        if binary_equal(f1, f2):
            log_same(filename)
        else:
            log_diff(filename)
            raise Exception("File do not match => {0}".format(filename))

    log_info("Compared {0} files successfully.".format(len(files1)))


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        log_error("Usage: python mrg_product_compare_test.py <dir1> <dir2>")
    else:
        compare_directories(sys.argv[1], sys.argv[2])
