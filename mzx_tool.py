import io
import math
import numpy as np
from Constants import *


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
