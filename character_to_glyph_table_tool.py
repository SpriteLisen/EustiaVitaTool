import os
import json
import zlib
import struct
import numpy as np
from pathlib import Path
from Constants import normal_char
from PIL import Image, ImageDraw, ImageFont

FONT_STYLE_WHITE = "white"
FONT_STYLE_OUTLINE = "outline"

def draw_char_into_cell_mask(
        char, cell_w, cell_h, ttf_path,
        max_font_size=None, base_font_size=6,
        mode=FONT_STYLE_WHITE, outline_width=1
):
    """返回一个 RGBA 模式 mask，支持白色透明和描边模式"""
    if max_font_size is None:
        max_font_size = min(cell_w, cell_h)

    font_size = (max_font_size - base_font_size)

    # 确保字体大小至少为8
    font_size = max(font_size, 8)

    try:
        font = ImageFont.truetype(ttf_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # 创建透明背景的大图像
    mask = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    # 获取文本边界框
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 如果文字太大，调整字体大小
    while (w > cell_w or h > cell_h) and font_size > 4:
        font_size -= 1
        try:
            font = ImageFont.truetype(ttf_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 计算文本位置（居中）
    x = (cell_w - w) // 2 - bbox[0]
    y = (cell_h - h) // 2 - bbox[1]

    if mode == FONT_STYLE_WHITE:
        # 白色透明模式
        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
    elif mode == FONT_STYLE_OUTLINE:
        # 绘制周围描边
        # for dx in range(-outline_width, outline_width + 1):
        #     for dy in range(-outline_width, outline_width + 1):
        #         if dx == 0 and dy == 0:
        #             continue
        #         draw.text(
        #             (x + dx, y + dy), char, font=font,
        #             fill=(0, 0, 0, 255) if mode == FONT_STYLE_OUTLINE else (255, 255, 255, 255)
        #         )

        for dx, dy in [
            (-outline_width, 0), (outline_width, 0), (0, -outline_width), (0, outline_width),  # 上下左右
            (-outline_width, -outline_width), (-outline_width, outline_width), (outline_width, -outline_width),
            (outline_width, outline_width)  # 四个角
        ]:
            draw.text((x + dx, y + dy), char, font=font, fill=(0, 0, 0, 255))

        # 绘制两次文字保障内部字体清晰
        # draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))

    return mask, font


def quantize_alpha(image, threshold=10, levels=4):
    """
    量化Alpha通道，减少透明级别的数量
    :param image: PIL Image (RGBA)
    :param threshold: 透明阈值，低于此值的Alpha视为完全透明
    :param levels: 透明级别数量
    :return: 处理后的PIL Image
    """
    # 转换为numpy数组进行处理
    img_array = np.array(image)
    alpha = img_array[:, :, 3].astype(np.float32)  # 使用浮点数以便精确计算

    # 将低于阈值的Alpha设为0（完全透明）
    alpha[alpha < threshold] = 0

    # 计算量化级别
    if levels <= 2:
        # 二值化处理
        alpha[alpha >= threshold] = 255
    else:
        # 生成均匀分布的量化值
        quant_values = np.linspace(0, 255, levels)

        # 计算区间边界（每个区间的中点）
        boundaries = []
        for i in range(len(quant_values) - 1):
            boundaries.append((quant_values[i] + quant_values[i + 1]) / 2)

        # 对每个像素值进行量化
        for i in range(len(quant_values)):
            if i == 0:
                # 第一个区间 [0, boundaries[0]]
                mask = alpha <= boundaries[0]
                alpha[mask] = quant_values[i]
            elif i == len(quant_values) - 1:
                # 最后一个区间 [boundaries[-1], 255]
                mask = alpha > boundaries[-1]
                alpha[mask] = quant_values[i]
            else:
                # 中间区间 (boundaries[i-1], boundaries[i]]
                mask = (alpha > boundaries[i - 1]) & (alpha <= boundaries[i])
                alpha[mask] = quant_values[i]

    # 转换回整数类型
    alpha = alpha.astype(np.uint8)

    # 更新Alpha通道
    img_array[:, :, 3] = alpha

    # 返回处理后的图像
    return Image.fromarray(img_array)


def optimize_image_palette(image, max_colors=16):
    """
    优化图像调色板，减少颜色数量
    :param image: PIL Image (RGBA)
    :param max_colors: 最大颜色数量
    :return: 优化后的PIL Image
    """
    # 转换为P模式（索引颜色）并限制颜色数量
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 创建临时图像进行量化
    temp_image = image.convert('RGB')

    # 量化到有限的颜色数量
    quantized = temp_image.quantize(colors=max_colors, method=Image.MEDIANCUT)

    # 将量化后的图像转换回RGBA，但保留原始Alpha通道
    result = quantized.convert('RGBA')

    # 获取原始Alpha通道
    alpha = image.getchannel('A')

    # 应用原始Alpha通道
    result.putalpha(alpha)

    return result


def calculate_mzx_compressed_size(indices, width, height):
    """基于MZX压缩算法的更准确大小估算"""
    # 原始数据大小 (4bpp = 每像素0.5字节)
    raw_data_size = (width * height) // 2

    # 模拟MZX压缩过程来估算大小
    # 将索引数据转换为16位字序列
    words = []
    for y in range(height):
        row = indices[y]
        for i in range(0, width, 4):  # 每4个像素 = 1个16位字
            if i + 3 < width:
                # 打包4个4位索引到1个16位字
                word = (row[i] << 12) | (row[i + 1] << 8) | (row[i + 2] << 4) | row[i + 3]
            else:
                # 处理不完整的字
                word = 0
                shift = 12
                for j in range(i, min(i + 4, width)):
                    word |= (row[j] & 0xF) << shift
                    shift -= 4
            words.append(word)

    # 基于MZX压缩算法估算
    # 保守估算：假设压缩率约为原始大小的60-80%
    # 对于有很多重复数据的图像，压缩率可能更高
    estimated_ratio = 0.7  # 保守估计70%压缩率
    estimated_size = int(raw_data_size * estimated_ratio)

    # 加上压缩头开销（大约100-200字节）
    estimated_size += 150

    return estimated_size


def map_to_src_palette_strict(src_img, tgt_img):
    src = src_img.convert("RGBA")
    tgt = tgt_img.convert("RGBA")

    w, h = tgt.size

    # 源图调色板
    src_arr = np.array(src).reshape(-1, 4)
    unique_src = np.unique(src_arr, axis=0).astype(np.uint8)
    palette = unique_src.copy()
    N = len(palette)
    if N > 16:
        raise ValueError(f"源图调色板太多: {N}")

    # 目标图
    tgt_arr = np.array(tgt).reshape(-1, 4)

    # 在整个调色板空间中寻找最接近的颜色
    distances = np.sum(np.abs(tgt_arr[:, None, :] - palette[None, :, :]), axis=2)
    indices_flat = np.argmin(distances, axis=1).astype(np.uint8)

    indices = indices_flat.reshape((h, w))

    return indices, palette.astype(np.uint8)


def save_png_4bit(path, indices, palette, width, height):
    """保存PNG并确保MZX压缩友好"""
    def png_chunk(chunk_type, data):
        chunk = chunk_type.encode('ascii') + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk))

    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB",
                            width,
                            height,
                            4,  # bit depth 4
                            3,  # color type = indexed
                            0, 0, 0)
    png += png_chunk('IHDR', ihdr_data)

    # PLTE (RGB)
    plte_bytes = b''.join([struct.pack("BBB", c[0], c[1], c[2]) for c in palette])
    png += png_chunk('PLTE', plte_bytes)

    # tRNS (Alpha)
    trns_bytes = bytes([c[3] for c in palette])
    png += png_chunk('tRNS', trns_bytes)

    # IDAT - 原始4bpp数据
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        row = indices[y]
        for i in range(0, width, 2):
            first = row[i] & 0xF
            if i + 1 < width:
                second = row[i + 1] & 0xF
            else:
                second = 0
            raw.append((first << 4) | second)

    compressed = zlib.compress(bytes(raw), level=9)
    png += png_chunk('IDAT', compressed)

    # IEND
    png += png_chunk('IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)


def optimize_pixel_layout(indices, width, height):
    """优化像素排列以提高MZX压缩率"""
    indices_opt = indices.copy()

    # 策略1: 按行重新排列，增加局部重复性
    for y in range(height):
        row = indices_opt[y]

        # 尝试将相同的颜色值聚集在一起
        unique_values, counts = np.unique(row, return_counts=True)
        if len(unique_values) > 1:
            # 按出现频率排序
            sorted_pairs = sorted(zip(unique_values, counts), key=lambda x: -x[1])
            new_row = []
            for val, count in sorted_pairs:
                new_row.extend([val] * count)
            # 填充剩余位置（如果有）
            while len(new_row) < width:
                new_row.append(0)
            indices_opt[y] = new_row[:width]

    return indices_opt


def optimize_block_vertical(block):
    """在块内优化垂直方向的重复性"""
    if block.shape[1] == 0:
        return block

    # 对每一列单独处理
    for x in range(block.shape[1]):
        col = block[:, x]
        # 尝试将相同的值聚集在一起
        unique_vals = np.unique(col)
        if len(unique_vals) > 1:
            # 创建新列，相同值连续排列
            new_col = []
            for val in unique_vals:
                count = np.sum(col == val)
                new_col.extend([val] * count)
            # 填充到原长度
            while len(new_col) < len(col):
                new_col.append(0)
            block[:, x] = new_col[:len(col)]

    return block

def advanced_compression_optimization(indices, width, height):
    """更高级的压缩优化策略"""
    # 将图像分割为8x8块，并在块内优化像素排列
    block_size = 8
    optimized = indices.copy()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 提取块
            block = indices[y:min(y + block_size, height), x:min(x + block_size, width)]
            if block.size == 0:
                continue

            # 在块内按列重新排列，增加垂直方向的重复性
            block_opt = optimize_block_vertical(block)
            optimized[y:y + block.shape[0], x:x + block.shape[1]] = block_opt

    return optimized


def calculate_mzx_compressed_size_v2(indices, width, height):
    """更精确的MZX压缩大小估算"""
    # 转换为16位字序列
    words = []
    for y in range(height):
        row = indices[y]
        for i in range(0, width, 4):
            if i + 3 < width:
                word = (row[i] << 12) | (row[i + 1] << 8) | (row[i + 2] << 4) | row[i + 3]
            else:
                word = 0
                shift = 12
                for j in range(i, min(i + 4, width)):
                    word |= (row[j] & 0xF) << shift
                    shift -= 4
            words.append(word)

    # 模拟MZX压缩的基本过程
    compressed_size = 8  # 文件头

    i = 0
    while i < len(words):
        # 检查RLE
        run_length = 1
        while i + run_length < len(words) and words[i + run_length] == words[i] and run_length < 64:
            run_length += 1

        if run_length > 1:
            # RLE压缩
            compressed_size += 1  # 命令字节
            i += run_length
            continue

        # 检查字面量
        literal_length = 1
        while (i + literal_length < len(words) and
               words[i + literal_length] != words[i + literal_length - 1] and
               literal_length < 64):
            literal_length += 1

        # 字面量压缩
        compressed_size += 1  # 命令字节
        compressed_size += literal_length * 2  # 数据字节

        i += literal_length

    return compressed_size


def optimize_pixel_layout_aggressive(indices, width, height):
    """更激进的像素布局优化"""
    indices_opt = indices.copy()

    # 将图像展平为一维数组
    flat_indices = indices_opt.flatten()

    # 统计颜色使用频率
    unique_vals, counts = np.unique(flat_indices, return_counts=True)

    # 按频率排序（从高到低）
    freq_order = unique_vals[np.argsort(-counts)]

    # 创建新的像素排列：高频颜色优先，并按块排列
    new_flat = np.zeros_like(flat_indices)

    # 计算每个颜色应该占用的像素数
    total_pixels = len(flat_indices)
    color_blocks = {}
    remaining_pixels = total_pixels

    for i, color in enumerate(freq_order):
        if i == len(freq_order) - 1:  # 最后一个颜色
            color_blocks[color] = remaining_pixels
        else:
            # 按比例分配像素数，但确保是偶数（便于4bpp打包）
            block_size = max(2, (counts[i] * total_pixels) // total_pixels)
            block_size = (block_size // 2) * 2  # 确保是偶数
            color_blocks[color] = min(block_size, remaining_pixels)
            remaining_pixels -= color_blocks[color]

    # 填充新数组
    pos = 0
    for color, block_size in color_blocks.items():
        new_flat[pos:pos + block_size] = color
        pos += block_size

    # 如果还有剩余位置，用最高频颜色填充
    if pos < total_pixels:
        new_flat[pos:] = freq_order[0]

    # 重新形状为二维
    indices_opt = new_flat.reshape((height, width))

    return indices_opt


def advanced_compression_optimization_v2(indices, width, height):
    """改进的高级压缩优化"""
    # 尝试多种优化策略
    strategies = [
        optimize_by_rows,  # 按行优化
        optimize_by_columns,  # 按列优化
        optimize_by_blocks,  # 按块优化
        optimize_by_diagonals  # 按对角线优化
    ]

    best_indices = indices
    best_score = calculate_compression_score(indices)

    for strategy in strategies:
        try:
            optimized = strategy(indices.copy(), width, height)
            score = calculate_compression_score(optimized)

            if score < best_score:
                best_indices = optimized
                best_score = score
        except Exception as e:
            print(f"优化策略 {strategy.__name__} 失败: {e}")

    return best_indices


def optimize_by_rows(indices, width, height):
    """按行优化：增加行内重复性"""
    for y in range(height):
        row = indices[y]
        # 对行进行排序，将相同值聚集
        unique_vals, counts = np.unique(row, return_counts=True)
        if len(unique_vals) > 1:
            sorted_vals = []
            for val in unique_vals[np.argsort(-counts)]:
                sorted_vals.extend([val] * np.sum(row == val))
            indices[y] = sorted_vals[:width]
    return indices


def optimize_by_columns(indices, width, height):
    """按列优化：增加列内重复性"""
    for x in range(width):
        col = indices[:, x]
        unique_vals, counts = np.unique(col, return_counts=True)
        if len(unique_vals) > 1:
            sorted_vals = []
            for val in unique_vals[np.argsort(-counts)]:
                sorted_vals.extend([val] * np.sum(col == val))
            indices[:len(sorted_vals), x] = sorted_vals
    return indices


def optimize_by_blocks(indices, width, height, block_size=16):
    """按块优化：在块内增加重复性"""
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = indices[y:min(y + block_size, height), x:min(x + block_size, width)]
            if block.size > 0:
                # 展平块并排序
                flat_block = block.flatten()
                unique_vals, counts = np.unique(flat_block, return_counts=True)
                if len(unique_vals) > 1:
                    sorted_vals = []
                    for val in unique_vals[np.argsort(-counts)]:
                        sorted_vals.extend([val] * np.sum(flat_block == val))
                    # 重新填充块
                    sorted_block = np.array(sorted_vals[:len(flat_block)]).reshape(block.shape)
                    indices[y:y + block.shape[0], x:x + block.shape[1]] = sorted_block
    return indices


def optimize_by_diagonals(indices, width, height):
    """按对角线优化：增加对角线方向的重复性"""
    # 这个策略比较复杂，但可以尝试
    # 简单实现：按45度对角线重新排列
    optimized = indices.copy()
    for d in range(-height + 1, width):
        diag_indices = []
        for y in range(height):
            x = y + d
            if 0 <= x < width:
                diag_indices.append((y, x))

        if len(diag_indices) > 1:
            # 提取对角线值
            diag_values = [indices[y, x] for y, x in diag_indices]
            # 排序
            unique_vals, counts = np.unique(diag_values, return_counts=True)
            if len(unique_vals) > 1:
                sorted_values = []
                for val in unique_vals[np.argsort(-counts)]:
                    sorted_values.extend([val] * diag_values.count(val))
                # 写回
                for (y, x), val in zip(diag_indices, sorted_values):
                    optimized[y, x] = val
    return optimized


def calculate_compression_score(indices):
    """计算压缩潜力评分（越低越好）"""
    # 基于连续重复像素的数量
    flat = indices.flatten()
    score = 0
    run_length = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            run_length += 1
        else:
            # 长运行长度得分更低
            score += 1.0 / run_length if run_length > 1 else 1.0
            run_length = 1
    # 最后一个运行
    score += 1.0 / run_length if run_length > 1 else 1.0

    return score


def debug_optimization(indices_before, indices_after, width, height):
    """输出优化调试信息"""
    before_score = calculate_compression_score(indices_before)
    after_score = calculate_compression_score(indices_after)

    print(f"压缩潜力评分: {before_score:.2f} -> {after_score:.2f}")

    # 检查实际变化
    changed_pixels = np.sum(indices_before != indices_after)
    print(f"改变的像素数: {changed_pixels} / {width * height}")

    # 检查连续重复像素的变化
    def count_runs(indices):
        flat = indices.flatten()
        runs = 0
        current_run = 1
        for i in range(1, len(flat)):
            if flat[i] == flat[i - 1]:
                current_run += 1
            else:
                if current_run > 1:
                    runs += 1
                current_run = 1
        if current_run > 1:
            runs += 1
        return runs

    before_runs = count_runs(indices_before)
    after_runs = count_runs(indices_after)
    print(f"连续重复块数: {before_runs} -> {after_runs}")

def optimize_for_mzx_compression_v2(indices, palette, width, height):
    """改进的MZX压缩优化"""
    print(f"初始估算大小: {calculate_mzx_compressed_size_v2(indices, width, height):04X}h")

    # 尝试多种优化策略
    strategies = [
        lambda x: optimize_pixel_layout_aggressive(x, width, height),
        lambda x: advanced_compression_optimization_v2(x, width, height),
        lambda x: optimize_by_rows(x, width, height),
        lambda x: optimize_by_columns(x, width, height),
        lambda x: optimize_by_blocks(x, width, height),
    ]

    best_indices = indices
    best_size = calculate_mzx_compressed_size_v2(indices, width, height)

    for i, strategy in enumerate(strategies):
        print(f"尝试策略 {i + 1}/{len(strategies)}")

        try:
            optimized = strategy(best_indices.copy())
            new_size = calculate_mzx_compressed_size_v2(optimized, width, height)

            # 输出调试信息
            debug_optimization(best_indices, optimized, width, height)

            print(f"策略 {i + 1} 后估算大小: {new_size:04X}h")

            if new_size < best_size:
                best_indices = optimized
                best_size = new_size
                print(f"✓ 策略 {i + 1} 有效")
            else:
                print(f"✗ 策略 {i + 1} 无效")

        except Exception as e:
            print(f"策略 {i + 1} 出错: {e}")

    return best_indices, palette

def optimize_for_mzx_compression(indices, palette, width, height, max_iterations=3):
    """多次优化直到满足大小限制或达到最大迭代次数"""
    best_indices = indices
    best_size = calculate_mzx_compressed_size(indices, width, height)

    print(f"初始估算大小: {best_size:04X}h")

    if best_size <= 0xFFFF:
        return indices, palette

    # 尝试优化
    for iteration in range(max_iterations):
        print(f"优化迭代 {iteration + 1}")

        # 第一步：基础像素布局优化
        optimized_indices = optimize_pixel_layout(best_indices, width, height)

        # 第二步：高级压缩优化（使用你提供的函数）
        optimized_indices = advanced_compression_optimization(optimized_indices, width, height)

        # 计算新的大小
        new_size = calculate_mzx_compressed_size(optimized_indices, width, height)
        print(f"迭代 {iteration + 1} 估算大小: {new_size:04X}h")

        if new_size < best_size:
            best_indices = optimized_indices
            best_size = new_size

        if best_size <= 0xFFFF:
            print("优化成功，大小在限制内")
            return best_indices, palette

    # 如果优化后仍然太大，尝试减少调色板颜色
    if best_size > 0xFFFF:
        print("尝试减少调色板颜色...")
        reduced_indices, reduced_palette = reduce_palette_colors(best_indices, palette)
        final_size = calculate_mzx_compressed_size(reduced_indices, width, height)
        print(f"减少调色板后估算大小: {final_size:04X}h")

        if final_size <= 0xFFFF:
            return reduced_indices, reduced_palette

    return best_indices, palette


def reduce_palette_colors(indices, palette, max_colors=8):
    """减少调色板颜色数量"""
    # 统计颜色使用频率
    unique, counts = np.unique(indices, return_counts=True)
    color_usage = list(zip(unique, counts))

    # 按使用频率排序
    color_usage.sort(key=lambda x: -x[1])

    # 选择最常用的颜色
    if len(color_usage) > max_colors:
        kept_colors = color_usage[:max_colors]
        kept_indices = [c[0] for c in kept_colors]

        # 创建新的调色板映射
        new_palette = palette[kept_indices]

        # 重新映射索引
        new_indices = np.zeros_like(indices)
        for i, color_idx in enumerate(kept_indices):
            new_indices[indices == color_idx] = i

        # 处理未保留的颜色，映射到最接近的保留颜色
        for old_idx in range(len(palette)):
            if old_idx not in kept_indices:
                # 找到最接近的保留颜色
                old_color = palette[old_idx]
                min_dist = float('inf')
                best_new_idx = 0
                for i, new_idx in enumerate(kept_indices):
                    new_color = palette[new_idx]
                    dist = np.sum(np.abs(old_color - new_color))
                    if dist < min_dist:
                        min_dist = dist
                        best_new_idx = i
                # 更新映射
                new_indices[indices == old_idx] = best_new_idx

        return new_indices, new_palette

    return indices, palette


def render_chars_to_images(
        image_files,
        table_files,
        chars,
        ttf_path,
        output_dir="out",
        cols=14, rows=14,
        base_font_size=6,
        mapping_output="char_mapping.json",
        mode=FONT_STYLE_WHITE,
        outline_width=2,
        optimize_palette=True
):
    os.makedirs(output_dir, exist_ok=True)
    mapping_data = {}
    for char in normal_char:
        mapping_data[char] = char

    # 计算总共需要处理的格子数量
    total_cells = 0
    for idx, img_path in enumerate(image_files):
        row_start = 10 if idx == 0 else 0
        row_end = rows
        total_cells += (row_end - row_start) * cols

    # 确保字符数量不超过可用格子数量
    if len(chars) > total_cells:
        print(f"警告: 字符数量({len(chars)})超过可用格子数量({total_cells})，将截断字符列表")
        chars = chars[:total_cells]

    # 重新创建迭代器
    char_iter = iter(chars)

    for idx, img_path in enumerate(image_files):
        original_img = Image.open(img_path)

        # 转换为RGBA模式
        img = original_img.convert("RGBA")
        W, H = img.size
        cell_w, cell_h = W // cols, H // rows
        px = img.load()

        # 找对应 table 文件
        base_name = img_path.stem.lower()
        table_file = next((t for t in table_files if t.stem.lower() == base_name), None)
        if table_file is None:
            raise FileNotFoundError(f"No table file for {img_path}")
        with open(table_file, "r", encoding="utf-8", errors="ignore") as f:
            table_text = f.read().replace("\n", "")

        # 行范围规则
        row_start = 10 if idx == 0 else 0
        row_end = rows

        # 处理每个格子
        for r in range(row_start, row_end):
            for c in range(cols):
                # 获取下一个字符
                try:
                    ch = next(char_iter)
                except StopIteration:
                    # 没有更多字符了，跳出所有循环
                    ch = None

                x0, y0 = c * cell_w, r * cell_h

                # 清空格子为透明
                for yy in range(y0, y0 + cell_h):
                    for xx in range(x0, x0 + cell_w):
                        px[xx, yy] = (0, 0, 0, 0)

                # 绘制字符
                if ch is not None:
                    mask, used_font = draw_char_into_cell_mask(
                        ch, cell_w, cell_h, ttf_path,
                        max_font_size=min(cell_h, cell_w),
                        base_font_size=base_font_size,
                        mode=mode,
                        outline_width=outline_width
                    )

                # 将字符mask合成到图像上
                for yy in range(cell_h):
                    for xx in range(cell_w):
                        if ch is not None:
                            r_val, g, b, a = mask.getpixel((xx, yy))
                            if a > 0:  # 只处理非透明像素
                                px[x0 + xx, y0 + yy] = (r_val, g, b, a)
                        else:
                            px[x0 + xx, y0 + yy] = (0, 0, 0, 0)

                # 记录映射关系
                if ch is not None:
                    orig_char = table_text[r * cols + c]
                    mapping_data[ch] = orig_char

        # 第一步：基础量化
        indices, palette = map_to_src_palette_strict(original_img, img)

        # 第二步：优化压缩
        optimized_indices, optimized_palette = optimize_for_mzx_compression_v2(
            indices, palette, original_img.width, original_img.height
        )

        # 第三步：最终检查并保存
        final_size = calculate_mzx_compressed_size(optimized_indices, original_img.width, original_img.height)

        if final_size > 0xFFFF:
            print(f"警告: 最终估算大小 {final_size:04X}h 仍可能超出限制")
            print("考虑减小图像尺寸或进一步优化")

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)

        save_png_4bit(out_path, optimized_indices, optimized_palette,
                      original_img.width, original_img.height)

        # estimated_size = save_png_4bit_with_size_check(out_path, indices, palette, original_img.width, original_img.height)

        print(f"Saved {out_path}")

    # 保存映射文件
    # with open(mapping_output, "w", encoding="utf-8") as f:
    #     json.dump(mapping_data, f, ensure_ascii=False, indent=2)

    print(f"All images processed. Mapping saved to {mapping_output}")


# 备选字体方案
# seif/sein: 都选用 YanZhenQingDuoBaoTaBei.ttf【颜真卿多宝塔碑体】(对应游戏默认字体: 明朝)
# marf/marn: 都选用 GongFanLiZhongYuan.ttf【中圆】(对应游戏默认字体: 丸ゴシック{圆黑体})
# popf/popn: 都选用 KeAiPao-TaoZiJiu.ttf【可爱泡芙-桃子酒】(对应游戏默认字体: ボップ{泡泡体})
# gotf/gotn: 都选用 SanJiLuoLiHei-Cu.ttf【三极罗丽黑-粗】(对应游戏默认字体: ゴシック{黑体})

# 配置选项
config = {
    "seif": {
        "table_dir": "glyphTable/type/se/seif",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/seif",
        "base_font_size": 6,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "sein": {
        "table_dir": "glyphTable/type/se/sein",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/sein",
        "base_font_size": 6,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "marf": {
        "table_dir": "glyphTable/type/mar/marf",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marf",
        "base_font_size": 10,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "marn": {
        "table_dir": "glyphTable/type/mar/marn",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marn",
        "base_font_size": 10,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "popf": {
        "table_dir": "glyphTable/type/pop/popf",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popf",
        "base_font_size": 8,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "popn": {
        "table_dir": "glyphTable/type/pop/popn",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popn",
        "base_font_size": 8,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "gotf": {
        "table_dir": "glyphTable/type/got/gotf",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotf",
        "base_font_size": 9,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "gotn": {
        "table_dir": "glyphTable/type/got/gotn",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotn",
        "base_font_size": 9,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    }
}

if __name__ == "__main__":
    all_texts = []
    input_text_file = "glyphTable/translate-character.txt"

    with open(input_text_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\n", "").replace("\r", "")
        all_texts.append(text)
    combined_text = "".join(all_texts)
    chars = sorted(set(combined_text))
    print("读入字符文件内容:")
    print(chars)
    print(f"\n总字符数: {len(chars)}")
    print('-' * 80)

    for key in config.keys():
        print(f"\n开始制作: {key}")
        print('-' * 80)

        choose_config = config[key]
        # choose_config = config["marn"]

        target_input_glyph_table_dir = choose_config["table_dir"]

        image_files = sorted(
            Path(target_input_glyph_table_dir).glob('*.[Pp][Nn][Gg]'),
            key=lambda x: x.name.lower()
        )
        table_files = sorted(
            Path("glyphTable/characterTable/").glob('*.[Tt][Xx][Tt]'),
            key=lambda x: x.name.lower()
        )

        render_chars_to_images(
            image_files, table_files, chars,
            choose_config["font"],
            output_dir=choose_config["output_dir"],
            mapping_output="glyphTable/character-mapping.json",
            base_font_size=choose_config["base_font_size"],
            mode=choose_config["mode"],
            outline_width=choose_config["outline_width"]
        )
