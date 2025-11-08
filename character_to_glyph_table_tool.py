import os
import json
import subprocess
import numpy as np
from pathlib import Path
from Constants import normal_char
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_STYLE_WHITE = "white"
FONT_STYLE_OUTLINE = "outline"

# 最终的颜色值 = MAX_COLORS_PER_IMAGE * ALPHA_LEVELS
# 例如 4bpp = 8 * 2 = 4 * 4
# 纯白最大的调色板数量
WHITE_MAX_COLORS_PER_IMAGE = 8
# 纯白透明度分级数量（比如 2/4/6/8）
WHITE_ALPHA_LEVELS = 2

# 黑色描边最大的调色板数量
OUTLINE_MAX_COLORS_PER_IMAGE = 4
# 黑色描边透明度分级数量（比如 2/4/6/8）
OUTLINE_ALPHA_LEVELS = 3

# Unsharp 强度（0.5 - 2.0 范围）
UNSHARP_AMOUNT = 0.8
# Unsharp 高斯模糊半径
UNSHARP_RADIUS = 2.0


def optimize_alpha_unsharp(pil_img, amount=1.2, radius=1.0, threshold=0):
    """
    对 RGBA 图像的 alpha 通道做 Unsharp Mask，返回修改后 RGBA PIL Image。
    如果没有 scipy，也会使用 PIL 的 GaussianBlur 做近似。
    """
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img, dtype=np.float32)
    alpha = arr[:, :, 3]

    try:
        # 优先使用 scipy 的 gaussian_filter
        from scipy import ndimage
        blurred = ndimage.gaussian_filter(alpha, sigma=radius)
    except Exception:
        # 回退：PIL 的 GaussianBlur（注意：PIL blur 对数组要先转回图像）
        blurred_img = Image.fromarray(alpha.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
        blurred = np.array(blurred_img, dtype=np.float32)

    mask = alpha - blurred
    if threshold > 0:
        mask = np.where(np.abs(mask) > threshold, mask, 0.0)

    enhanced = alpha + amount * mask
    enhanced = np.clip(enhanced, 0, 255)

    # 保留原始严格透明和完全不透明的决策（避免核心被扭曲）
    core_mask = alpha >= 240
    enhanced[core_mask] = 255

    arr[:, :, 3] = enhanced.astype(np.uint8)
    return Image.fromarray(arr.astype(np.uint8))


def quantize_alpha_levels(image, levels=4, threshold=10):
    """
    将 alpha 通道量化为指定数量 levels（包含 0 和 255），返回 RGBA PIL Image。
    levels >= 2
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    arr = np.array(image)
    alpha = arr[:, :, 3].astype(np.float32)

    alpha[alpha < threshold] = 0.0

    if levels <= 2:
        alpha = np.where(alpha >= threshold, 255.0, 0.0)
    else:
        # 包括 0 和 255 在内，生成 levels 个值均匀分布
        quant_values = np.linspace(0, 255, levels)
        # boundaries between quant values
        boundaries = (quant_values[:-1] + quant_values[1:]) / 2.0
        out = np.zeros_like(alpha)
        for i, v in enumerate(quant_values):
            if i == 0:
                mask = alpha <= boundaries[0]
            elif i == len(quant_values) - 1:
                mask = alpha > boundaries[-1]
            else:
                mask = (alpha > boundaries[i - 1]) & (alpha <= boundaries[i])
            out[mask] = v
        alpha = out

    arr[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(arr)


def draw_char_into_cell_mask(
        char, cell_w, cell_h, ttf_path,
        max_font_size=None, base_font_size=6,
        mode=FONT_STYLE_WHITE, outline_width=1,
        baseline_point=0.6
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

    def calc_info(fnt):
        # 获取字符bbox
        left, top, right, bottom = draw.textbbox((0, 0), char, font=fnt)
        text_width = right - left

        # 水平居中
        calc_x = (cell_w - text_width) // 2 - left
        # 计算 baseline 区域
        x_height = font_size * baseline_point
        x_height_line = cell_w // 2
        calc_y = x_height_line - x_height

        return calc_x, calc_y, right - left, bottom - top

    x, y, w, h = calc_info(font)

    # 如果文字太大，调整字体大小
    while (w > cell_w or h > cell_h) and font_size > 4:
        font_size -= 1
        try:
            font = ImageFont.truetype(ttf_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        x, y, w, h = calc_info(font)


    if mode == FONT_STYLE_WHITE:
        # 白色透明模式
        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
    elif mode == FONT_STYLE_OUTLINE:
        # 绘制周围描边（减少偏移位置数量）
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), char, font=font, fill=(0, 0, 0, 255))

        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))

        # 先用 Unsharp 对 alpha 做增强（防止后续量化导致断层）
        mask = optimize_alpha_unsharp(
            mask,
            amount=UNSHARP_AMOUNT,
            radius=UNSHARP_RADIUS
        )

        # 再把 alpha 量化成指定级别（减少中间 alpha 的数量）
        mask = quantize_alpha_levels(
            mask,
            levels=WHITE_ALPHA_LEVELS if mode == FONT_STYLE_WHITE else OUTLINE_ALPHA_LEVELS,
            threshold=6
        )

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


def optimize_image_palette(image, mode, max_colors=16):
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
    if mode == FONT_STYLE_WHITE:
        quantized = temp_image.quantize(colors=max_colors, method=Image.MEDIANCUT)
    else:
        quantized = temp_image.quantize(
            colors=max_colors,
            method=Image.MAXCOVERAGE,
            kmeans=4,
            dither=Image.FLOYDSTEINBERG
        )
        # quantized = temp_image.quantize(colors=max_colors, method=Image.FASTOCTREE, kmeans=0)

    # 将量化后的图像转换回RGBA，但保留原始Alpha通道
    result = quantized.convert('RGBA')

    # 获取原始Alpha通道
    alpha = image.getchannel('A')

    # 应用原始Alpha通道
    result.putalpha(alpha)

    return result


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
        baseline_point=0.6,
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
        # 转换为RGBA模式
        img = Image.open(img_path).convert("RGBA")
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
                        outline_width=outline_width,
                        baseline_point=baseline_point
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

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)

        # 优化图像调色板
        if optimize_palette:
            img = optimize_image_palette(
                img, mode,
                max_colors=WHITE_MAX_COLORS_PER_IMAGE if mode == FONT_STYLE_WHITE else OUTLINE_MAX_COLORS_PER_IMAGE
            )

            img.save(out_path, optimize=True)

        # 把图片压缩，用最大颜色数限制
        # subprocess.run([
        #     "pngquant",
        #     "--force",
        #     "--ordered",
        #     "--output", out_path,
        #     "--colors", f"{MAX_COLORS_PER_IMAGE}",
        #     "--speed", "1",
        #     out_path
        # ])
        print(f"Saved {out_path}")

    # 保存映射文件
    with open(mapping_output, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)

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
        "baseline_point": 0.6,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "sein": {
        "table_dir": "glyphTable/type/se/sein",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/sein",
        "base_font_size": 6,
        "baseline_point": 0.6,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "marf": {
        "table_dir": "glyphTable/type/mar/marf",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marf",
        "base_font_size": 10,
        "baseline_point": 0.52,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "marn": {
        "table_dir": "glyphTable/type/mar/marn",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marn",
        "base_font_size": 10,
        "baseline_point": 0.52,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "popf": {
        "table_dir": "glyphTable/type/pop/popf",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popf",
        "base_font_size": 8,
        "baseline_point": 0.6,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "popn": {
        "table_dir": "glyphTable/type/pop/popn",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popn",
        "base_font_size": 8,
        "baseline_point": 0.6,
        "outline_width": 1,
        "mode": FONT_STYLE_WHITE,
    },

    "gotf": {
        "table_dir": "glyphTable/type/got/gotf",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotf",
        "base_font_size": 9,
        "baseline_point": 0.5,
        "outline_width": 1,
        "mode": FONT_STYLE_OUTLINE,
    },
    "gotn": {
        "table_dir": "glyphTable/type/got/gotn",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotn",
        "base_font_size": 9,
        "baseline_point": 0.5,
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
            outline_width=choose_config["outline_width"],
            baseline_point=choose_config["baseline_point"]
        )
