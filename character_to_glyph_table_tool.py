import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def find_closest_palette_index(palette, target=(255, 255, 255)):
    """返回与 target RGB 最接近的调色板索引"""
    best_idx = 0
    best_dist = None
    for i in range(256):
        r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
        dist = (r - target[0]) ** 2 + (g - target[1]) ** 2 + (b - target[2]) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def detect_transparent_index(img):
    """获取 P 模式图片的透明索引"""
    transparency = img.info.get("transparency", None)
    if isinstance(transparency, int):
        return transparency
    elif isinstance(transparency, (bytes, bytearray)):
        for i, alpha in enumerate(transparency):
            if alpha == 0:
                return i
        return 0
    else:
        return 0


def build_gray_to_palette_map(palette):
    """
    构建灰度值(0~255) -> 调色板索引映射表
    根据 RGB 灰度 = (R+G+B)//3
    """
    gray_map = [0] * 256
    for g in range(256):
        best_idx = 0
        best_dist = None
        for i in range(256):
            r, g_i, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
            gray = (r + g_i + b) // 3
            dist = abs(gray - g)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        gray_map[g] = best_idx
    return gray_map


def draw_char_into_cell_mask(char, cell_w, cell_h, ttf_path,
                             max_fontsize=None, scale=10, gray_levels=5):
    """
    返回一个 L 模式的 mask（大小 cell_w x cell_h），文字为白(255)背景黑(0)，
    使用超采样抗锯齿，可控制灰阶数量。

    参数：
    - scale: 超采样倍率，越大边缘越平滑
    - gray_levels: 输出灰阶数量，>=2
    """
    if max_fontsize is None:
        max_fontsize = min(cell_w, cell_h)

    w_big, h_big = cell_w * scale, cell_h * scale
    fontsize = (max_fontsize - 6) * scale  # 调小4像素留边

    while fontsize > 4 * scale:
        try:
            font = ImageFont.truetype(ttf_path, fontsize)
        except Exception:
            font = ImageFont.load_default()
        mask_big = Image.new("L", (w_big, h_big), 0)
        draw = ImageDraw.Draw(mask_big)
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= w_big and h <= h_big:
            # 居中绘制
            x = (w_big - w) // 2 - bbox[0]
            y = (h_big - h) // 2 - bbox[1]
            draw.text((x, y), char, font=font, fill=255)

            # 缩小回原始格子尺寸
            mask = mask_big.resize((cell_w, cell_h), Image.LANCZOS)

            # 灰阶量化
            def quantize_gray(p):
                level = int(p / 255 * (gray_levels - 1) + 0.5)
                return int(level * 255 / (gray_levels - 1))

            mask = mask.point(quantize_gray)
            return mask, font
        fontsize -= scale

    # 最小字号绘制
    font = ImageFont.truetype(ttf_path, max(8, 4) * scale)
    mask_big = Image.new("L", (w_big, h_big), 0)
    draw = ImageDraw.Draw(mask_big)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w_big - w) // 2 - bbox[0]
    y = (h_big - h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=255)
    mask = mask_big.resize((cell_w, cell_h), Image.LANCZOS)

    def quantize_gray(p):
        level = int(p / 255 * (gray_levels - 1) + 0.5)
        return int(level * 255 / (gray_levels - 1))

    mask = mask.point(quantize_gray)
    return mask, font


def render_chars_to_images(
        image_files,
        chars,
        ttf_path,
        output_dir="out",
        cols=14, rows=14,
):
    os.makedirs(output_dir, exist_ok=True)
    char_iter = iter(chars)

    for img_path in image_files:
        img = Image.open(img_path)
        if img.mode != "P":
            img = img.convert("P")
        W, H = img.size
        cell_w, cell_h = W // cols, H // rows
        palette = img.getpalette()
        if palette is None:
            raise RuntimeError(f"{img_path} has no palette")
        # white_index = find_closest_palette_index(palette, (128, 128, 128))
        transparent_index = detect_transparent_index(img)
        gray_map = build_gray_to_palette_map(palette)
        px = img.load()

        for r in range(rows):
            for c in range(cols):
                try:
                    ch = next(char_iter)
                except StopIteration:
                    ch = None

                x0, y0 = c * cell_w, r * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h

                # 清空格子为透明
                if transparent_index is not None:
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            px[xx, yy] = transparent_index

                if ch is None:
                    continue

                # 生成抗锯齿 mask
                mask, used_font = draw_char_into_cell_mask(ch, cell_w, cell_h, ttf_path,
                                                           max_fontsize=min(cell_h, cell_w))

                # 写入 P 图，灰度映射到调色板
                for yy in range(cell_h):
                    yy_abs = y0 + yy
                    for xx in range(cell_w):
                        gray_val = mask.getpixel((xx, yy))
                        if gray_val > 0:
                            px[x0 + xx, yy_abs] = gray_map[gray_val]
                            # px[x0 + xx, yy_abs] = white_index
                        # else 保持透明格子

        # 保存输出
        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)
        save_kwargs = {}
        if "transparency" in img.info:
            save_kwargs["transparency"] = img.info["transparency"]
        img.save(out_path, **save_kwargs)
        print(f"Saved {out_path}")

    print("All images processed.")


# 备选字体方案
# seif/sein: 都选用 YanZhenQingDuoBaoTaBei.ttf【颜真卿多宝塔碑体】(对应游戏默认字体: 明朝)
# marf/marn: 都选用 GongFanLiZhongYuan.ttf【中圆】(对应游戏默认字体: 丸ゴシック{圆黑体})
# popf/popn: 都选用 KeAiPao-TaoZiJiu.ttf【可爱泡芙-桃子酒】(对应游戏默认字体: ボップ{泡泡体})
# gotf/gotn: 都选用 SanJiLuoLiHei-Cu.ttf【三极罗丽黑-粗】(对应游戏默认字体: ゴシック{黑体})

if __name__ == "__main__":
    target_input_glyph_table_dir = "glyphTable/seif"
    image_files = list(Path(target_input_glyph_table_dir).glob('*.[Pp][Nn][Gg]'))

    all_texts = []
    with open("poems_19_x14x14.txt", "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\n", "").replace("\r", "")
        all_texts.append(text)
    combined_text = "".join(all_texts)
    chars = sorted(set(combined_text))
    print("读入字符文件内容:")
    print(chars)
    print(f"\n总字符数: {len(chars)}")

    render_chars_to_images(
        image_files, chars,
        "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        output_dir="glyphTable/test"
    )
