import os
import json
from pathlib import Path
from Constants import normal_char
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
    """返回一个 L 模式 mask，文字为白(255)背景黑(0)，灰度可量化"""
    if max_fontsize is None:
        max_fontsize = min(cell_w, cell_h)

    w_big, h_big = cell_w * scale, cell_h * scale
    fontsize = (max_fontsize - 6) * scale  # 调小留边

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
            x = (w_big - w) // 2 - bbox[0]
            y = (h_big - h) // 2 - bbox[1]
            draw.text((x, y), char, font=font, fill=255)

            mask = mask_big.resize((cell_w, cell_h), Image.LANCZOS)

            def quantize_gray(p):
                level = int(p / 255 * (gray_levels - 1) + 0.5)
                return int(level * 255 / (gray_levels - 1))

            mask = mask.point(quantize_gray)
            return mask, font
        fontsize -= scale

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
        table_files,
        chars,
        ttf_path,
        output_dir="out",
        cols=14, rows=14,
        mapping_output="char_mapping.json"
):
    os.makedirs(output_dir, exist_ok=True)
    char_iter = iter(chars)
    mapping_data = {}
    # 增加默认的字符映射写入, 这些映射是不进行修改的
    for char in normal_char:
        mapping_data[char] = char

    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path)
        if img.mode != "P":
            img = img.convert("P")
        W, H = img.size
        cell_w, cell_h = W // cols, H // rows
        palette = img.getpalette()
        if palette is None:
            raise RuntimeError(f"{img_path} has no palette")
        transparent_index = detect_transparent_index(img)
        gray_map = build_gray_to_palette_map(palette)
        px = img.load()

        # 找对应 table 文件
        base_name = img_path.stem.lower()
        table_file = next((t for t in table_files if t.stem.lower() == base_name), None)
        if table_file is None:
            raise FileNotFoundError(f"No table file for {img_path}")
        with open(table_file, "r", encoding="utf-8", errors="ignore") as f:
            table_text = f.read().replace("\n", "")
        # if len(table_text) != rows * cols:
        #     raise ValueError(f"{table_file} size mismatch, expected {rows * cols} chars, got {len(table_text)}")

        img_mapping = {}

        # 行范围规则
        row_start = 10 if idx == 0 else 0
        row_end = rows

        for r in range(rows):
            for c in range(cols):
                # 只在指定行范围内修改
                if not (row_start <= r < row_end):
                    continue

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

                mask, used_font = draw_char_into_cell_mask(ch, cell_w, cell_h, ttf_path,
                                                           max_fontsize=min(cell_h, cell_w))

                for yy in range(cell_h):
                    yy_abs = y0 + yy
                    for xx in range(cell_w):
                        gray_val = mask.getpixel((xx, yy))
                        if gray_val > 0:
                            px[x0 + xx, yy_abs] = gray_map[gray_val]

                orig_char = table_text[r * cols + c]
                mapping_data[ch] = orig_char

        #         img_mapping[f"{r},{c}"] = {
        #             "orig": orig_char,
        #             "new": ch
        #         }
        #
        # mapping_data[img_path.name] = img_mapping

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)
        save_kwargs = {}
        if "transparency" in img.info:
            save_kwargs["transparency"] = img.info["transparency"]
        img.save(out_path, **save_kwargs)
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

config = {
    "se": {
        "table_dir": "glyphTable/type/se/seif",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/se",
    },
    "mar": {
        "table_dir": "glyphTable/type/mar/marf",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/mar",
    },
    "pop": {
        "table_dir": "glyphTable/type/pop/popf",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/pop",
    },
    "got": {
        "table_dir": "glyphTable/type/got/gotf",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/got",
    }
}

if __name__ == "__main__":
    choose_config = config["got"]

    target_input_glyph_table_dir = choose_config["table_dir"]

    image_files = sorted(
        Path(target_input_glyph_table_dir).glob('*.[Pp][Nn][Gg]'),
        key=lambda x: x.name.lower()
    )
    table_files = sorted(
        Path("glyphTable/characterTable/").glob('*.[Tt][Xx][Tt]'),
        key=lambda x: x.name.lower()
    )

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

    render_chars_to_images(
        image_files, table_files, chars,
        choose_config["font"],
        output_dir=choose_config["output_dir"],
        mapping_output="glyphTable/character-mapping.json"
    )
