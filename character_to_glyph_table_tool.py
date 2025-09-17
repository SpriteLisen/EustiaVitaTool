import os
import json
from pathlib import Path
from Constants import normal_char
from PIL import Image, ImageDraw, ImageFont


def draw_char_into_cell_mask(
        char, cell_w, cell_h, ttf_path,
        max_font_size=None, base_font_size=6, scale=10,
        mode="white", outline_width=2
):
    """返回一个 RGBA 模式 mask，支持白色透明和描边模式"""
    if max_font_size is None:
        max_font_size = min(cell_w, cell_h)

    w_big, h_big = cell_w * scale, cell_h * scale
    font_size = (max_font_size - base_font_size) * scale

    # 确保字体大小至少为8
    font_size = max(font_size, 8 * scale)

    try:
        font = ImageFont.truetype(ttf_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # 创建透明背景的大图像
    mask_big = Image.new("RGBA", (w_big, h_big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_big)

    # 获取文本边界框
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 如果文字太大，调整字体大小
    while (w > w_big or h > h_big) and font_size > 4 * scale:
        font_size -= scale
        try:
            font = ImageFont.truetype(ttf_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 计算文本位置（居中）
    x = (w_big - w) // 2 - bbox[0]
    y = (h_big - h) // 2 - bbox[1]

    if mode == "white":
        # 白色透明模式
        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))
    elif mode == "outline":
        # 描边模式：先绘制黑色描边，再绘制白色文字
        # 绘制描边（多个偏移位置）
        for dx, dy in [(-outline_width, 0), (outline_width, 0),
                       (0, -outline_width), (0, outline_width),
                       (-outline_width, -outline_width), (-outline_width, outline_width),
                       (outline_width, -outline_width), (outline_width, outline_width)]:
            draw.text((x + dx, y + dy), char, font=font, fill=(0, 0, 0, 255))
        # 绘制主体文字
        draw.text((x, y), char, font=font, fill=(255, 255, 255, 255))

    # 缩小图像
    mask = mask_big.resize((cell_w, cell_h), Image.LANCZOS)
    return mask, font


def render_chars_to_images(
        image_files,
        table_files,
        chars,
        ttf_path,
        output_dir="out",
        cols=14, rows=14,
        base_font_size=6,
        mapping_output="char_mapping.json",
        mode="white",  # 新增参数：white 或 outline
        outline_width=2  # 新增参数：描边宽度
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

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path)
        print(f"Saved {out_path}")

    # 保存映射文件
    with open(mapping_output, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)

    print(f"All images processed. Mapping saved to {mapping_output}")


# 配置选项
config = {
    "seif": {
        "table_dir": "glyphTable/type/se/seif",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/seif",
        "base_font_size": 5,
        "outline_width": 12,
        "mode": "outline",
    },
    "sein": {
        "table_dir": "glyphTable/type/se/sein",
        "font": "glyphTable/font/YanZhenQingDuoBaoTaBei.ttf",
        "output_dir": "glyphTable/modified/sein",
        "base_font_size": 5,
        "outline_width": 12,
        "mode": "white",
    },

    "marf": {
        "table_dir": "glyphTable/type/mar/marf",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marf",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "outline",
    },
    "marn": {
        "table_dir": "glyphTable/type/mar/marn",
        "font": "glyphTable/font/GongFanLiZhongYuan.ttf",
        "output_dir": "glyphTable/modified/marn",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "white",
    },

    "popf": {
        "table_dir": "glyphTable/type/pop/popf",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popf",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "outline",
    },
    "popn": {
        "table_dir": "glyphTable/type/pop/popn",
        "font": "glyphTable/font/KeAiPao-TaoZiJiu.ttf",
        "output_dir": "glyphTable/modified/popn",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "white",
    },

    "gotf": {
        "table_dir": "glyphTable/type/got/gotf",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotf",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "outline",
    },
    "gotn": {
        "table_dir": "glyphTable/type/got/gotn",
        "font": "glyphTable/font/SanJiLuoLiHei-Cu.ttf",
        "output_dir": "glyphTable/modified/gotn",
        "base_font_size": 6,
        "outline_width": 12,
        "mode": "white",
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