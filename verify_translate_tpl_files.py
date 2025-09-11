from pathlib import Path


def compare_file_sizes(origin_path: str, translated_path: str):
    origin_dir = Path(origin_path)
    trans_dir = Path(translated_path)

    if not origin_dir.exists() or not trans_dir.exists():
        print("❌ 路径不存在，请检查输入路径")
        return

    mismatched = []
    matched = []

    for file in sorted(origin_dir.glob("*.tpl"), key=lambda x: x.name.lower()):
        trans_file = trans_dir / file.name
        if not trans_file.exists():
            print(f"⚠️ 缺少翻译文件: {trans_file}")
            continue

        origin_size = file.stat().st_size
        trans_size = trans_file.stat().st_size

        if origin_size == trans_size:
            matched.append((file.name, origin_size))
        else:
            mismatched.append((file.name, origin_size, trans_size))

    # 输出结果
    print("\n=== 比对结果 ===")
    print(f"✅ 完全一致: {len(matched)} 个")
    for name, size in matched:
        print(f"  {name} -> {size} bytes")

    print(f"\n❌ 不一致: {len(mismatched)} 个")
    for name, o_size, t_size in mismatched:
        print(f"  {name}: 原始={o_size}, 翻译={t_size}, 差值={t_size - o_size}")


if __name__ == "__main__":
    origin_path = "game_script/psv"
    translated_path = "game_script/psv/translated"
    compare_file_sizes(origin_path, translated_path)
