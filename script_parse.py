import re
from pathlib import Path

output_dir = "extract_text"


def parse_ons_script(ons_script_path):
    file_path = Path(ons_script_path)

    output_path = file_path.with_name(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_path / file_path.name
    output_file = open(output_file_path, "w", encoding="utf-8")

    # 全屏一行渲染的文案内容
    screen_msgs = []
    # 单独的对白文案
    says = []
    # 人物名称
    persons = []
    # 人物对白文案
    person_says = []

    # 1. b__screenmsg: 抓取第一个 "" 中的内容
    re_screen_msg = re.compile(r'^b__screenmsg.*?"(.*?)"')
    # 2. b__say: 抓取第一个 "" 中的内容
    re_say = re.compile(r'^b__say(?!_)\S*\s*"(.*?)"')
    # 3. b__say_1: 抓取两个 "" 中的内容
    re_say1 = re.compile(r'^b__say_1.*?"(.*?)".*?"(.*?)"')

    with open(file_path, "r", encoding="gb2312", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 匹配 b__screenmsg
            m1 = re_screen_msg.match(line)
            if m1:
                screen_msg = m1.group(1)
                screen_msgs.append(screen_msg)

                output_file.write(screen_msg + "\n")
                continue

            # 匹配 b__say（排除 b__say_1）
            m2 = re_say.match(line)
            if m2:
                say = m2.group(1)
                says.append(say)

                output_file.write(say + "\n")
                continue

            # 匹配 b__say_1
            m3 = re_say1.match(line)
            if m3:
                person_say = m3.group(1)
                person_name = m3.group(2)
                person_says.append(person_say)  # 对白
                persons.append(person_name)  # 人物名

                output_file.write(person_name + "\n")
                output_file.write(person_say + "\n")
                continue

    output_file.close()

    # 输出测试
    # print("screenmsg:", screen_msgs)
    # print("says:", says)
    # print("persons:", persons)
    # print("person_says:", person_says)

    # 合并所有文本
    all_text = "".join(screen_msgs + says + persons + person_says)

    # 统计字符集合
    # \u3000 是全角的空格
    unique_chars = sorted(set(all_text))

    # 输出
    print("总字符数:", len(unique_chars))
    print("字符列表:", unique_chars)


if __name__ == "__main__":
    parse_ons_script("game_script/ons_version.txt")
